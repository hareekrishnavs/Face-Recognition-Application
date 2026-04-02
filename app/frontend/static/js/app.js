/**
 * app.js — Core FaceVault application logic.
 *
 * Responsibilities:
 *   - SocketIO connection + event routing
 *   - MJPEG stream display
 *   - Canvas bounding-box overlay
 *   - Threshold slider
 *   - Clock
 *   - Toast notifications
 *   - Shared state object used by all modules
 */

'use strict';

// ── Shared global state ───────────────────────────────────────────────────────
window.FV = {
  socket:        null,
  threshold:     0.65,
  modelLoaded:   false,
  demoMode:      false,
  lastDetections: [],

  /** Emit a SocketIO event safely. */
  emit(event, data) {
    if (this.socket) this.socket.emit(event, data);
  },
};

// ── Clock ─────────────────────────────────────────────────────────────────────
(function initClock() {
  const el = document.getElementById('clock');
  function tick() {
    el.textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
  }
  tick();
  setInterval(tick, 1000);
})();

// ── Toast ─────────────────────────────────────────────────────────────────────
/**
 * Show a toast notification.
 * @param {string} msg
 * @param {'info'|'success'|'warn'|'error'} type
 * @param {number} ms — auto-dismiss delay
 */
window.toast = function toast(msg, type = 'info', ms = 3200) {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  container.appendChild(el);

  setTimeout(() => {
    el.classList.add('toast-out');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, ms);
};

// ── Threshold slider ──────────────────────────────────────────────────────────
(function initThreshold() {
  const slider = document.getElementById('threshold-slider');
  const valEl  = document.getElementById('threshold-val');
  let debounce = null;

  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    valEl.textContent = v.toFixed(2);
    FV.threshold = v;

    clearTimeout(debounce);
    debounce = setTimeout(() => {
      fetch('/api/threshold', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ value: v }),
      });
      FV.emit('set_threshold', v);
    }, 400);
  });

  window._setThresholdUI = function(v) {
    slider.value      = v;
    valEl.textContent = parseFloat(v).toFixed(2);
    FV.threshold      = parseFloat(v);
  };
})();

// ── Model status badge ────────────────────────────────────────────────────────
window._setModelStatus = function(loaded, demo) {
  const el = document.getElementById('model-status');
  const db = document.getElementById('demo-badge');
  if (demo) {
    el.className  = 'unloaded';
    el.textContent = 'DEMO MODE';
    db.classList.add('visible');
  } else if (loaded) {
    el.className  = 'loaded';
    el.textContent = 'MODEL READY';
    db.classList.remove('visible');
  } else {
    el.className  = 'unloaded';
    el.textContent = 'NO MODEL';
    db.classList.remove('visible');
  }
  FV.modelLoaded = loaded;
  FV.demoMode    = demo;
};

// ── MJPEG stream ──────────────────────────────────────────────────────────────
(function initStream() {
  const img = document.getElementById('video-stream');
  img.onerror = () => {
    // Retry after 2 seconds if the MJPEG stream drops
    setTimeout(() => { img.src = '/video_feed?' + Date.now(); }, 2000);
  };
})();

// ── Canvas overlay for bounding boxes ────────────────────────────────────────
(function initOverlay() {
  const canvas = document.getElementById('detection-overlay');
  const ctx    = canvas.getContext('2d');
  const img    = document.getElementById('video-stream');
  const cont   = document.getElementById('video-container');

  function resizeCanvas() {
    canvas.width  = cont.offsetWidth;
    canvas.height = cont.offsetHeight;
  }

  const ro = new ResizeObserver(resizeCanvas);
  ro.observe(cont);
  resizeCanvas();

  /**
   * Draw bounding boxes on the canvas.
   * Server sends bbox in original frame coordinates; we scale to canvas size.
   * @param {Array<{name,confidence,bbox,is_known}>} faces
   * @param {number} frameW
   * @param {number} frameH
   */
  window._drawDetections = function(faces, frameW, frameH) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!faces || !faces.length) return;

    // The <img> element uses object-fit: contain — compute letter-box offset
    const dispW = cont.offsetWidth;
    const dispH = cont.offsetHeight;

    const scale  = Math.min(dispW / frameW, dispH / frameH);
    const offX   = (dispW - frameW * scale) / 2;
    const offY   = (dispH - frameH * scale) / 2;

    faces.forEach(f => {
      const [fx, fy, fw, fh] = f.bbox;
      const x = offX + fx * scale;
      const y = offY + fy * scale;
      const w = fw * scale;
      const h = fh * scale;

      const color = f.is_known ? '#00E5A0' : '#FF6B35';
      const glow  = f.is_known ? 'rgba(0,229,160,0.25)' : 'rgba(255,107,53,0.25)';

      // Glow fill
      ctx.fillStyle = glow;
      ctx.fillRect(x, y, w, h);

      // Border
      ctx.strokeStyle = color;
      ctx.lineWidth   = 1.5;
      ctx.strokeRect(x, y, w, h);

      // Corner brackets
      const cs = 10;
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2;
      [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([cx, cy]) => {
        const dx = cx === x ? 1 : -1;
        const dy = cy === y ? 1 : -1;
        ctx.beginPath();
        ctx.moveTo(cx + dx * cs, cy);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx, cy + dy * cs);
        ctx.stroke();
      });

      // Label background
      const conf   = (f.confidence * 100).toFixed(1) + '%';
      const label  = f.name + '  ' + conf;
      const fsize  = Math.max(10, Math.min(13, w * 0.12));
      ctx.font     = `500 ${fsize}px "DM Mono", monospace`;
      const tw     = ctx.measureText(label).width;
      const pad    = 5;
      const lh     = fsize + pad * 2;
      const ly     = y > lh ? y - lh : y + h;

      ctx.fillStyle = color;
      ctx.fillRect(x, ly, tw + pad * 2, lh);

      // Label text
      ctx.fillStyle = '#000';
      ctx.fillText(label, x + pad, ly + fsize + pad * 0.5);
    });
  };
})();

// ── SocketIO connection ───────────────────────────────────────────────────────
(function initSocket() {
  const socket = io({ transports: ['websocket', 'polling'] });
  FV.socket = socket;

  socket.on('connect', () => {
    console.log('[FaceVault] Socket connected:', socket.id);
  });

  socket.on('disconnect', () => {
    console.log('[FaceVault] Socket disconnected');
  });

  // Server sends initial state on connect
  socket.on('init', (data) => {
    _setThresholdUI(data.threshold ?? 0.65);
    _setModelStatus(data.model_loaded, data.demo_mode);

    if (typeof window._updateStats === 'function') {
      _updateStats(data.detected_today ?? 0, data.sessions_today ?? 0);
    }
    if (typeof window._setKnownCount === 'function') {
      _setKnownCount(data.known_count ?? 0);
    }
  });

  // Real-time face detection
  socket.on('detection', (data) => {
    FV.lastDetections = data.faces || [];
    _drawDetections(data.faces || [], data.frame_w || 640, data.frame_h || 480);
  });

  // Activity log entry
  socket.on('log_entry', (data) => {
    if (typeof window._prependLogEntry === 'function') {
      _prependLogEntry(data);
    }
  });

  // Stats update
  socket.on('stats_update', (data) => {
    if (typeof window._updateStats === 'function') {
      _updateStats(data.detected_today, data.sessions_today);
    }
  });

  // Threshold acknowledged
  socket.on('threshold_updated', (data) => {
    _setThresholdUI(data.threshold);
  });

  // Unknown person detected — kick off enrollment flow
  socket.on('unknown_detected', (data) => {
    if (typeof window._enrollOnUnknown === 'function') {
      _enrollOnUnknown(data.face_crop_b64, data.timestamp);
    }
  });

  // Verification approved by known face
  socket.on('verification_approved', (data) => {
    if (typeof window._enrollOnVerified === 'function') {
      _enrollOnVerified(data.verifier, data.face_crop_b64);
    }
  });

  // Verification timed out
  socket.on('verification_timeout', () => {
    if (typeof window._enrollOnTimeout === 'function') {
      _enrollOnTimeout();
    }
  });

  // Enrollment complete
  socket.on('enrollment_complete', (data) => {
    toast(`✓ ${data.name} enrolled (${data.image_count} images)`, 'success', 4000);
    if (typeof window._galleryRefresh === 'function') {
      _galleryRefresh();
    }
  });
})();

// ── Export CSV ────────────────────────────────────────────────────────────────
document.getElementById('btn-export-csv').addEventListener('click', () => {
  window.location.href = '/api/export_csv';
});
