const socket = io();
let cameraActive = false;

// ── HUD state ─────────────────────────────────────────────
let _lastFrameTs = 0, _fpsDisplay = '--', _scanCount = 0;
const hudFps   = document.getElementById('hud-fps');
const hudScans = document.getElementById('hud-scans');
const hudStats = document.getElementById('hud-stats');

// ── Typewriter for inactive state ─────────────────────────
const TYPEWRITER_PHRASES = [
  'INITIALIZING NEURAL SCAN…',
  'AWAITING BIOMETRIC INPUT…',
  'FACE RECOGNITION READY…',
  'SYSTEM STANDBY…',
];
let _twPhrase = 0, _twChar = 0, _twDeleting = false, _twTimer = null;
const _twEl = document.getElementById('typewriter-text');

function _typewriterTick() {
  if (!_twEl) return;
  const phrase = TYPEWRITER_PHRASES[_twPhrase];
  if (_twDeleting) {
    _twChar--;
    _twEl.textContent = phrase.slice(0, _twChar);
    if (_twChar === 0) {
      _twDeleting = false;
      _twPhrase = (_twPhrase + 1) % TYPEWRITER_PHRASES.length;
      _twTimer = setTimeout(_typewriterTick, 600);
      return;
    }
  } else {
    _twChar++;
    _twEl.textContent = phrase.slice(0, _twChar);
    if (_twChar === phrase.length) {
      _twDeleting = true;
      _twTimer = setTimeout(_typewriterTick, 2000);
      return;
    }
  }
  _twTimer = setTimeout(_typewriterTick, _twDeleting ? 40 : 80);
}
_typewriterTick();

// ── Config sync ──────────────────────────────────────────
fetch('/api/config')
  .then(r => r.json())
  .then(cfg => {
    const slider = document.getElementById('threshold-slider');
    const val    = document.getElementById('threshold-value');
    if (cfg.threshold) {
      slider.value      = parseFloat(cfg.threshold).toFixed(2);
      val.textContent   = parseFloat(cfg.threshold).toFixed(2);
    }
    if (cfg.demo_mode) document.getElementById('demo-banner').classList.add('visible');
  })
  .catch(() => {});

// ── DOM refs ─────────────────────────────────────────────
const cameraBtn       = document.getElementById('camera-toggle-btn');
const btnLabel        = cameraBtn.querySelector('.btn-label');
const videoImg        = document.getElementById('video-stream');
const inactiveOverlay = document.getElementById('camera-inactive');
const scanLine        = document.querySelector('.scan-line');
const liveBadge       = document.getElementById('live-badge');

// ── Camera toggle ────────────────────────────────────────
cameraBtn.addEventListener('click', () => {
  if (cameraActive) {
    socket.emit('stop_camera');
  } else {
    socket.emit('start_camera');
    videoImg.src = '/video_feed?' + Date.now();
    videoImg.style.display = 'block';
    inactiveOverlay.style.display = 'none';
  }
});

// ── Camera status ────────────────────────────────────────
socket.on('camera_status', ({ active }) => {
  cameraActive = !!active;
  if (active) {
    btnLabel.textContent = 'STOP CAMERA';
    cameraBtn.classList.add('active');
    inactiveOverlay.style.display = 'none';
    scanLine.classList.add('active');
    liveBadge.classList.add('active');
    if (hudStats) hudStats.style.opacity = '1';
    _scanCount = 0;
    if (hudScans) hudScans.textContent = '0';
    if (hudFps) hudFps.textContent = '--';
  } else {
    btnLabel.textContent = 'ACTIVATE CAMERA';
    cameraBtn.classList.remove('active');
    inactiveOverlay.style.display = 'flex';
    videoImg.style.display = 'none';
    scanLine.classList.remove('active');
    liveBadge.classList.remove('active');
    if (hudStats) hudStats.style.opacity = '0';
  }
});

// ── Detection results → canvas + HUD ─────────────────────
socket.on('detection_result', ({ faces }) => {
  // Only draw bounding boxes for faces above threshold (is_known = true)
  const recognised = (faces || []).filter(f => f.is_known);
  drawDetections(recognised);

  // FPS
  const now = performance.now();
  if (_lastFrameTs) {
    const fps = Math.round(1000 / (now - _lastFrameTs));
    _fpsDisplay = Math.min(fps, 30);
    if (hudFps) hudFps.textContent = _fpsDisplay;
  }
  _lastFrameTs = now;

  // Scan count (each frame = 1 scan)
  _scanCount++;
  if (hudScans) hudScans.textContent = _scanCount;
});

// ── Known face recognised toast ──────────────────────────
socket.on('known_detected', ({ name, confidence }) => {
  const pct = Math.round((confidence || 0) * 100);
  showToast(`✓  ${name}  ·  ${pct}%`, 'success', 4500);
});

// ── Canvas overlay ───────────────────────────────────────
function drawDetections(faces) {
  const canvas = document.getElementById('detection-overlay');
  const img    = document.getElementById('video-stream');
  const rect   = img.getBoundingClientRect();
  canvas.width  = rect.width;
  canvas.height = rect.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  faces.forEach(({ name, confidence, bbox, is_known }) => {
    if (!bbox) return;
    const color  = is_known ? '#4ADE80' : '#FB923C';
    const glow   = is_known ? 'rgba(74,222,128,0.5)' : 'rgba(251,146,60,0.5)';
    const { x, y, w, h } = bbox;
    const scX = canvas.width  / (img.naturalWidth  || canvas.width);
    const scY = canvas.height / (img.naturalHeight || canvas.height);
    const sx = x * scX, sy = y * scY, sw = w * scX, sh = h * scY;

    ctx.save();
    ctx.shadowColor = glow; ctx.shadowBlur = 18;

    // Bounding box
    ctx.strokeStyle = color; ctx.lineWidth = 1.8;
    ctx.strokeRect(sx, sy, sw, sh);

    // Corner accents
    const c = 16; ctx.lineWidth = 3;
    [[sx,sy,1,1],[sx+sw,sy,-1,1],[sx,sy+sh,1,-1],[sx+sw,sy+sh,-1,-1]].forEach(([px,py,dx,dy]) => {
      ctx.beginPath();
      ctx.moveTo(px, py + dy * c); ctx.lineTo(px, py); ctx.lineTo(px + dx * c, py);
      ctx.stroke();
    });

    // Label pill
    ctx.shadowBlur = 0;
    const label    = `${is_known ? name : 'Unknown'}  ${Math.round(confidence * 100)}%`;
    const fontSize = 11;
    ctx.font       = `600 ${fontSize}px 'JetBrains Mono', monospace`;
    const tw = ctx.measureText(label).width;
    const ph = fontSize + 10, pw = tw + 18;
    const px = sx, py2 = Math.max(4, sy - ph - 6);

    // Pill background
    ctx.fillStyle = is_known ? 'rgba(74,222,128,0.18)' : 'rgba(251,146,60,0.18)';
    roundRect(ctx, px, py2, pw, ph, 5); ctx.fill();
    ctx.strokeStyle = color; ctx.lineWidth = 1;
    roundRect(ctx, px, py2, pw, ph, 5); ctx.stroke();
    ctx.fillStyle = color;
    ctx.fillText(label, px + 9, py2 + fontSize + 2);

    ctx.restore();
  });
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y);
  ctx.arcTo(x+w,y,x+w,y+r,r); ctx.lineTo(x+w,y+h-r);
  ctx.arcTo(x+w,y+h,x+w-r,y+h,r); ctx.lineTo(x+r,y+h);
  ctx.arcTo(x,y+h,x,y+h-r,r); ctx.lineTo(x,y+r);
  ctx.arcTo(x,y,x+r,y,r); ctx.closePath();
}

// ── Threshold slider ─────────────────────────────────────
const slider     = document.getElementById('threshold-slider');
const sliderVal  = document.getElementById('threshold-value');
slider.addEventListener('input', () => {
  const v = parseFloat(slider.value) || 0.65;
  sliderVal.textContent = v.toFixed(2);
  socket.emit('set_threshold', v);
});

// ── Toast helper ─────────────────────────────────────────
function showToast(message, type = 'info', duration = 3500) {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className   = `toast ${type}`;
  t.textContent = message;
  c.appendChild(t);
  setTimeout(() => t.remove(), duration);
}
window.showToast = showToast;
