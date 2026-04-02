/**
 * dashboard.js — Live stats cards and scrollable activity log.
 *
 * Exposes:
 *   window._updateStats(detectedToday, sessionsToday)
 *   window._setKnownCount(n)
 *   window._prependLogEntry({timestamp, name, confidence, is_known})
 */

'use strict';

// ── Stats cards ───────────────────────────────────────────────────────────────
const _detectedEl = document.getElementById('stat-detected');
const _sessionsEl = document.getElementById('stat-sessions');

window._updateStats = function(detectedToday, sessionsToday) {
  _animateCounter(_detectedEl, parseInt(detectedToday) || 0);
  _animateCounter(_sessionsEl, parseInt(sessionsToday) || 0);
};

window._setKnownCount = function(n) {
  // reserved for future "Known users" stat card
};

function _animateCounter(el, target) {
  const current = parseInt(el.textContent) || 0;
  if (current === target) return;

  // Quick flash highlight
  el.classList.add('highlight');
  el.textContent = target;
  setTimeout(() => el.classList.remove('highlight'), 600);
}

// ── Activity log ──────────────────────────────────────────────────────────────
const _activityList = document.getElementById('activity-list');
const LOG_MAX       = 100;

/**
 * Build a confidence-bar class from a 0–1 confidence value.
 */
function _barClass(conf) {
  if (conf >= 0.85) return 'bar-high';
  if (conf >= 0.65) return 'bar-mid';
  return 'bar-low';
}

/**
 * Format a UTC ISO timestamp to HH:MM:SS local time.
 */
function _fmtTime(iso) {
  if (!iso) return '--:--:--';
  try {
    return new Date(iso).toLocaleTimeString('en-US', { hour12: false });
  } catch {
    return '--:--:--';
  }
}

/**
 * Prepend a new detection entry at the top of the activity log.
 * Called by app.js when a 'log_entry' socket event arrives.
 */
window._prependLogEntry = function(det) {
  const isKnown = !!det.is_known;
  const conf    = parseFloat(det.confidence) || 0;
  const pct     = (conf * 100).toFixed(1);
  const cls     = isKnown ? 'known' : 'unknown';
  const barCls  = _barClass(conf);
  const barW    = Math.round(conf * 100);
  const time    = _fmtTime(det.timestamp);
  const name    = det.name || 'Unknown';

  const el = document.createElement('div');
  el.className = `log-entry ${cls}`;
  el.innerHTML = `
    <span class="log-dot"></span>
    <span class="log-name">${_esc(name)}</span>
    <span class="log-conf-bar-wrap">
      <span class="log-conf-bar">
        <span class="log-conf-fill ${barCls}" style="width:${barW}%"></span>
      </span>
      <span class="log-conf-pct">${pct}%</span>
    </span>
    <span class="log-time">${time}</span>
    ${!isKnown ? '<button class="log-add-btn" title="Enroll this person">+Add</button>' : ''}
  `;

  // "+Add" button opens enrollment panel for this unknown
  if (!isKnown) {
    el.querySelector('.log-add-btn').addEventListener('click', () => {
      if (typeof window._enrollManual === 'function') {
        _enrollManual();
      }
    });
  }

  _activityList.prepend(el);

  // Trim to max
  while (_activityList.children.length > LOG_MAX) {
    _activityList.lastChild.remove();
  }
};

/** Minimal HTML escape */
function _esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Seed log from HTTP on load ─────────────────────────────────────────────────
(async function seedLog() {
  try {
    const res  = await fetch('/api/activity_log');
    const rows = await res.json();

    // API returns newest-first; display them oldest-first so prepend works
    const slice = rows.slice(0, 50).reverse();
    slice.forEach(r => _prependLogEntry({
      timestamp:  r.timestamp,
      name:       r.name,
      confidence: r.confidence,
      is_known:   r.is_known === 1 || r.is_known === true,
    }));
  } catch (e) {
    console.warn('[Dashboard] Could not seed log:', e);
  }
})();

// ── Seed stats from HTTP on load ──────────────────────────────────────────────
(async function seedStats() {
  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();
    _updateStats(data.detected_today ?? 0, data.sessions_today ?? 0);
    _setModelStatus(data.model_loaded, data.demo_mode);
    _setThresholdUI(data.threshold ?? 0.65);
  } catch (e) {
    console.warn('[Dashboard] Could not seed stats:', e);
  }
})();
