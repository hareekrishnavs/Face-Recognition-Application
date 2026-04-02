/**
 * enrollment.js — Slide-in enrollment panel state machine.
 *
 * Steps:
 *   1 → Unknown detected / manual add — "Verify & Add" or "Dismiss"
 *   2 → Verification in progress      — 15-second countdown
 *   3 → Verified — enter name         — "Confirm & Save"
 *
 * Exposes:
 *   window._openEnrollPanel(cropB64)  — open panel (null = manual, no crop)
 *   window._enrollOnUnknown(cropB64, ts) — called by app.js on 'unknown_detected'
 *   window._enrollOnVerified(verifier, cropB64) — called by app.js on 'verification_approved'
 *   window._enrollOnTimeout()         — called by app.js on 'verification_timeout'
 */

'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const _panel       = document.getElementById('enrollment-panel');
const _previewImg  = document.getElementById('enroll-preview-img');
const _step1       = document.getElementById('enroll-step-1');
const _step2       = document.getElementById('enroll-step-2');
const _step3       = document.getElementById('enroll-step-3');
const _verifyDot   = document.getElementById('verify-status-dot');
const _verifyText  = document.getElementById('verify-status-text');
const _verifyTimer = document.getElementById('verify-timer');
const _verifierName= document.getElementById('verifier-name');
const _nameInput   = document.getElementById('enroll-name-input');

// ── Internal state ────────────────────────────────────────────────────────────
let _timerInterval = null;
let _timeLeft      = 15;
let _isOpen        = false;

// ── Step helpers ──────────────────────────────────────────────────────────────
function _showStep(n) {
  [_step1, _step2, _step3].forEach((s, i) => {
    s.classList.toggle('active', i + 1 === n);
  });
}

function _setPreview(b64) {
  if (b64) {
    _previewImg.src = 'data:image/jpeg;base64,' + b64;
    _previewImg.style.display = 'block';
  } else {
    _previewImg.src = '';
    _previewImg.style.display = 'none';
  }
}

// ── Panel open / close ────────────────────────────────────────────────────────
window._openEnrollPanel = function(cropB64) {
  _stopTimer();
  _setPreview(cropB64);
  _showStep(1);
  _nameInput.value = '';
  _panel.classList.add('open');
  _isOpen = true;
};

function _closePanel() {
  _panel.classList.remove('open');
  _isOpen = false;
  _stopTimer();
  // Tell server to reset enrollment state
  fetch('/api/enroll/reject', { method: 'POST' }).catch(() => {});
}

document.getElementById('enroll-close-btn').addEventListener('click', _closePanel);

// ── Step 1 buttons ────────────────────────────────────────────────────────────
document.getElementById('btn-verify-start').addEventListener('click', () => {
  // Tell server to start watching for a known face
  FV.emit('start_verification');
  _startVerification();
});

document.getElementById('btn-enroll-dismiss').addEventListener('click', _closePanel);

// ── Step 2: verification ──────────────────────────────────────────────────────
function _startVerification() {
  _showStep(2);
  _verifyDot.className = 'waiting';
  _verifyText.textContent = 'Scanning for known face…';
  _startTimer();
}

function _startTimer() {
  _stopTimer();
  _timeLeft = 15;
  _verifyTimer.textContent = _timeLeft + 's';

  _timerInterval = setInterval(() => {
    _timeLeft--;
    _verifyTimer.textContent = _timeLeft + 's';
    if (_timeLeft <= 0) _stopTimer();
  }, 1000);
}

function _stopTimer() {
  if (_timerInterval) {
    clearInterval(_timerInterval);
    _timerInterval = null;
  }
}

document.getElementById('btn-verify-cancel').addEventListener('click', () => {
  _stopTimer();
  _showStep(1);
  FV.emit('enrollment_response', { approved: false });
});

// ── Step 3: name input ────────────────────────────────────────────────────────
document.getElementById('btn-enroll-confirm').addEventListener('click', _submitEnrollment);
document.getElementById('btn-enroll-cancel').addEventListener('click', _closePanel);

_nameInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') _submitEnrollment();
});

async function _submitEnrollment() {
  const name = _nameInput.value.trim();
  if (!name) {
    toast('Enter a name first', 'error');
    _nameInput.focus();
    return;
  }

  const btn = document.getElementById('btn-enroll-confirm');
  btn.disabled    = true;
  btn.textContent = 'Saving…';

  try {
    const res  = await fetch('/api/enroll/approve', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ name }),
    });
    const data = await res.json();

    if (res.ok) {
      toast(`✓ ${data.enrolled} saved (${data.images} images)`, 'success', 4000);
      _panel.classList.remove('open');
      _isOpen = false;
      if (typeof window._galleryRefresh === 'function') _galleryRefresh();
    } else {
      toast(data.error || 'Enrollment failed', 'error');
    }
  } catch {
    toast('Network error', 'error');
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Confirm & Save';
  }
}

// ── Called from app.js ────────────────────────────────────────────────────────

/**
 * Socket event: unknown face triggered enrollment.
 * Only opens the panel if it isn't already open (avoid interrupting a flow).
 */
window._enrollOnUnknown = function(cropB64, ts) {
  if (_isOpen) return;
  _openEnrollPanel(cropB64);
};

/**
 * Socket event: a known face has appeared during the verification window.
 */
window._enrollOnVerified = function(verifier, cropB64) {
  _stopTimer();
  _verifyDot.className    = 'approved';
  _verifyText.textContent = `Approved by ${verifier}`;

  if (cropB64) _setPreview(cropB64);
  _verifierName.textContent = verifier;

  // Brief pause so user can see the "Approved" indicator before moving to step 3
  setTimeout(() => _showStep(3), 800);
};

/**
 * Socket event: verification window expired without a known face.
 */
window._enrollOnTimeout = function() {
  _stopTimer();
  _verifyDot.className    = 'timeout';
  _verifyText.textContent = 'Timeout — no known face found';
  _verifyTimer.textContent = '0s';

  toast('Verification timed out — try again', 'warn');
  // Return to step 1 so user can retry
  setTimeout(() => _showStep(1), 1800);
};
