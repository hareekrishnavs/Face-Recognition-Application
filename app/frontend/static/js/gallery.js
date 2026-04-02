/**
 * gallery.js — Known users gallery in the footer bar.
 *
 * Exposes:
 *   window._galleryRefresh()   — re-fetches /api/known_users and redraws cards
 *   window._enrollManual()     — open enrollment panel without an unknown trigger
 */

'use strict';

const _galleryCards = document.getElementById('gallery-cards');

// ── Render cards ──────────────────────────────────────────────────────────────
async function _galleryLoad() {
  try {
    const res   = await fetch('/api/known_users');
    const users = await res.json();
    _renderGallery(users);
  } catch (e) {
    console.warn('[Gallery] Load failed:', e);
  }
}

function _renderGallery(users) {
  _galleryCards.innerHTML = '';

  if (!users || !users.length) {
    const empty = document.createElement('span');
    empty.style.cssText = 'font-family:var(--font-mono);font-size:11px;color:var(--text-muted);padding:0 8px;';
    empty.textContent   = 'No enrolled persons';
    _galleryCards.appendChild(empty);
    return;
  }

  users.forEach(u => {
    const card = document.createElement('div');
    card.className = 'user-card';

    if (u.sample_b64) {
      card.innerHTML = `
        <img class="user-avatar" src="data:image/jpeg;base64,${u.sample_b64}" alt="${_esc(u.name)}" />
        <span class="user-name" title="${_esc(u.name)}">${_esc(u.name)}</span>
        <span class="user-remove" title="Remove ${_esc(u.name)}">✕</span>
      `;
    } else {
      const initial = u.name.charAt(0).toUpperCase();
      card.innerHTML = `
        <div class="user-avatar-initial">${_esc(initial)}</div>
        <span class="user-name" title="${_esc(u.name)}">${_esc(u.name)}</span>
        <span class="user-remove" title="Remove ${_esc(u.name)}">✕</span>
      `;
    }

    // Remove button
    card.querySelector('.user-remove').addEventListener('click', (e) => {
      e.stopPropagation();
      _confirmRemove(u.name);
    });

    _galleryCards.appendChild(card);
  });
}

// ── Remove person ─────────────────────────────────────────────────────────────
let _pendingRemoveName = null;
const _confirmOverlay   = document.getElementById('confirm-overlay');
const _confirmNameEl    = document.getElementById('confirm-name');
const _confirmOkBtn     = document.getElementById('confirm-ok-btn');
const _confirmCancelBtn = document.getElementById('confirm-cancel-btn');

function _confirmRemove(name) {
  _pendingRemoveName      = name;
  _confirmNameEl.textContent = name;
  _confirmOverlay.classList.add('open');
}

_confirmOkBtn.addEventListener('click', async () => {
  if (!_pendingRemoveName) return;
  const name = _pendingRemoveName;
  _confirmOverlay.classList.remove('open');
  _pendingRemoveName = null;

  try {
    const res = await fetch('/api/remove_user', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ name }),
    });
    if (res.ok) {
      toast(`Removed ${name}`, 'warn');
      _galleryLoad();
    } else {
      toast('Remove failed', 'error');
    }
  } catch {
    toast('Network error', 'error');
  }
});

_confirmCancelBtn.addEventListener('click', () => {
  _confirmOverlay.classList.remove('open');
  _pendingRemoveName = null;
});

// Close on backdrop click
_confirmOverlay.addEventListener('click', (e) => {
  if (e.target === _confirmOverlay) {
    _confirmOverlay.classList.remove('open');
    _pendingRemoveName = null;
  }
});

// ── Public API ────────────────────────────────────────────────────────────────
window._galleryRefresh = _galleryLoad;

window._enrollManual = function() {
  // Open the enrollment panel in step-1 without a specific face crop
  if (typeof window._openEnrollPanel === 'function') {
    _openEnrollPanel(null);
  }
};

// "Add" button in footer
document.getElementById('btn-add-user').addEventListener('click', () => {
  window._enrollManual();
});

// ── HTML escape util ──────────────────────────────────────────────────────────
function _esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Initial load ──────────────────────────────────────────────────────────────
_galleryLoad();
