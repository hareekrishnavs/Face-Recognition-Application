function loadGallery() {
  fetch('/api/known_users')
    .then(r => r.json())
    .then(users => {
      const gallery = document.getElementById('gallery-users');
      if (!gallery) return;
      gallery.innerHTML = '';
      users.forEach(u => gallery.appendChild(createUserCard(u)));

      const countLabel = document.getElementById('user-count');
      if (countLabel) {
        countLabel.textContent = `${users.length} enrolled`;
      }
    })
    .catch(() => {});
}

function createUserCard({ name, sample_image_b64, detection_count }) {
  const card = document.createElement('div');
  card.className = 'user-card';
  const hasImage = !!sample_image_b64;

  card.innerHTML = `
    <div class="user-avatar">
      ${hasImage
        ? `<img src="data:image/jpeg;base64,${sample_image_b64}" alt="${name}">`
        : `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted);font-size:28px">◈</div>`
      }
    </div>
    ${detection_count > 0 ? `<div class="user-badge">${detection_count}</div>` : ''}
    <span class="user-name">${name}</span>
  `;

  // Holographic tilt effect
  card.addEventListener('mousemove', e => {
    const r = card.getBoundingClientRect();
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const rx = ((e.clientY - cy) / (r.height / 2)) * -12;
    const ry = ((e.clientX - cx) / (r.width  / 2)) *  12;
    card.style.transform = `perspective(600px) rotateX(${rx}deg) rotateY(${ry}deg) scale(1.04)`;
  });
  card.addEventListener('mouseleave', () => {
    card.style.transform = '';
  });

  return card;
}

function createAddCard() {
  const card = document.createElement('div');
  card.className = 'user-card user-card-add';
  card.innerHTML = `
    <div class="user-avatar">＋</div>
    <span class="user-name">Add</span>
  `;
  card.addEventListener('click', () => window.triggerEnrollment());
  return card;
}

function removeUser(name) {
  if (!confirm(`Remove "${name}" from the system?`)) return;
  fetch('/api/remove_user', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name })
  }).then(() => {
    showToast(`${name} removed`, 'warning');
    loadGallery();
  }).catch(() => {});
}

socket.on('enrollment_complete', () => loadGallery());
socket.on('user_removed', () => loadGallery());

loadGallery();
