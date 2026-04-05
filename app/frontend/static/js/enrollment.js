const panel = document.getElementById('enrollment-panel');
let enrollmentStep = 0;  // 0=idle, 1=confirm, 2=naming

function openPanel() {
  panel.classList.add('open');
}

function closePanel() {
  panel.classList.remove('open');
  fetch('/api/enroll/cancel', { method: 'POST' }).catch(() => {});
  enrollmentStep = 0;
}

// Server triggers enrollment when an unknown face is detected
socket.on('unknown_detected', ({ face_crop_b64 }) => {
  const preview = document.getElementById('enrollment-face-preview');
  if (preview) {
    preview.innerHTML = face_crop_b64
      ? `<img src="data:image/jpeg;base64,${face_crop_b64}" style="width:100%;height:100%;object-fit:cover">`
      : '';
  }
  showStep(1);
  openPanel();
});

function showStep(step) {
  enrollmentStep = step;
  const dots = panel.querySelectorAll('.step-dot');
  dots.forEach((d, i) => {
    d.classList.toggle('active', i === step - 1);
    d.classList.toggle('done', i < step - 1);
  });

  panel.querySelector('#step-1').style.display = step === 1 ? 'flex' : 'none';
  panel.querySelector('#step-2').style.display = step === 2 ? 'flex' : 'none';
}

// Step 1: "Add Person" clicked — go straight to name input
panel.querySelector('#btn-add').addEventListener('click', () => {
  showStep(2);
  const input = panel.querySelector('#enroll-name-input');
  if (input) {
    input.value = '';
    input.focus();
  }
});

// Step 2: Confirm name and enroll
panel.querySelector('#btn-confirm').addEventListener('click', () => {
  const input = panel.querySelector('#enroll-name-input');
  const name = (input ? input.value : '').trim();
  if (!name) return;

  fetch('/api/enroll/confirm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name })
  })
    .then(() => {
      showToast(`${name} added to FaceVault`, 'success');
      closePanel();
    })
    .catch(() => {});
});

panel.querySelector('#btn-dismiss').addEventListener('click', closePanel);
panel.querySelector('#btn-cancel').addEventListener('click', closePanel);

window.triggerEnrollment = openPanel;
window.closeEnrollmentPanel = closePanel;
