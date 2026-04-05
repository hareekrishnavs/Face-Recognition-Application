const activityList = document.getElementById('activity-list');

// Clear activity log on every page load — fresh session on each reload
fetch('/api/activity_log/clear', { method: 'POST' }).catch(() => {});

// New entries via socket
socket.on('log_entry', addLogEntry);

function addLogEntry(entry) {
  const { timestamp, name, confidence, is_known } = entry;
  const el = document.createElement('div');
  el.className = `log-entry ${is_known ? 'known' : 'unknown'}`;

  const pct = Math.round(confidence * 100);
  const barColor = pct >= 85 ? '#00E5A0' : pct >= 65 ? '#00D9FF' : '#FF6B35';
  const timeStr = timestamp && timestamp.includes('T')
    ? new Date(timestamp).toLocaleTimeString('en-CA', { hour12: false })
    : (timestamp || '').split(' ')[1] || '';

  el.innerHTML = `
    <div class="log-dot"></div>
    <span class="log-name">${is_known ? name : 'Unknown'}</span>
    <div class="log-confidence-bar">
      <div class="log-confidence-fill" style="width:${pct}%;background:${barColor};box-shadow:0 0 6px ${barColor}"></div>
    </div>
    <span style="color:${barColor};font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;white-space:nowrap">${pct}%</span>
    <span class="log-timestamp">${timeStr}</span>
  `;

  // Remove empty state if present
  const empty = activityList.querySelector('.log-empty');
  if (empty) empty.remove();

  activityList.appendChild(el);
  while (activityList.children.length > 100) {
    activityList.removeChild(activityList.firstChild);
  }
  activityList.scrollTop = activityList.scrollHeight;
}

// Export CSV
const exportBtn = document.getElementById('export-csv-btn');
if (exportBtn) {
  exportBtn.addEventListener('click', () => {
    window.location.href = '/api/activity_log/export';
  });
}
