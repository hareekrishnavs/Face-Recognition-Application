/* ── Neural Network Particle Background ─────────────────── */
(function () {
  const canvas = document.getElementById('neural-bg');
  const ctx    = canvas.getContext('2d');
  const COLORS = ['rgba(139,92,246,', 'rgba(34,211,238,', 'rgba(244,114,182,'];
  let nodes = [];

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    init();
  }

  function init() {
    const count = Math.floor((canvas.width * canvas.height) / 18000);
    nodes = Array.from({ length: count }, () => ({
      x:   Math.random() * canvas.width,
      y:   Math.random() * canvas.height,
      vx:  (Math.random() - 0.5) * 0.35,
      vy:  (Math.random() - 0.5) * 0.35,
      r:   Math.random() * 1.5 + 0.5,
      col: COLORS[Math.floor(Math.random() * COLORS.length)],
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const n = nodes;

    // Move
    n.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > canvas.width)  p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
    });

    // Connections
    for (let i = 0; i < n.length; i++) {
      for (let j = i + 1; j < n.length; j++) {
        const dx = n[i].x - n[j].x, dy = n[i].y - n[j].y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < 130) {
          const a = (1 - d / 130) * 0.12;
          ctx.beginPath();
          ctx.strokeStyle = n[i].col + a + ')';
          ctx.lineWidth   = 0.6;
          ctx.moveTo(n[i].x, n[i].y);
          ctx.lineTo(n[j].x, n[j].y);
          ctx.stroke();
        }
      }
    }

    // Nodes
    n.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.col + '0.45)';
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  resize();
  draw();
})();
