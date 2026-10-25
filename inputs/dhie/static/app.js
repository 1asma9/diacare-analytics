// ── example preset buttons ────────────────────────────────────────────────
document.querySelectorAll('[data-example]').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var d = JSON.parse(btn.getAttribute('data-example'));
    Object.keys(d).forEach(function (k) {
      var el = document.getElementById(k);
      if (el) {
        el.value = d[k];
        el.dispatchEvent(new Event('input'));
        el.dispatchEvent(new Event('change'));
      }
    });
  });
});

// ── submit feedback ───────────────────────────────────────────────────────
var form = document.getElementById('form');
if (form) {
  form.addEventListener('submit', function () {
    var btn = document.getElementById('go');
    if (btn) {
      btn.disabled = true;
      btn.textContent = 'Calculating…';
    }
  });
}
