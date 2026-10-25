/**
 * DiaCare Analytics - Shared Interactive JavaScript
 */

document.addEventListener('DOMContentLoaded', function () {
  'use strict';

  // 1. Sync range sliders with value labels
  document.querySelectorAll('input[type="range"]').forEach(function (slider) {
    var output = document.getElementById(slider.id + '_v');
    function updateValue() {
      if (output) {
        output.textContent = slider.value;
      }
    }
    slider.addEventListener('input', updateValue);
    updateValue();
  });

  // 2. Preset example chips auto-fill
  document.querySelectorAll('[data-example]').forEach(function (chip) {
    chip.addEventListener('click', function () {
      var rawData = chip.getAttribute('data-example');
      if (!rawData) return;
      try {
        var data = JSON.parse(rawData);
        Object.keys(data).forEach(function (key) {
          var input = document.getElementById(key);
          if (input) {
            input.value = data[key];
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
          }
        });
      } catch (err) {
        console.error('Error parsing preset example data:', err);
      }
    });
  });

  // 3. Form submit state
  var calcForm = document.getElementById('calc-form') || document.getElementById('form');
  if (calcForm) {
    calcForm.addEventListener('submit', function () {
      var submitBtn = document.getElementById('btn-submit') || document.getElementById('go');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Calculating…';
      }
    });
  }

  // 4. File input filename feedback for CSV upload
  var fileInput = document.getElementById('csv-file') || document.getElementById('file');
  var fileNameDisplay = document.getElementById('file-name-display');
  if (fileInput && fileNameDisplay) {
    fileInput.addEventListener('change', function () {
      if (fileInput.files && fileInput.files.length > 0) {
        fileNameDisplay.textContent = fileInput.files[0].name + ' (' + (fileInput.files[0].size / 1024).toFixed(1) + ' KB)';
      } else {
        fileNameDisplay.textContent = 'No file chosen';
      }
    });
  }
});
