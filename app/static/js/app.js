/**
 * EduPredict AI - Frontend Application Logic
 * Bidirectional inputs, live predictions, instant feedback & pedagogical tips
 */

document.addEventListener('DOMContentLoaded', () => {
  initDualInputs();
  initPresetButtons();
  initForm();
  loadAnalytics();
  loadCohort();

  // Run initial prediction so the results card is immediately alive
  triggerPrediction();
});

const PRESETS = {
  honor: { studytime: 3, failures: 0, absences: 2, g1: 16, g2: 17 },
  average: { studytime: 2, failures: 0, absences: 6, g1: 11, g2: 12 },
  at_risk: { studytime: 1, failures: 2, absences: 24, g1: 6, g2: 7 }
};

/* --------------------------------------------------------------------------
   Bidirectional Input Synchronization (Slider <-> Number Box <-> Badge)
   -------------------------------------------------------------------------- */
function initDualInputs() {
  bindSliderAndNumber('input-g1', 'num-g1', 'badge-g1', ' / 20');
  bindSliderAndNumber('input-g2', 'num-g2', 'badge-g2', ' / 20');
  bindSliderAndNumber('input-absences', 'num-absences', 'badge-absences', ' Days');
  bindSliderAndNumber('input-failures', 'num-failures', 'badge-failures', ' Classes');

  const studySelect = document.getElementById('input-studytime');
  if (studySelect) {
    studySelect.addEventListener('change', () => triggerPrediction());
  }
}

function bindSliderAndNumber(sliderId, numId, badgeId, suffix = '') {
  const slider = document.getElementById(sliderId);
  const numBox = document.getElementById(numId);
  const badge = document.getElementById(badgeId);

  if (!slider || !numBox) return;

  slider.addEventListener('input', (e) => {
    numBox.value = e.target.value;
    if (badge) badge.textContent = `${e.target.value}${suffix}`;
    triggerPredictionDebounced();
  });

  numBox.addEventListener('input', (e) => {
    slider.value = e.target.value;
    if (badge) badge.textContent = `${e.target.value}${suffix}`;
    triggerPredictionDebounced();
  });
}

/* --------------------------------------------------------------------------
   Quick Load Example Profiles
   -------------------------------------------------------------------------- */
function initPresetButtons() {
  const btns = document.querySelectorAll('.example-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      const presetKey = btn.getAttribute('data-preset');
      const data = PRESETS[presetKey];
      if (!data) return;

      // Update study time
      const studyEl = document.getElementById('input-studytime');
      if (studyEl) studyEl.value = data.studytime;

      // Update G1
      setFieldPair('input-g1', 'num-g1', 'badge-g1', data.g1, ' / 20');
      // Update G2
      setFieldPair('input-g2', 'num-g2', 'badge-g2', data.g2, ' / 20');
      // Update Absences
      setFieldPair('input-absences', 'num-absences', 'badge-absences', data.absences, ' Days');
      // Update Failures
      setFieldPair('input-failures', 'num-failures', 'badge-failures', data.failures, ' Classes');

      // Trigger instant calculation
      triggerPrediction();
    });
  });
}

function setFieldPair(sliderId, numId, badgeId, value, suffix) {
  const slider = document.getElementById(sliderId);
  const numBox = document.getElementById(numId);
  const badge = document.getElementById(badgeId);

  if (slider) slider.value = value;
  if (numBox) numBox.value = value;
  if (badge) badge.textContent = `${value}${suffix}`;
}

/* --------------------------------------------------------------------------
   Form Submission & API Prediction
   -------------------------------------------------------------------------- */
function initForm() {
  const form = document.getElementById('grade-predict-form');
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      triggerPrediction();
    });
  }
}

let debounceTimer = null;
function triggerPredictionDebounced() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(triggerPrediction, 120);
}

async function triggerPrediction() {
  const studytime = parseFloat(document.getElementById('input-studytime')?.value || 2);
  const failures = parseInt(document.getElementById('num-failures')?.value || document.getElementById('input-failures')?.value || 0, 10);
  const absences = parseInt(document.getElementById('num-absences')?.value || document.getElementById('input-absences')?.value || 4, 10);
  const g1 = parseFloat(document.getElementById('num-g1')?.value || document.getElementById('input-g1')?.value || 12);
  const g2 = parseFloat(document.getElementById('num-g2')?.value || document.getElementById('input-g2')?.value || 13);

  // Sync hidden spans for backward compatibility
  const vG1 = document.getElementById('val-g1'); if (vG1) vG1.textContent = g1;
  const vG2 = document.getElementById('val-g2'); if (vG2) vG2.textContent = g2;
  const vAbs = document.getElementById('val-absences'); if (vAbs) vAbs.textContent = absences;
  const vFail = document.getElementById('val-failures'); if (vFail) vFail.textContent = failures;

  const payload = { studytime, failures, absences, g1, g2 };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) return;
    const data = await res.json();
    renderPredictionOutput(data);
  } catch (err) {
    console.error('Prediction API call failed:', err);
  }
}

function renderPredictionOutput(data) {
  const scoreEl = document.getElementById('res-score');
  const letterEl = document.getElementById('res-letter');
  const statusEl = document.getElementById('res-status');
  const probEl = document.getElementById('res-probability');
  const ciEl = document.getElementById('res-ci');
  const recsContainer = document.getElementById('res-recs');

  if (scoreEl) scoreEl.textContent = data.predicted_g3.toFixed(1);
  if (letterEl) letterEl.textContent = data.letter_grade;

  if (statusEl) {
    const isPass = data.status === 'PASS';
    statusEl.textContent = isPass ? 'PASS' : 'AT RISK';
    statusEl.className = 'status-badge-lg ' + (isPass ? 'pass' : 'fail');
  }

  if (probEl) probEl.textContent = `${data.pass_probability}%`;
  if (ciEl) ciEl.textContent = `[${data.confidence_interval[0]} – ${data.confidence_interval[1]}]`;

  if (recsContainer) {
    recsContainer.innerHTML = '';
    if (data.recommendations && data.recommendations.length > 0) {
      data.recommendations.forEach(rec => {
        const item = document.createElement('div');
        let typeClass = 'feedback-item';
        if (rec.includes('CRITICAL') || rec.includes('Caution') || rec.includes('risk') || rec.includes('Multiple past')) {
          typeClass += ' alert';
        } else if (rec.includes('Exemplary') || rec.includes('Outstanding') || rec.includes('Strong')) {
          typeClass += ' success';
        }
        item.className = typeClass;
        item.textContent = rec;
        recsContainer.appendChild(item);
      });
    } else {
      recsContainer.innerHTML = '<div class="feedback-item success">Student performance is consistent. Maintain regular study routines.</div>';
    }
  }
}

/* --------------------------------------------------------------------------
   Collapsible Research Background: Correlation Matrix & Cohort
   -------------------------------------------------------------------------- */
async function loadAnalytics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) return;
    const meta = await res.json();
    renderCorrelationTable(meta.correlation_matrix);
  } catch (err) {
    console.error('Failed to load metrics:', err);
  }
}

function renderCorrelationTable(corr) {
  const container = document.getElementById('correlation-heatmap-container');
  if (!container || !corr) return;

  const features = ['studytime', 'failures', 'absences', 'G1', 'G2'];
  let html = '<table class="heatmap-table"><thead><tr><th>Factor</th>';
  features.forEach(f => {
    html += `<th>${f}</th>`;
  });
  html += '</tr></thead><tbody>';

  features.forEach(row => {
    html += `<tr><th style="background:#f8fafc; font-weight:600;">${row}</th>`;
    features.forEach(col => {
      const val = corr[row] ? corr[row][col] : (row === col ? 1.0 : 0.0);
      let bg = '#ffffff';
      let color = '#334155';
      if (val >= 0.7) { bg = '#dbeafe'; color = '#1e40af'; }
      else if (val <= -0.3) { bg = '#fee2e2'; color = '#991b1b'; }
      html += `<td style="background-color: ${bg}; color: ${color}; font-weight: 600;">${Number(val).toFixed(2)}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

async function loadCohort() {
  const tbody = document.getElementById('cohort-tbody');
  if (!tbody) return;

  try {
    const res = await fetch('/api/sample-students');
    if (!res.ok) return;
    const students = await res.json();

    tbody.innerHTML = '';
    students.forEach(s => {
      const g3Pred = (-1.71 + 0.99 * s.G2 + 0.15 * s.G1 - 0.32 * s.failures - 0.16 * s.studytime + 0.04 * s.absences).toFixed(1);
      const isPass = parseFloat(g3Pred) >= 10.0;
      const badge = isPass 
        ? `<span style="color:#065f46; background:#ecfdf5; border:1px solid #a7f3d0; padding:2px 8px; border-radius:12px; font-weight:700; font-size:0.75rem;">PASS</span>` 
        : `<span style="color:#991b1b; background:#fef2f2; border:1px solid #fecaca; padding:2px 8px; border-radius:12px; font-weight:700; font-size:0.75rem;">AT RISK</span>`;

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${s.id}</strong></td>
        <td>${s.name}</td>
        <td><span style="background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:0.78rem;">${s.profile}</span></td>
        <td>${s.studytime === 1 ? '<2h' : s.studytime === 2 ? '2-5h' : '5+h'}</td>
        <td>${s.failures}</td>
        <td>${s.absences} days</td>
        <td>${s.G1}</td>
        <td>${s.G2}</td>
        <td><strong>${g3Pred} / 20</strong></td>
        <td>${badge}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Failed to load sample cohort:', err);
  }
}
