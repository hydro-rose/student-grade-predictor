/**
 * EduPredict AI - Frontend Application Logic
 * Interactivity, Live Model Predictions, Sensitivity Simulator, and Dynamic Visualizations
 */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initPresetButtons();
  initPredictionForm();
  initSensitivitySimulator();
  loadAnalyticsAndMetrics();
  loadSampleCohort();
});

/* --------------------------------------------------------------------------
   Tab Navigation
   -------------------------------------------------------------------------- */
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');

      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }
    });
  });
}

/* --------------------------------------------------------------------------
   Preset Profiles
   -------------------------------------------------------------------------- */
const PRESETS = {
  honor: { studytime: 3, failures: 0, absences: 2, g1: 16, g2: 17 },
  average: { studytime: 2, failures: 0, absences: 6, g1: 11, g2: 12 },
  at_risk: { studytime: 1, failures: 2, absences: 24, g1: 6, g2: 7 },
  improver: { studytime: 3, failures: 0, absences: 4, g1: 9, g2: 14 }
};

function initPresetButtons() {
  const chips = document.querySelectorAll('.preset-chip');
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      const presetKey = chip.getAttribute('data-preset');
      const data = PRESETS[presetKey];
      if (!data) return;

      document.getElementById('input-studytime').value = data.studytime;
      document.getElementById('input-failures').value = data.failures;
      document.getElementById('val-failures').textContent = data.failures;

      document.getElementById('input-absences').value = data.absences;
      document.getElementById('val-absences').textContent = data.absences;

      document.getElementById('input-g1').value = data.g1;
      document.getElementById('val-g1').textContent = data.g1;

      document.getElementById('input-g2').value = data.g2;
      document.getElementById('val-g2').textContent = data.g2;

      // Trigger instant prediction
      triggerPrediction();
    });
  });

  // Attach dynamic range slider number updates
  setupSliderListener('input-failures', 'val-failures');
  setupSliderListener('input-absences', 'val-absences');
  setupSliderListener('input-g1', 'val-g1');
  setupSliderListener('input-g2', 'val-g2');
}

function setupSliderListener(inputId, labelId) {
  const slider = document.getElementById(inputId);
  const label = document.getElementById(labelId);
  if (slider && label) {
    slider.addEventListener('input', (e) => {
      label.textContent = e.target.value;
    });
  }
}

/* --------------------------------------------------------------------------
   Single Prediction Form
   -------------------------------------------------------------------------- */
function initPredictionForm() {
  const form = document.getElementById('grade-predict-form');
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      triggerPrediction();
    });
  }
}

async function triggerPrediction() {
  const payload = {
    studytime: parseFloat(document.getElementById('input-studytime').value),
    failures: parseInt(document.getElementById('input-failures').value, 10),
    absences: parseInt(document.getElementById('input-absences').value, 10),
    g1: parseFloat(document.getElementById('input-g1').value),
    g2: parseFloat(document.getElementById('input-g2').value),
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Prediction API error');
    const data = await res.json();
    renderPredictionResult(data);
  } catch (err) {
    console.error('Error fetching prediction:', err);
  }
}

function renderPredictionResult(data) {
  const scoreEl = document.getElementById('res-score');
  const letterEl = document.getElementById('res-letter');
  const statusEl = document.getElementById('res-status');
  const probEl = document.getElementById('res-probability');
  const ciEl = document.getElementById('res-ci');
  const recsContainer = document.getElementById('res-recs');

  if (scoreEl) scoreEl.textContent = data.predicted_g3.toFixed(1);
  if (letterEl) letterEl.textContent = data.letter_grade;

  if (statusEl) {
    statusEl.textContent = data.status;
    statusEl.className = 'status-badge ' + (data.status === 'PASS' ? 'status-pass' : 'status-fail');
  }

  if (probEl) probEl.textContent = `${data.pass_probability}%`;
  if (ciEl) ciEl.textContent = `[${data.confidence_interval[0]} – ${data.confidence_interval[1]}]`;

  if (recsContainer) {
    recsContainer.innerHTML = '';
    if (data.recommendations && data.recommendations.length > 0) {
      data.recommendations.forEach(rec => {
        const item = document.createElement('div');
        let typeClass = 'rec-item';
        if (rec.includes('CRITICAL') || rec.includes('Caution')) {
          typeClass += ' alert';
        } else if (rec.includes('Exemplary') || rec.includes('Outstanding') || rec.includes('Positive')) {
          typeClass += ' success';
        }
        item.className = typeClass;
        item.textContent = rec;
        recsContainer.appendChild(item);
      });
    } else {
      recsContainer.innerHTML = '<div class="rec-item">Student is on track. Maintain current study patterns.</div>';
    }
  }
}

/* --------------------------------------------------------------------------
   What-If Sensitivity Simulator
   -------------------------------------------------------------------------- */
let coefficients = {
  studytime: -0.159,
  failures: -0.318,
  absences: 0.043,
  G1: 0.147,
  G2: 0.991,
  intercept: -1.713,
};

function initSensitivitySimulator() {
  const simStudy = document.getElementById('sim-studytime');
  const simAbs = document.getElementById('sim-absences');
  const simG1 = document.getElementById('sim-g1');
  const simG2 = document.getElementById('sim-g2');

  const listeners = [simStudy, simAbs, simG1, simG2];
  listeners.forEach(el => {
    if (el) {
      el.addEventListener('input', runSensitivitySimulation);
    }
  });

  setupSliderListener('sim-studytime', 'val-sim-studytime');
  setupSliderListener('sim-absences', 'val-sim-absences');
  setupSliderListener('sim-g1', 'val-sim-g1');
  setupSliderListener('sim-g2', 'val-sim-g2');

  runSensitivitySimulation();
}

function runSensitivitySimulation() {
  const study = parseFloat(document.getElementById('sim-studytime')?.value || 2);
  const abs = parseInt(document.getElementById('sim-absences')?.value || 6, 10);
  const g1 = parseFloat(document.getElementById('sim-g1')?.value || 11);
  const g2 = parseFloat(document.getElementById('sim-g2')?.value || 12);

  // Linear regression estimation
  const simulatedScore = Math.max(0, Math.min(20, 
    coefficients.intercept +
    coefficients.studytime * study +
    coefficients.failures * 0 +
    coefficients.absences * abs +
    coefficients.G1 * g1 +
    coefficients.G2 * g2
  ));

  const baseScore = Math.max(0, Math.min(20, 
    coefficients.intercept +
    coefficients.studytime * 2 +
    coefficients.failures * 0 +
    coefficients.absences * 6 +
    coefficients.G1 * 11 +
    coefficients.G2 * 11
  ));

  const delta = (simulatedScore - baseScore).toFixed(1);
  const simScoreEl = document.getElementById('sim-result-score');
  const simDeltaEl = document.getElementById('sim-delta-badge');
  const simAttPctEl = document.getElementById('sim-att-pct');

  if (simScoreEl) simScoreEl.textContent = simulatedScore.toFixed(1);
  if (simDeltaEl) {
    simDeltaEl.textContent = delta >= 0 ? `+${delta} pts` : `${delta} pts`;
    simDeltaEl.style.background = delta >= 0 ? 'rgba(16, 185, 129, 0.2)' : 'rgba(244, 63, 94, 0.2)';
    simDeltaEl.style.color = delta >= 0 ? '#10b981' : '#f43f5e';
  }

  if (simAttPctEl) {
    const attPct = Math.max(0, Math.min(100, (100 - (abs / 90) * 100))).toFixed(0);
    simAttPctEl.textContent = `${attPct}%`;
    simAttPctEl.style.color = attPct >= 60 ? '#10b981' : '#f43f5e';
  }
}

/* --------------------------------------------------------------------------
   Analytics & Metrics
   -------------------------------------------------------------------------- */
async function loadAnalyticsAndMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) return;
    const meta = await res.json();

    if (meta.coefficients) {
      coefficients = meta.coefficients;
    }

    renderCorrelationHeatmap(meta.correlation_matrix);
    renderMetricsTable(meta);
    renderAttendanceStats(meta.attendance_insights);
  } catch (err) {
    console.error('Failed to load metrics:', err);
  }
}

function renderCorrelationHeatmap(corr) {
  const container = document.getElementById('correlation-heatmap-container');
  if (!container || !corr) return;

  const features = ['studytime', 'failures', 'absences', 'G1', 'G2'];
  let html = '<table class="heatmap-table"><thead><tr><th>Feature</th>';
  features.forEach(f => {
    html += `<th>${f}</th>`;
  });
  html += '</tr></thead><tbody>';

  features.forEach(row => {
    html += `<tr><th>${row}</th>`;
    features.forEach(col => {
      const val = corr[row] ? corr[row][col] : (row === col ? 1.0 : 0.0);
      const color = getHeatmapColor(val);
      html += `<td class="heatmap-cell" style="background-color: ${color}; color: #fff;">${Number(val).toFixed(2)}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

function getHeatmapColor(val) {
  // Magma-style gradient interpolation
  if (val >= 0.7) return 'rgba(245, 158, 11, 0.85)';
  if (val >= 0.4) return 'rgba(217, 70, 239, 0.75)';
  if (val >= 0.1) return 'rgba(99, 102, 241, 0.65)';
  if (val >= -0.1) return 'rgba(31, 41, 55, 0.85)';
  if (val >= -0.25) return 'rgba(76, 29, 149, 0.7)';
  return 'rgba(15, 23, 42, 0.9)';
}

function renderMetricsTable(meta) {
  const lr = meta.linear_regression || {};
  const ridge = meta.ridge_regression || {};
  const rf = meta.rf_regressor || {};
  const clf = meta.classifier || {};

  const lrMse = document.getElementById('metric-lr-mse');
  const lrR2 = document.getElementById('metric-lr-r2');
  const lrCv = document.getElementById('metric-lr-cv');

  if (lrMse) lrMse.textContent = lr.mse ?? '2.62';
  if (lrR2) lrR2.textContent = `${lr.r2_percentage ?? '86.14'}%`;
  if (lrCv) lrCv.textContent = `${lr.cv_r2_mean ?? '0.84'} ± ${lr.cv_r2_std ?? '0.03'}`;

  const ridgeMse = document.getElementById('metric-ridge-mse');
  const ridgeR2 = document.getElementById('metric-ridge-r2');
  if (ridgeMse) ridgeMse.textContent = ridge.mse ?? '2.62';
  if (ridgeR2) ridgeR2.textContent = `${ridge.r2_percentage ?? '86.14'}%`;

  const rfMse = document.getElementById('metric-rf-mse');
  const rfR2 = document.getElementById('metric-rf-r2');
  if (rfMse) rfMse.textContent = rf.mse ?? '3.01';
  if (rfR2) rfR2.textContent = `${rf.r2_percentage ?? '84.07'}%`;

  const clfAcc = document.getElementById('metric-clf-acc');
  const clfF1 = document.getElementById('metric-clf-f1');
  if (clfAcc) clfAcc.textContent = `${clf.accuracy_pct ?? '91.14'}%`;
  if (clfF1) clfF1.textContent = clf.f1_score ?? '0.93';
}

function renderAttendanceStats(att) {
  if (!att) return;
  const highAvg = document.getElementById('stat-att-high-grade');
  const lowAvg = document.getElementById('stat-att-low-grade');
  const highPass = document.getElementById('stat-att-high-pass');
  const lowPass = document.getElementById('stat-att-low-pass');

  if (highAvg && att.above_60_attendance) highAvg.textContent = `${att.above_60_attendance.avg_grade} / 20`;
  if (lowAvg && att.below_60_attendance) lowAvg.textContent = `${att.below_60_attendance.avg_grade} / 20`;
  if (highPass && att.above_60_attendance) highPass.textContent = `${att.above_60_attendance.pass_rate}% Pass`;
  if (lowPass && att.below_60_attendance) lowPass.textContent = `${att.below_60_attendance.pass_rate}% Pass`;
}

/* --------------------------------------------------------------------------
   Sample Cohort Loader
   -------------------------------------------------------------------------- */
async function loadSampleCohort() {
  const tbody = document.getElementById('cohort-tbody');
  if (!tbody) return;

  try {
    const res = await fetch('/api/sample-students');
    if (!res.ok) return;
    const students = await res.json();

    tbody.innerHTML = '';
    students.forEach(s => {
      // Calculate prediction locally or from API
      const g3Pred = Math.max(0, Math.min(20,
        coefficients.intercept +
        coefficients.studytime * s.studytime +
        coefficients.failures * s.failures +
        coefficients.absences * s.absences +
        coefficients.G1 * s.G1 +
        coefficients.G2 * s.G2
      )).toFixed(1);

      const isPass = g3Pred >= 10.0;
      const statusBadge = isPass 
        ? `<span class="status-badge status-pass">PASS</span>` 
        : `<span class="status-badge status-fail">AT RISK</span>`;

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${s.id}</strong></td>
        <td>${s.name}</td>
        <td><span class="badge-tag">${s.profile}</span></td>
        <td>${s.studytime} (${s.studytime === 1 ? '<2h' : s.studytime === 2 ? '2-5h' : '5+h'})</td>
        <td>${s.failures}</td>
        <td>${s.absences}</td>
        <td>${s.G1}</td>
        <td>${s.G2}</td>
        <td><strong>${g3Pred} / 20</strong></td>
        <td>${statusBadge}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Error loading sample cohort:', err);
  }
}
