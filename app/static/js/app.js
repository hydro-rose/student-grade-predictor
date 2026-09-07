/**
 * SynapseGrade AI - Frontend Application Logic (Obsidian Glassmorphism Suite)
 * Interactive HUD tabs, bidirectional inputs, multi-model switching, XAI waterfall,
 * counterfactual goal optimization, benchmark leaderboard, and persona clusters.
 */

document.addEventListener('DOMContentLoaded', () => {
  initHudTabs();
  initDualInputs();
  initPresetButtons();
  initForm();
  initModelSelector();
  initGoalPlanner();
  loadAnalytics();
  loadCohort();
  loadLeaderboard();
  loadClusters();

  // Run initial prediction so the results card is immediately alive
  triggerPrediction();
});

/* --------------------------------------------------------------------------
   HUD Tab Navigation
   -------------------------------------------------------------------------- */
function initHudTabs() {
  const tabBtns = document.querySelectorAll('.hud-tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      document.querySelectorAll('.tab-content').forEach(tab => {
        if (tab.id === targetId) {
          tab.classList.add('active');
          tab.style.display = 'block';
        } else {
          tab.classList.remove('active');
          tab.style.display = 'none';
        }
      });
    });
  });
}

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

function initModelSelector() {
  const selectEl = document.getElementById('select-ai-model');
  if (selectEl) {
    selectEl.addEventListener('change', () => {
      const tag = document.getElementById('active-model-tag');
      if (tag) tag.textContent = selectEl.value;
      triggerPrediction();
    });
  }
}

/* --------------------------------------------------------------------------
   Quick Load Example Profiles
   -------------------------------------------------------------------------- */
function initPresetButtons() {
  const btns = document.querySelectorAll('.example-btn[data-preset]');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      const presetKey = btn.getAttribute('data-preset');
      const data = PRESETS[presetKey];
      if (!data) return;

      const studyEl = document.getElementById('input-studytime');
      if (studyEl) studyEl.value = data.studytime;

      setFieldPair('input-g1', 'num-g1', 'badge-g1', data.g1, ' / 20');
      setFieldPair('input-g2', 'num-g2', 'badge-g2', data.g2, ' / 20');
      setFieldPair('input-absences', 'num-absences', 'badge-absences', data.absences, ' Days');
      setFieldPair('input-failures', 'num-failures', 'badge-failures', data.failures, ' Classes');

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
  debounceTimer = setTimeout(triggerPrediction, 100);
}

function getFormValues() {
  const studytime = parseFloat(document.getElementById('input-studytime')?.value || 2);
  const failures = parseInt(document.getElementById('num-failures')?.value || document.getElementById('input-failures')?.value || 0, 10);
  const absences = parseInt(document.getElementById('num-absences')?.value || document.getElementById('input-absences')?.value || 4, 10);
  const g1 = parseFloat(document.getElementById('num-g1')?.value || document.getElementById('input-g1')?.value || 12);
  const g2 = parseFloat(document.getElementById('num-g2')?.value || document.getElementById('input-g2')?.value || 13);
  const model_name = document.getElementById('select-ai-model')?.value || 'Linear Regression';

  // Sync hidden spans
  const vG1 = document.getElementById('val-g1'); if (vG1) vG1.textContent = g1;
  const vG2 = document.getElementById('val-g2'); if (vG2) vG2.textContent = g2;
  const vAbs = document.getElementById('val-absences'); if (vAbs) vAbs.textContent = absences;
  const vFail = document.getElementById('val-failures'); if (vFail) vFail.textContent = failures;

  return { studytime, failures, absences, g1, g2, model_name };
}

async function triggerPrediction() {
  const payload = getFormValues();

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
  const archetypeEl = document.getElementById('res-archetype');
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

  if (archetypeEl && data.archetype) {
    archetypeEl.textContent = `Cluster: ${data.archetype.persona_name}`;
  }

  // Render XAI Waterfall Breakdown
  renderXAIWaterfall(data.xai_breakdown);

  // Render Recommendations
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
   Explainable AI (XAI) Feature Attribution Waterfall (Obsidian Theme)
   -------------------------------------------------------------------------- */
function renderXAIWaterfall(xai) {
  const container = document.getElementById('xai-waterfall-container');
  if (!container || !xai || !xai.contributions) return;

  container.innerHTML = '';
  xai.contributions.forEach(item => {
    const isPos = item.impact_points >= 0;
    const sign = isPos ? '+' : '';
    const barWidth = Math.min(100, Math.max(12, Math.abs(item.impact_points) * 16));
    const barColor = isPos ? 'linear-gradient(90deg, #10b981, #06b6d4)' : 'linear-gradient(90deg, #f43f5e, #e11d48)';
    const textColor = isPos ? '#34d399' : '#fb7185';
    const tagBg = isPos ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)';

    const row = document.createElement('div');
    row.style.cssText = 'display: flex; align-items: center; justify-content: space-between; font-size: 0.82rem; margin-bottom: 8px;';
    row.innerHTML = `
      <div style="width: 85px; font-weight: 600; color: #cbd5e1;">${item.feature}</div>
      <div style="flex: 1; margin: 0 12px; background: rgba(255,255,255,0.06); height: 16px; border-radius: 6px; overflow: hidden; display: flex; align-items: center; border: 1px solid rgba(255,255,255,0.05);">
        <div style="width: ${barWidth}%; background: ${barColor}; height: 100%; border-radius: 5px; box-shadow: 0 0 10px ${isPos ? 'rgba(6,182,212,0.4)' : 'rgba(244,63,94,0.4)'};"></div>
      </div>
      <div style="width: 78px; text-align: right; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: ${textColor}; background: ${tagBg}; padding: 2px 8px; border-radius: 6px; border: 1px solid ${isPos ? 'rgba(16,185,129,0.3)' : 'rgba(244,63,94,0.3)'};">
        ${sign}${item.impact_points.toFixed(2)} pts
      </div>
    `;
    container.appendChild(row);
  });
}

/* --------------------------------------------------------------------------
   Counterfactual AI Goal Planner
   -------------------------------------------------------------------------- */
function initGoalPlanner() {
  const btn = document.getElementById('btn-run-goal');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    const target_g3 = parseFloat(document.getElementById('input-goal-target')?.value || 15.0);
    const formVals = getFormValues();

    const payload = {
      target_g3,
      studytime: formVals.studytime,
      failures: formVals.failures,
      absences: formVals.absences,
      g1: formVals.g1,
      g2: formVals.g2,
    };

    try {
      const res = await fetch('/api/optimize-goal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) return;
      const result = await res.json();
      
      const box = document.getElementById('goal-results-box');
      if (box) {
        box.style.display = 'block';
        box.innerHTML = `
          <div style="font-weight: 700; margin-bottom: 10px; color: #f8fafc; font-size: 0.95rem;">
            🎯 Target: <span style="color:#06b6d4;">${result.target_g3.toFixed(1)} / 20</span> &nbsp;|&nbsp; Current: <span style="color:#94a3b8;">${result.current_predicted_g3.toFixed(1)}</span> &nbsp;|&nbsp; Required Leap: <span style="color:${result.points_gap > 0 ? '#34d399' : '#f43f5e'}; font-weight:800;">${result.points_gap > 0 ? '+' + result.points_gap.toFixed(1) : result.points_gap.toFixed(1)} pts</span>
          </div>
          <div style="display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap;">
            <span style="background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.4); color: #c084fc; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;">
              📚 Recommended Study: ${result.recommended_studytime}
            </span>
            <span style="background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.4); color: #22d3ee; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;">
              📅 Absence Threshold: ≤ ${result.recommended_absences} Days
            </span>
          </div>
          <ul style="margin-left: 18px; line-height: 1.6; color: #e2e8f0; font-size: 0.84rem;">
            ${result.action_steps.map(step => `<li style="margin-bottom: 4px;">${step}</li>`).join('')}
          </ul>
        `;
      }
    } catch (err) {
      console.error('Goal planning failed:', err);
    }
  });
}

/* --------------------------------------------------------------------------
   Model Benchmark Leaderboard Loader
   -------------------------------------------------------------------------- */
async function loadLeaderboard() {
  const tbody = document.getElementById('leaderboard-tbody');
  if (!tbody) return;

  try {
    const res = await fetch('/api/models');
    if (!res.ok) return;
    const data = await res.json();
    const rows = data.regression_leaderboard || [];

    tbody.innerHTML = '';
    rows.forEach((row, idx) => {
      const isTop = idx === 0;
      const isPDF = row.model_name === 'Linear Regression';
      let tag = '';
      if (isTop) tag = ' 🌟 <span style="color:#10b981; font-weight:700; font-size:0.72rem; text-shadow:0 0 8px rgba(16,185,129,0.5);">Top Regressor</span>';
      else if (isPDF) tag = ' 📄 <span style="color:#06b6d4; font-weight:700; font-size:0.72rem;">PDF Benchmark</span>';

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="color:#64748b; font-family:'JetBrains Mono',monospace;">#${idx + 1}</td>
        <td><strong style="color:#f8fafc;">${row.model_name}</strong>${tag}</td>
        <td><span style="color:#34d399; font-weight:700; font-family:'JetBrains Mono',monospace;">${row.r2_percentage}%</span></td>
        <td style="font-family:'JetBrains Mono',monospace; color:#cbd5e1;">${row.mse.toFixed(4)}</td>
        <td style="font-family:'JetBrains Mono',monospace; color:#cbd5e1;">${row.mae.toFixed(4)}</td>
        <td style="font-family:'JetBrains Mono',monospace; color:#94a3b8;">${row.cv_r2_mean ? (row.cv_r2_mean * 100).toFixed(1) + '%' : 'N/A'}</td>
        <td style="font-family:'JetBrains Mono',monospace; color:#64748b;">${row.mse_95_ci ? `[${row.mse_95_ci[0]} – ${row.mse_95_ci[1]}]` : 'N/A'}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Failed to load leaderboard:', err);
  }
}

/* --------------------------------------------------------------------------
   Unsupervised Persona Clusters Loader (Obsidian Theme)
   -------------------------------------------------------------------------- */
async function loadClusters() {
  const container = document.getElementById('clusters-summary-container');
  if (!container) return;

  try {
    const res = await fetch('/api/clusters');
    if (!res.ok) return;
    const data = await res.json();
    const profiles = data.profiles || [];

    container.innerHTML = '';
    profiles.forEach(p => {
      const card = document.createElement('div');
      card.style.cssText = 'background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px; font-size: 0.82rem; backdrop-filter: blur(8px);';
      card.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
          <strong style="color: #f8fafc; font-size: 0.9rem;">${p.persona_name}</strong>
          <span style="background: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.3); padding: 2px 8px; border-radius: 12px; font-weight: 600; font-size: 0.74rem;">
            ${p.student_count} students (${p.percentage}%)
          </span>
        </div>
        <div style="color: #94a3b8; font-size: 0.78rem; line-height: 1.5;">
          Mean Final Grade: <strong style="color: #34d399;">${p.mean_G3} / 20</strong> &bull; Pass Rate: <strong style="color: #38bdf8;">${p.pass_rate}%</strong> &bull; Mean Absences: <span style="color:#cbd5e1;">${p.mean_absences} days</span>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load clusters:', err);
  }
}

/* --------------------------------------------------------------------------
   Research Background: Correlation Matrix & Cohort (Obsidian Theme)
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
  let html = '<table class="heatmap-table" style="width:100%; border-collapse: collapse; font-size:0.8rem; font-family:\'JetBrains Mono\', monospace;"><thead><tr><th style="padding:8px; border:1px solid rgba(255,255,255,0.06); color:#64748b;">Factor</th>';
  features.forEach(f => {
    html += `<th style="padding:8px; border:1px solid rgba(255,255,255,0.06); color:#cbd5e1;">${f}</th>`;
  });
  html += '</tr></thead><tbody>';

  features.forEach(row => {
    html += `<tr><th style="padding:8px; border:1px solid rgba(255,255,255,0.06); color:#94a3b8; text-align:left;">${row}</th>`;
    features.forEach(col => {
      const val = corr[row] ? corr[row][col] : (row === col ? 1.0 : 0.0);
      let bg = 'rgba(255, 255, 255, 0.02)';
      let color = '#94a3b8';
      if (val >= 0.7) { 
        bg = 'rgba(6, 182, 212, 0.2)'; 
        color = '#22d3ee'; 
      } else if (val <= -0.3) { 
        bg = 'rgba(244, 63, 94, 0.2)'; 
        color = '#fb7185'; 
      } else if (val > 0.1) {
        bg = 'rgba(16, 185, 129, 0.1)';
        color = '#34d399';
      }
      html += `<td style="background-color: ${bg}; color: ${color}; font-weight: 600; text-align:center; padding:8px; border:1px solid rgba(255,255,255,0.06);">${Number(val).toFixed(2)}</td>`;
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
        ? `<span style="color:#34d399; background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3); padding:2px 8px; border-radius:12px; font-weight:700; font-size:0.75rem;">PASS</span>` 
        : `<span style="color:#fb7185; background:rgba(244,63,94,0.15); border:1px solid rgba(244,63,94,0.3); padding:2px 8px; border-radius:12px; font-weight:700; font-size:0.75rem;">AT RISK</span>`;

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="color:#64748b; font-family:'JetBrains Mono',monospace;">${s.id}</td>
        <td style="color:#f8fafc; font-weight:600;">${s.name}</td>
        <td><span style="background:rgba(255,255,255,0.05); color:#cbd5e1; padding:2px 8px; border-radius:4px; font-size:0.76rem;">${s.profile}</span></td>
        <td style="color:#94a3b8;">${s.studytime === 1 ? '<2h' : s.studytime === 2 ? '2-5h' : '5+h'}</td>
        <td style="color:#cbd5e1;">${s.failures}</td>
        <td style="color:#cbd5e1;">${s.absences} days</td>
        <td style="color:#cbd5e1; font-family:'JetBrains Mono',monospace;">${s.G1}</td>
        <td style="color:#cbd5e1; font-family:'JetBrains Mono',monospace;">${s.G2}</td>
        <td><strong style="color:#34d399; font-family:'JetBrains Mono',monospace;">${g3Pred} / 20</strong></td>
        <td>${badge}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Failed to load sample cohort:', err);
  }
}
