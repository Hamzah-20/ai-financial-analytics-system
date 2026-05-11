const S = {
  step: 1,
  dataUploaded: false,
  columns: [],
  dateColumns: [],
  modelType: null,
  modelTrained: false,
  metrics: null,
  plotB64: null,
};

const METRIC_LABELS = {
  accuracy:'Accuracy', mse:'MSE', rmse:'RMSE', mae:'MAE',
  r2_score:'R²', samples:'Samples', classes:'Classes',
  clusters:'Clusters', silhouette_score:'Silhouette',
  inertia:'Inertia', aic:'AIC', adj_r2:'Adj. R²',
  f_statistic:'F-Statistic', log_loss:'Log Loss',
};

const MODEL_COLOR = {
  timeseries: 'var(--ts-color)', regression: 'var(--reg-color)',
  classification: 'var(--cls-color)', clustering: 'var(--clu-color)',
};


function toast(msg, type = 'info') {
  const icons = { success:'✓', error:'✕', warning:'⚠', info:'ℹ' };
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `<span class="toast-icon">${icons[type]}</span><span class="toast-msg">${msg}</span><button class="toast-close" onclick="this.parentElement.remove()">✕</button>`;
  document.getElementById('toastContainer').appendChild(el);
  setTimeout(() => el.remove(), 4500);
}


function canGo(n) {
  if (n <= 1) return true;
  if (n === 2) return S.dataUploaded;
  if (n === 3) return S.dataUploaded && S.modelType;
  return S.modelTrained;
}

function tryGoToStep(n) {
  if (canGo(n)) goToStep(n);
  else toast('Please complete previous steps first', 'warning');
}

function goToStep(n) {
  if (!canGo(n)) { toast('Please complete previous steps first', 'warning'); return; }
  S.step = n;
  for (let i = 1; i <= 5; i++) {
    document.getElementById(`sec${i}`).style.display = (i === n) ? 'block' : 'none';
    const node = document.getElementById(`sn${i}`);
    const circle = document.getElementById(`sc${i}`);
    const track = document.getElementById(`st${i}`);
    node.className = 'step-node' + (i === n ? ' active' : i < n ? ' done' : canGo(i) ? '' : ' locked');
    circle.textContent = i < n ? '✓' : i;
    if (track) track.className = 'step-track' + (i < n ? ' done' : '');
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}


const uploadZone = document.getElementById('uploadZone');
const fileInput  = document.getElementById('fileInput');

uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('drag'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag');
  if (e.dataTransfer.files[0]) { fileInput.files = e.dataTransfer.files; handleUpload(); }
});
fileInput.addEventListener('change', handleUpload);

async function handleUpload() {
  const file = fileInput.files[0];
  if (!file) return;
  toast('Uploading file…', 'info');
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/upload_data', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.success) {
      showUploadSuccess(d);
      S.dataUploaded = true;
      S.columns = d.columns;
      S.dateColumns = d.date_columns || [];
      populateSelects(d.columns, d.date_columns || []);
      toast('Data uploaded successfully!', 'success');
      setTimeout(() => goToStep(2), 900);
    } else toast(d.error || 'Upload failed', 'error');
  } catch(e) { toast('Upload failed: ' + e.message, 'error'); }
}

async function loadSampleData() {
  toast('Loading sample data…', 'info');
  try {
    const r = await fetch('/get_sample_data');
    const d = await r.json();
    if (d.success) {
      const cols = Object.keys(d.sample_data);
      showUploadSuccessRaw('sample_banking_data.csv', 12, cols.length, d.preview);
      S.dataUploaded = true;
      S.columns = cols;
      S.dateColumns = ['Date'];
      populateSelects(cols, ['Date']);
      toast('Sample data loaded!', 'success');
      setTimeout(() => goToStep(2), 900);
    } else toast(d.error || 'Failed', 'error');
  } catch(e) { toast('Failed: ' + e.message, 'error'); }
}

function showUploadSuccess(d) {
  showUploadSuccessRaw(d.filename, d.shape.rows, d.shape.cols, d.preview);
}

function showUploadSuccessRaw(name, rows, cols, previewHtml) {
  document.getElementById('fileName').textContent = name;
  document.getElementById('fileMeta').textContent = `${rows} rows × ${cols} columns`;
  document.getElementById('previewTable').innerHTML = previewHtml || '';
  document.getElementById('uploadPreview').style.display = 'block';
}

function populateSelects(cols, dateCols) {
  const t = document.getElementById('targetSelect');
  const f = document.getElementById('featureSelect');
  t.innerHTML = '<option value="">Select target...</option>';
  f.innerHTML = '<option value="">Select feature (optional)</option>';
  cols.forEach(c => {
    const isDate = dateCols.includes(c);
    t.innerHTML += `<option value="${c}">${c}${isDate ? ' (date)' : ''}</option>`;
    f.innerHTML += `<option value="${c}">${c}${isDate ? ' (date)' : ''}</option>`;
  });
}


const MODEL_META = {
  timeseries:     { cls: 'selected-ts',  color: 'var(--ts-color)',  label: ' Time Series Forecasting' },
  regression:     { cls: 'selected-reg', color: 'var(--reg-color)', label: ' Regression Analysis' },
  classification: { cls: 'selected-cls', color: 'var(--cls-color)', label: ' Classification Model' },
  clustering:     { cls: 'selected-clu', color: 'var(--clu-color)', label: ' Clustering (K-Means)' },
};

function selectModel(type) {
  S.modelType = type;
  ['timeseries','regression','classification','clustering'].forEach(t => {
    const card = document.getElementById(`mc-${t}`);
    card.className = 'model-card' + (t === type ? ` ${MODEL_META[t].cls}` : '');
    const badge = card.querySelector('.selected-badge');
    if (badge) badge.remove();
  });
  const card = document.getElementById(`mc-${type}`);
  const badge = document.createElement('div');
  badge.className = 'selected-badge';
  badge.style.background = MODEL_META[type].color;
  badge.textContent = 'SELECTED';
  card.appendChild(badge);
  document.getElementById('btnStep2Next').disabled = false;

  document.getElementById('tsParams').style.display      = type === 'timeseries'  ? 'block' : 'none';
  document.getElementById('clusterParams').style.display = type === 'clustering'  ? 'block' : 'none';

  const banner = document.getElementById('modelInfoBanner');
  banner.style.display = 'flex';
  banner.style.background = MODEL_META[type].color + '18';
  banner.style.border = `1px solid ${MODEL_META[type].color}30`;
  banner.innerHTML = `<span style="font-size:18px">${MODEL_META[type].label.split(' ')[0]}</span><span style="color:${MODEL_META[type].color};font-size:14px">${MODEL_META[type].label}</span>`;
}


async function setupModel() {
  const target = document.getElementById('targetSelect').value;
  const feature = document.getElementById('featureSelect').value;
  if (!target) { toast('Please select a target variable', 'warning'); return; }

  const btn = document.getElementById('setupBtn');
  btn.innerHTML = '<span class="spinner">⟳</span> Configuring…';
  btn.disabled = true;

  const params = {};
  if (S.modelType === 'timeseries') {
    params.ts_model = document.getElementById('tsModel').value;
    params.p = document.getElementById('tsP').value;
    params.d = document.getElementById('tsD').value;
    params.q = document.getElementById('tsQ').value;
  }
  if (S.modelType === 'clustering') {
    params.n_clusters = document.getElementById('nClusters').value;
  }

  try {
    const r = await fetch('/setup_model', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_type: S.modelType, feature_column: feature, target_column: target, model_params: params })
    });
    const d = await r.json();
    btn.innerHTML = '⚙ Confirm Settings'; btn.disabled = false;
    if (d.success) {
      const meta = MODEL_META[S.modelType];
      const badge = document.getElementById('configBadge');
      badge.textContent = d.problem_description;
      badge.style.background = meta.color + '20';
      badge.style.color = meta.color;
      badge.style.border = `1px solid ${meta.color}40`;
      document.getElementById('configDesc').textContent = `Target: ${d.target_column} · Feature: ${d.feature_column || 'auto'}`;
      document.getElementById('modelConfig').style.display = 'block';
      toast('Model configured!', 'success');
    } else toast(d.error || 'Setup failed', 'error');
  } catch(e) { btn.innerHTML = '⚙ Confirm Settings'; btn.disabled = false; toast('Error: ' + e.message, 'error'); }
}

async function trainModel() {
  const btn = document.getElementById('trainBtn');
  btn.innerHTML = '<span class="spinner">⟳</span> Training…'; btn.disabled = true;
  toast('Training model — this may take a moment…', 'info');

  const params = {};
  if (S.modelType === 'timeseries') {
    params.ts_model = document.getElementById('tsModel').value;
    params.p = document.getElementById('tsP').value;
    params.d = document.getElementById('tsD').value;
    params.q = document.getElementById('tsQ').value;
  }
  if (S.modelType === 'clustering') params.n_clusters = document.getElementById('nClusters').value;

  try {
    const r = await fetch('/train_model', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_params: params })
    });
    const d = await r.json();
    btn.innerHTML = ' Train Model'; btn.disabled = false;
    if (d.success) {
      S.modelTrained = true;
      S.metrics = d.metrics;
      S.plotB64 = d.plot;

      renderMetrics('metricsRow3', d.metrics);
      document.getElementById('step3Metrics').style.display = 'block';

      if (d.plot) {
        document.getElementById('step3PlotImg').src = `data:image/png;base64,${d.plot}`;
        document.getElementById('step3Plot').style.display = 'block';
      }

      renderMetrics('metricsRow4', d.metrics);
      if (d.plot) {
        document.getElementById('step4PlotImg').src = `data:image/png;base64,${d.plot}`;
        document.getElementById('step4Plot').style.display = 'block';
      }

      document.getElementById('goStep4Btn').style.display = 'inline-flex';
      toast('Model trained successfully!', 'success');
    } else toast(d.error || 'Training failed', 'error');
  } catch(e) { btn.innerHTML = ' Train Model'; btn.disabled = false; toast('Error: ' + e.message, 'error'); }
}

function renderMetrics(containerId, metrics) {
  const color = MODEL_COLOR[S.modelType] || 'var(--purple-light)';
  const el = document.getElementById(containerId);
  el.innerHTML = Object.entries(metrics)
    .filter(([k]) => k !== 'coefficients' && k !== 'note')
    .map(([k, v]) => `
      <div class="metric-badge">
        <span class="metric-label">${METRIC_LABELS[k] || k}</span>
        <span class="metric-value" style="color:${color}">${v}</span>
      </div>`).join('');
}


async function makePrediction() {
  const val = document.getElementById('predictInput').value.trim();
  if (!val) { toast('Please enter a value', 'warning'); return; }
  const btn = document.getElementById('predictBtn');
  btn.innerHTML = '<span class="spinner">⟳</span>'; btn.disabled = true;

  try {
    const r = await fetch('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feature_value: val })
    });
    const d = await r.json();
    btn.innerHTML = 'Predict'; btn.disabled = false;
    if (d.success) { renderPrediction(d); toast('Prediction complete!', 'success'); }
    else toast(d.error || 'Prediction failed', 'error');
  } catch(e) { btn.innerHTML = 'Predict'; btn.disabled = false; toast('Error: ' + e.message, 'error'); }
}

function renderPrediction(d) {
  const color = MODEL_COLOR[S.modelType] || 'var(--purple-light)';
  let html = '';

  if (d.model_type === 'timeseries') {
    const rows = d.predictions.map(p => `
      <tr>
        <td>${p.period}</td>
        <td style="color:var(--text-muted)">${p.date}</td>
        <td style="color:${color};font-family:'JetBrains Mono',monospace;font-weight:700">${p.formatted}</td>
      </tr>`).join('');
    html = `
      <div class="result-card">
        <p style="font-size:13px;color:var(--text-muted);margin-bottom:12px">Forecast — ${d.steps} periods ahead</p>
        <div style="overflow-x:auto;border-radius:8px">
          <table class="data-table">
            <thead><tr><th>Period</th><th>Date</th><th>Forecast</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <div class="metrics-row" style="margin-top:16px">
          <div class="metric-badge"><span class="metric-label">Average</span><span class="metric-value" style="color:${color}">${d.statistics?.average}</span></div>
          <div class="metric-badge"><span class="metric-label">Trend</span><span class="metric-value" style="color:${d.statistics?.trend==='increasing'?'var(--green-light)':'#f87171'}">${d.statistics?.trend==='increasing'?'↗ Rising':'↘ Falling'}</span></div>
        </div>
      </div>`;

  } else if (d.model_type === 'classification') {
    html = `<div class="result-card">
      <div class="metrics-row">
        <div class="metric-badge"><span class="metric-label">Input</span><span class="metric-value" style="color:var(--text-muted)">${d.input_value}</span></div>
        <div class="metric-badge"><span class="metric-label">Class</span><span class="metric-value" style="color:${color}">${d.result?.prediction}</span></div>
        ${d.result?.confidence ? `<div class="metric-badge"><span class="metric-label">Confidence</span><span class="metric-value" style="color:var(--green-light)">${d.result.confidence}</span></div>` : ''}
      </div></div>`;

  } else if (d.model_type === 'clustering') {
    html = `<div class="result-card">
      <div class="metrics-row">
        <div class="metric-badge"><span class="metric-label">Input</span><span class="metric-value" style="color:var(--text-muted)">${d.input_value}</span></div>
        <div class="metric-badge"><span class="metric-label">Cluster</span><span class="metric-value" style="color:${color}">Cluster ${d.result?.cluster}</span></div>
        <div class="metric-badge"><span class="metric-label">Center</span><span class="metric-value" style="color:var(--text-muted)">${d.result?.center_value?.toFixed(2)}</span></div>
      </div></div>`;

  } else {
    html = `<div class="result-card">
      <div class="metrics-row">
        <div class="metric-badge"><span class="metric-label">Input</span><span class="metric-value" style="color:var(--text-muted)">${d.input_value}</span></div>
        <div class="metric-badge"><span class="metric-label">Prediction</span><span class="metric-value" style="color:${color}">${d.result?.prediction_formatted}</span></div>
      </div></div>`;
  }
  document.getElementById('predResult').innerHTML = html;
}


async function generateReport() {
  const btn = document.getElementById('reportBtn');
  btn.innerHTML = '<span class="spinner">⟳</span> Generating…'; btn.disabled = true;
  try {
    const r = await fetch('/generate_report', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ report_type: 'full', include_explanation: true })
    });
    const d = await r.json();
    btn.innerHTML = 'Generate Report'; btn.disabled = false;
    if (d.success) {
      const rp = d.report;
      const color = MODEL_COLOR[S.modelType] || 'var(--purple-light)';
      const recs = (rp.recommendations || []).map(r => `<div class="rec-item"><span class="rec-arrow">→</span><span class="rec-text">${r}</span></div>`).join('');
      document.getElementById('reportOutput').innerHTML = `
        <div class="card-inner" style="animation:fadeUp 0.3s ease">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px">
            <span style="font-size:20px">📄</span>
            <div>
              <h3 style="margin:0;color:${color}">${rp.project_title}</h3>
              <span style="font-size:12px;color:var(--text-muted)">${rp.report_date}</span>
            </div>
          </div>
          <div class="report-detail">
            <div>
              <p class="report-section-title">Model Info</p>
              <div class="report-row"><span>Type</span><span>${rp.model_info?.model_type}</span></div>
              <div class="report-row"><span>Model</span><span>${rp.model_info?.model_name}</span></div>
              <div class="report-row"><span>Target</span><span>${rp.model_info?.target_variable}</span></div>
              <div class="report-row"><span>Samples</span><span>${rp.model_info?.training_samples}</span></div>
            </div>
            <div>
              <p class="report-section-title">Data Summary</p>
              <div class="report-row"><span>Source</span><span>${rp.data_summary?.source}</span></div>
              <div class="report-row"><span>Rows</span><span>${rp.data_summary?.rows}</span></div>
              <div class="report-row"><span>Columns</span><span>${rp.data_summary?.columns}</span></div>
            </div>
          </div>
          ${recs ? `<div style="margin-top:20px"><p class="report-section-title">Recommendations</p>${recs}</div>` : ''}
        </div>`;
      document.getElementById('reportOutput').style.display = 'block';
      toast('Report generated!', 'success');
    } else toast(d.error || 'Error', 'error');
  } catch(e) { btn.innerHTML = 'Generate Report'; btn.disabled = false; toast('Error: ' + e.message, 'error'); }
}

async function exportNotebook() {
  const btn = document.getElementById('exportBtn');
  btn.innerHTML = '<span class="spinner">⟳</span> Exporting…'; btn.disabled = true;
  try {
    const r = await fetch('/export_notebook', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format: 'python' })
    });
    const d = await r.json();
    btn.innerHTML = 'Export .py'; btn.disabled = false;
    if (d.success) {
      const a = document.createElement('a');
      a.href = d.download_url; a.download = 'financial_analysis.py'; a.click();
      toast('Python file downloaded!', 'success');
    } else toast(d.error || 'Error', 'error');
  } catch(e) { btn.innerHTML = 'Export .py'; btn.disabled = false; toast('Error: ' + e.message, 'error'); }
}

async function compareModels() {
  const btn = document.getElementById('compareBtn');
  btn.innerHTML = '<span class="spinner">⟳</span> Comparing…'; btn.disabled = true;
  try {
    const r = await fetch('/compare_models', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ models: ['regression','classification','timeseries','clustering'] })
    });
    const d = await r.json();
    btn.innerHTML = 'Run Comparison'; btn.disabled = false;
    if (d.success) {
      const cmp = d.comparison;
      const rows = Object.entries(cmp.models_comparison).map(([name, res]) => {
        const perf = res.r2_score ? `R²: ${res.r2_score}` : res.accuracy ? `Acc: ${res.accuracy}` : res.aic ? `AIC: ${res.aic}` : res.inertia ? `Inertia: ${res.inertia}` : '—';
        const best = name === cmp.best_model;
        return `<tr class="${best ? 'best' : ''}">
          <td>${name}${best ? ' ★' : ''}</td>
          <td><span class="model-pill pill-${res.model_type === 'timeseries' ? 'ts' : res.model_type === 'regression' ? 'reg' : res.model_type === 'classification' ? 'cls' : 'clu'}">${res.model_type}</span></td>
          <td style="font-family:'JetBrains Mono',monospace;color:var(--purple-light)">${perf}</td>
          <td style="color:var(--text-muted)">${res.complexity}</td>
          <td style="color:var(--text-muted)">${res.interpretability}</td>
        </tr>`;
      }).join('');
      document.getElementById('compareOutput').innerHTML = `
        <div class="card-inner" style="animation:fadeUp 0.3s ease">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
            <span style="font-size:20px">⚖</span>
            <h3 style="color:var(--green-light);margin:0">Comparison Results</h3>
          </div>
          <div class="metrics-row" style="margin-bottom:16px">
            <div class="metric-badge"><span class="metric-label">Models Tested</span><span class="metric-value" style="color:var(--blue-light)">${cmp.total_models_tested}</span></div>
            <div class="metric-badge"><span class="metric-label">Best Model</span><span class="metric-value" style="color:var(--green-light);font-size:14px">${cmp.best_model}</span></div>
          </div>
          <div style="overflow-x:auto;border-radius:8px">
            <table class="data-table">
              <thead><tr><th>Model</th><th>Type</th><th>Performance</th><th>Complexity</th><th>Interpretability</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
          <p style="margin-top:14px;font-size:13px;color:var(--text-muted);font-style:italic"> ${cmp.recommendation}</p>
        </div>`;
      document.getElementById('compareOutput').style.display = 'block';
      toast('Comparison complete!', 'success');
    } else toast(d.error || 'Error', 'error');
  } catch(e) { btn.innerHTML = 'Run Comparison'; btn.disabled = false; toast('Error: ' + e.message, 'error'); }
}

async function clearAll() {
  if (!confirm('Clear all data and start over?')) return;
  try {
    await fetch('/clear_data', { method: 'POST' });
    Object.assign(S, { step:1, dataUploaded:false, columns:[], dateColumns:[], modelType:null, modelTrained:false, metrics:null, plotB64:null });
    document.getElementById('uploadPreview').style.display = 'none';
    document.getElementById('fileInput').value = '';
    document.getElementById('reportOutput').style.display = 'none';
    document.getElementById('compareOutput').style.display = 'none';
    document.getElementById('predResult').innerHTML = '';
    document.getElementById('predictInput').value = '';
    document.getElementById('modelConfig').style.display = 'none';
    document.getElementById('step3Metrics').style.display = 'none';
    document.getElementById('step3Plot').style.display = 'none';
    document.getElementById('goStep4Btn').style.display = 'none';
    document.getElementById('btnStep2Next').disabled = true;
    document.getElementById('modelInfoBanner').style.display = 'none';
    ['timeseries','regression','classification','clustering'].forEach(t => {
      const c = document.getElementById(`mc-${t}`);
      c.className = 'model-card';
      const b = c.querySelector('.selected-badge');
      if (b) b.remove();
    });
    goToStep(1);
    toast('All data cleared. Ready to start over.', 'info');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

goToStep(1);