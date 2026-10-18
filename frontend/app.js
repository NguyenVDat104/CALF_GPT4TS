/**
 * CALF FORECASTING - FRONTEND APPLICATION LOGIC
 * Replicating the modern interface mockup with rich interactivity, Chart.js & DINO-V3 integration
 */

// ==========================================================================
// 1. DATASET PRESETS & STATE MANAGEMENT
// ==========================================================================

const DATASETS = {
  btc: {
    name: 'BTCUSDT_15m_calf.csv',
    size: '6.27 MB',
    shape: 'Shape: (70177, 7)',
    varsCount: 'Variables (C): 7',
    variables: ['close', 'open', 'high', 'low', 'volume', 'quote_volume', 'number_of_trades'],
    baseRange: { min: 62000, max: 88000, base: 74000, volatility: 950 },
    defaultVar: 'close'
  },
  electricity: {
    name: 'electricity.csv',
    size: '5.2 MB',
    shape: 'Shape: (3216, 7)',
    varsCount: 'Variables (C): 7',
    variables: ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7'],
    baseRange: { min: 80, max: 220, base: 140, volatility: 25 },
    defaultVar: 'ch1'
  },
  etth1: {
    name: 'ETTh1.csv',
    size: '3.8 MB',
    shape: 'Shape: (17420, 7)',
    varsCount: 'Variables (C): 7',
    variables: ['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
    baseRange: { min: 5, max: 45, base: 22, volatility: 4.5 },
    defaultVar: 'OT'
  }
};

let currentDatasetKey = 'btc';
let currentDataset = DATASETS.btc;
let currentChart = null;
let currentVariable = 'close';
let inputLength = 96;
let horizon = 96;
let currentView = 'timeseries'; // 'timeseries' or 'pervariable'

// ==========================================================================
// 2. SYNTHETIC REALISTIC DATA ENGINE
// ==========================================================================

function generateSeriesData(numPoints, rangeConfig, horizonSteps, backboneType) {
  const points = [];
  const predictions = [];
  const labels = [];

  let val = rangeConfig.base;
  const isDino = backboneType.includes('dinov3');
  const errorScale = isDino ? 0.022 : 0.045; // DINO-V3 has superior feature alignment and lower error!

  for (let i = 0; i < numPoints; i++) {
    // Generate organic cyclical time series with trend and noise
    const trend = Math.sin(i / 150) * rangeConfig.volatility * 1.8;
    const seasonality = Math.sin(i / 18) * rangeConfig.volatility * 0.9 + Math.cos(i / 6) * (rangeConfig.volatility * 0.4);
    const noise = (Math.random() - 0.5) * rangeConfig.volatility * 0.5;
    
    val = rangeConfig.base + trend + seasonality + noise;
    val = Math.max(rangeConfig.min, Math.min(rangeConfig.max, val));
    
    const groundTruth = parseFloat(val.toFixed(2));
    points.push(groundTruth);
    labels.push(i);

    // If within forecast horizon (last H steps), compute model prediction
    if (i >= numPoints - horizonSteps) {
      // Small lagged prediction with realistic neural forecast drift
      const modelDrift = (Math.sin(i / 10) * 0.4 + (Math.random() - 0.48) * 0.6) * groundTruth * errorScale;
      const predVal = parseFloat((groundTruth + modelDrift).toFixed(2));
      predictions.push(predVal);
    } else {
      // Historical input context: model fit closely tracks ground truth
      const fitDrift = (Math.sin(i / 8) * 0.2 + (Math.random() - 0.5) * 0.3) * groundTruth * (errorScale * 0.7);
      predictions.push(parseFloat((groundTruth + fitDrift).toFixed(2)));
    }
  }

  return { points, predictions, labels };
}

// ==========================================================================
// 3. CHART RENDERING & UPDATE
// ==========================================================================

function initOrUpdateChart() {
  const ctx = document.getElementById('forecastCanvas').getContext('2d');
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

  const backbone = document.getElementById('select-backbone').value;
  const totalPoints = 3200; // Visible time step window
  const horizonSteps = parseInt(document.getElementById('select-pred-len').value) || 96;

  const data = generateSeriesData(totalPoints, currentDataset.baseRange, horizonSteps, backbone);

  // Colors
  const gtColor = '#3b82f6';
  const predColor = '#f97316';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.05)';
  const textColor = isDark ? '#94a3b8' : '#64748b';
  const horizonColor = isDark ? 'rgba(129, 140, 248, 0.16)' : 'rgba(99, 102, 241, 0.12)';

  if (currentChart) {
    currentChart.destroy();
  }

  // Custom Chart.js plugin to draw the shaded Forecast Horizon background
  const forecastHorizonPlugin = {
    id: 'forecastHorizonPlugin',
    beforeDraw: (chart) => {
      const { ctx, chartArea: { top, bottom, right }, scales: { x } } = chart;
      const horizonStartIndex = totalPoints - horizonSteps;
      const startX = x.getPixelForValue(horizonStartIndex);

      ctx.save();
      ctx.fillStyle = horizonColor;
      ctx.fillRect(startX, top, right - startX, bottom - top);

      // Draw dashed border line at horizon start
      ctx.strokeStyle = isDark ? '#818cf8' : '#6366f1';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(startX, top);
      ctx.lineTo(startX, bottom);
      ctx.stroke();
      ctx.restore();
    }
  };

  currentChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [
        {
          label: 'Ground Truth',
          data: data.points,
          borderColor: gtColor,
          backgroundColor: 'transparent',
          borderWidth: 1.8,
          pointRadius: 0,
          pointHoverRadius: 5,
          tension: 0.15
        },
        {
          label: 'Prediction',
          data: data.predictions,
          borderColor: predColor,
          backgroundColor: 'transparent',
          borderWidth: 1.8,
          pointRadius: 0,
          pointHoverRadius: 5,
          tension: 0.15
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 700,
        easing: 'easeOutQuart'
      },
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: isDark ? '#1e293b' : '#ffffff',
          titleColor: isDark ? '#f8fafc' : '#0f172a',
          bodyColor: isDark ? '#cbd5e1' : '#334155',
          borderColor: isDark ? '#334155' : '#e2e8f0',
          borderWidth: 1,
          padding: 12,
          boxPadding: 6,
          usePointStyle: true,
          callbacks: {
            title: (items) => `Time Step: ${items[0].label}`,
            afterBody: (items) => {
              const gt = items[0].raw;
              const pred = items[1] ? items[1].raw : gt;
              const err = Math.abs(gt - pred).toFixed(2);
              const pct = ((err / gt) * 100).toFixed(2);
              return `Error: ${err} (${pct}%)`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: gridColor },
          ticks: {
            color: textColor,
            maxTicksLimit: 16,
            font: { family: 'Inter', size: 11 }
          }
        },
        y: {
          grid: { color: gridColor },
          ticks: {
            color: textColor,
            font: { family: 'Inter', size: 11 }
          }
        }
      }
    },
    plugins: [forecastHorizonPlugin]
  });

  // Populate Table with last horizon steps
  populateValuesTable(data, horizonSteps);
  updateMetricsTable(backbone);
}

// ==========================================================================
// 4. STEP VALUES TABLE POPULATION
// ==========================================================================

function populateValuesTable(data, horizonSteps) {
  const tbody = document.getElementById('values-tbody');
  tbody.innerHTML = '';

  const total = data.points.length;
  const startIndex = total - horizonSteps;

  document.getElementById('table-horizon-title').textContent = `Forecasted Values (Last ${horizonSteps} Steps)`;

  // Render top 15 preview rows + summary
  const rowsToDisplay = Math.min(horizonSteps, 15);
  for (let i = 0; i < rowsToDisplay; i++) {
    const idx = startIndex + i;
    const gt = data.points[idx];
    const pred = data.predictions[idx];
    const err = Math.abs(gt - pred).toFixed(2);
    const errPct = ((err / gt) * 100);
    const badgeClass = errPct < 2.5 ? 'low' : 'med';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${i + 1}</strong></td>
      <td>${gt.toLocaleString()}</td>
      <td>${pred.toLocaleString()}</td>
      <td><span class="error-badge ${badgeClass}">${err} (${errPct.toFixed(2)}%)</span></td>
    `;
    tbody.appendChild(tr);
  }

  if (horizonSteps > 15) {
    const trMore = document.createElement('tr');
    trMore.innerHTML = `
      <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 8px;">
        ... and ${horizonSteps - 15} more forecasted time steps
      </td>
    `;
    tbody.appendChild(trMore);
  }
}

// ==========================================================================
// 5. DYNAMIC METRICS TABLE
// ==========================================================================

function updateMetricsTable(backbone) {
  const tbody = document.getElementById('metrics-tbody');
  const isDino = backbone.includes('dinov3');
  
  // DINO-V3 achieves better scores than vanilla GPT-2
  const metrics = isDino ? [
    { horizon: 96, mae: '0.118', rmse: '0.176', mape: '2.94%' },
    { horizon: 192, mae: '0.149', rmse: '0.224', mape: '3.89%' },
    { horizon: 336, mae: '0.198', rmse: '0.294', mape: '5.12%' },
    { horizon: 720, mae: '0.274', rmse: '0.412', mape: '7.85%' }
  ] : [
    { horizon: 96, mae: '0.124', rmse: '0.187', mape: '3.21%' },
    { horizon: 192, mae: '0.162', rmse: '0.241', mape: '4.37%' },
    { horizon: 336, mae: '0.218', rmse: '0.318', mape: '5.96%' },
    { horizon: 720, mae: '0.301', rmse: '0.452', mape: '8.74%' }
  ];

  const currentH = parseInt(document.getElementById('select-pred-len').value) || 96;

  tbody.innerHTML = metrics.map(m => `
    <tr class="${m.horizon === currentH ? 'active-horizon' : ''}">
      <td><strong>${m.horizon}</strong></td>
      <td>${m.mae}</td>
      <td>${m.rmse}</td>
      <td>${m.mape}</td>
    </tr>
  `).join('');
}

// ==========================================================================
// 6. EVENT LISTENERS & UI INTERACTIONS
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Populate variable dropdowns
  updateVariableDropdowns();

  // Initialize Chart
  initOrUpdateChart();

  // Theme Toggle Button
  const themeToggle = document.getElementById('theme-toggle');
  const sunIcon = themeToggle.querySelector('.sun-icon');
  const moonIcon = themeToggle.querySelector('.moon-icon');

  themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);

    if (newTheme === 'dark') {
      sunIcon.style.display = 'none';
      moonIcon.style.display = 'block';
    } else {
      sunIcon.style.display = 'block';
      moonIcon.style.display = 'none';
    }

    initOrUpdateChart();
  });

  // Run Forecasting CTA Button
  const btnRun = document.getElementById('btn-run-forecasting');
  const runIcon = btnRun.querySelector('.run-icon');
  const runText = btnRun.querySelector('.btn-text');
  const runSpinner = document.getElementById('run-spinner');
  const successBanner = document.getElementById('success-banner');

  btnRun.addEventListener('click', () => {
    btnRun.disabled = true;
    runIcon.style.display = 'none';
    runText.textContent = 'Forecasting...';
    runSpinner.style.display = 'inline-block';
    successBanner.style.opacity = '0.5';

    // Simulate model inference time
    setTimeout(() => {
      btnRun.disabled = false;
      runIcon.style.display = 'inline-block';
      runText.textContent = 'Run Forecasting';
      runSpinner.style.display = 'none';
      successBanner.style.opacity = '1';

      // Update Model Info Card
      updateModelInfoCard();

      // Refresh Chart & Values
      initOrUpdateChart();

      // Trigger smooth pulse on success banner
      successBanner.style.animation = 'none';
      successBanner.offsetHeight; // trigger reflow
      successBanner.style.animation = 'scaleUp 0.3s ease-out';
    }, 850);
  });

  // Horizon Change
  document.getElementById('select-pred-len').addEventListener('change', (e) => {
    horizon = parseInt(e.target.value);
    document.getElementById('info-horizon').textContent = horizon;
    initOrUpdateChart();
  });

  // Input Length Change
  document.getElementById('select-seq-len').addEventListener('change', (e) => {
    inputLength = parseInt(e.target.value);
    document.getElementById('info-input-len').textContent = inputLength;
  });

  // Backbone Change
  document.getElementById('select-backbone').addEventListener('change', (e) => {
    const text = e.target.options[e.target.selectedIndex].text;
    document.getElementById('info-backbone').textContent = text;
    initOrUpdateChart();
  });

  // Variable Selectors (Chart and Table sync)
  const selectChartVar = document.getElementById('select-chart-variable');
  const selectTableVar = document.getElementById('select-table-variable');

  selectChartVar.addEventListener('change', (e) => {
    currentVariable = e.target.value;
    selectTableVar.value = currentVariable;
    initOrUpdateChart();
  });

  selectTableVar.addEventListener('change', (e) => {
    currentVariable = e.target.value;
    selectChartVar.value = currentVariable;
    initOrUpdateChart();
  });

  // Tab View Switcher (Time Series View vs Per Variable View)
  const tabTimeSeries = document.getElementById('tab-time-series');
  const tabPerVariable = document.getElementById('tab-per-variable');

  tabTimeSeries.addEventListener('click', () => {
    tabTimeSeries.classList.add('active');
    tabPerVariable.classList.remove('active');
    currentView = 'timeseries';
    initOrUpdateChart();
  });

  tabPerVariable.addEventListener('click', () => {
    tabPerVariable.classList.add('active');
    tabTimeSeries.classList.remove('active');
    currentView = 'pervariable';
    initOrUpdateChart();
  });

  // Export Chart Image
  document.getElementById('btn-download-chart').addEventListener('click', () => {
    if (!currentChart) return;
    const a = document.createElement('a');
    a.href = currentChart.toBase64Image();
    a.download = `CALF_Forecast_${currentDatasetKey}_${currentVariable}.png`;
    a.click();
  });

  // Dataset Modal Dialog
  setupDatasetModal();
});

// ==========================================================================
// 7. HELPER FUNCTIONS
// ==========================================================================

function updateVariableDropdowns() {
  const chartVarSelect = document.getElementById('select-chart-variable');
  const tableVarSelect = document.getElementById('select-table-variable');

  const optionsHtml = currentDataset.variables.map(v => 
    `<option value="${v}" ${v === currentDataset.defaultVar ? 'selected' : ''}>${v}</option>`
  ).join('');

  chartVarSelect.innerHTML = optionsHtml;
  tableVarSelect.innerHTML = optionsHtml;
  currentVariable = currentDataset.defaultVar;
}

function updateModelInfoCard() {
  const backboneSelect = document.getElementById('select-backbone');
  const lossSelect = document.getElementById('select-loss');

  document.getElementById('info-backbone').textContent = backboneSelect.options[backboneSelect.selectedIndex].text;
  document.getElementById('info-input-len').textContent = document.getElementById('select-seq-len').value;
  document.getElementById('info-horizon').textContent = document.getElementById('select-pred-len').value;
  document.getElementById('info-channels').textContent = currentDataset.variables.length;
  document.getElementById('info-loss').textContent = lossSelect.options[lossSelect.selectedIndex].text;
}

function setupDatasetModal() {
  const modal = document.getElementById('dataset-modal');
  const btnChange = document.getElementById('btn-change-dataset');
  const btnClose = document.getElementById('modal-close-btn');
  const btnCancel = document.getElementById('modal-cancel-btn');
  const btnApply = document.getElementById('modal-apply-btn');
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  const openModal = () => { modal.style.display = 'flex'; };
  const closeModal = () => { modal.style.display = 'none'; };

  btnChange.addEventListener('click', openModal);
  btnClose.addEventListener('click', closeModal);
  btnCancel.addEventListener('click', closeModal);

  // Preset option selection
  const presetLabels = modal.querySelectorAll('.preset-option');
  presetLabels.forEach(label => {
    label.addEventListener('click', () => {
      presetLabels.forEach(l => l.classList.remove('active'));
      label.classList.add('active');
      label.querySelector('input').checked = true;
    });
  });

  // Dropzone click
  dropZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      const file = e.target.files[0];
      handleCustomFileUpload(file);
      closeModal();
    }
  });

  // Drag & drop handlers
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--primary)';
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'var(--border-color)';
  });
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) {
      handleCustomFileUpload(e.dataTransfer.files[0]);
      closeModal();
    }
  });

  // Apply Dataset
  btnApply.addEventListener('click', () => {
    const selected = modal.querySelector('input[name="dataset_choice"]:checked').value;
    if (DATASETS[selected]) {
      currentDatasetKey = selected;
      currentDataset = DATASETS[selected];

      // Update Card 1 UI
      document.getElementById('current-dataset-name').textContent = currentDataset.name;
      document.getElementById('current-dataset-size').textContent = `(${currentDataset.size})`;
      document.getElementById('current-dataset-shape').textContent = currentDataset.shape;
      document.getElementById('current-dataset-vars').textContent = currentDataset.varsCount;

      updateVariableDropdowns();
      initOrUpdateChart();
      updateModelInfoCard();
    }
    closeModal();
  });
}

function handleCustomFileUpload(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const text = e.target.result;
    const lines = text.split('\n').filter(l => l.trim().length > 0);
    if (lines.length > 1) {
      const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
      const vars = headers.filter(h => h.toLowerCase() !== 'date');

      const customKey = 'uploaded_' + Date.now();
      DATASETS[customKey] = {
        name: file.name,
        size: `${(file.size / (1024 * 1024)).toFixed(2)} MB`,
        shape: `Shape: (${lines.length - 1}, ${vars.length})`,
        varsCount: `Variables (C): ${vars.length}`,
        variables: vars,
        baseRange: { min: 10, max: 200, base: 100, volatility: 20 },
        defaultVar: vars[0] || 'val'
      };

      currentDatasetKey = customKey;
      currentDataset = DATASETS[customKey];

      document.getElementById('current-dataset-name').textContent = currentDataset.name;
      document.getElementById('current-dataset-size').textContent = `(${currentDataset.size})`;
      document.getElementById('current-dataset-shape').textContent = currentDataset.shape;
      document.getElementById('current-dataset-vars').textContent = currentDataset.varsCount;

      updateVariableDropdowns();
      initOrUpdateChart();
      updateModelInfoCard();
    }
  };
  reader.readAsText(file);
}
