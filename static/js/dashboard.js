/**
 * dashboard.js — NoteGuard Evaluation Dashboard Controller
 * Loads metrics from /api/metrics and renders comparison cards,
 * styled confusion matrices, Chart.js graphs, and training distribution.
 */

document.addEventListener("DOMContentLoaded", async () => {
  // Adjust navigation links if running under static VS Code Live Server (port 5500)
  if (window.location.port === "5500" || window.location.protocol === "file:") {
    document.querySelectorAll('a[href="/"]').forEach(a => a.href = "index.html");
    document.querySelectorAll('a[href="/dashboard"]').forEach(a => a.href = "dashboard.html");
    document.querySelectorAll('a[href="/security"]').forEach(a => a.href = "security.html");
    document.querySelectorAll('a[href="/about"]').forEach(a => a.href = "about.html");
  }

  const API_BASE = (window.location.port === "5500" || window.location.protocol === "file:")
    ? "http://127.0.0.1:5000"
    : "";

  // Hamburger menu toggle
  const hamburger = document.getElementById("navHamburger");
  const navLinks = document.getElementById("navLinks");
  if (hamburger && navLinks) {
    hamburger.addEventListener("click", () => navLinks.classList.toggle("open"));
  }

  try {
    const resp = await fetch(`${API_BASE}/api/metrics`);
    if (!resp.ok) {
      throw new Error(`Failed to load metrics (${resp.status})`);
    }
    const metricsData = await resp.json();
    renderDashboard(metricsData);
  } catch (err) {
    console.warn("Using built-in metrics data:", err);
    // Built-in fallback data matching actual Philippine Banknote Dataset metrics
    const fallbackMetrics = {
      best_model: "random_forest",
      models: {
        decision_tree: {
          model_name: "Decision Tree",
          accuracy: 0.8634,
          precision: 0.8559,
          recall: 0.9135,
          f1_score: 0.8837,
          confusion_matrix: [[63, 16], [9, 95]],
          feature_importances: { variance: 0.0447, skewness: 0.3756, curtosis: 0.0276, entropy: 0.3569, r_mean: 0.0652, g_mean: 0.0, b_mean: 0.0538, r_std: 0.0272, g_std: 0.0296, b_std: 0.0194 },
          best_params: { criterion: "gini", max_depth: 6, min_samples_split: 5 }
        },
        random_forest: {
          model_name: "Random Forest",
          accuracy: 0.8743,
          precision: 0.9010,
          recall: 0.8750,
          f1_score: 0.8878,
          confusion_matrix: [[69, 10], [13, 91]],
          feature_importances: { variance: 0.0638, skewness: 0.2322, curtosis: 0.0589, entropy: 0.2272, r_mean: 0.1237, g_mean: 0.0740, b_mean: 0.0711, r_std: 0.0471, g_std: 0.0565, b_std: 0.0454 },
          best_params: { n_estimators: 100, max_depth: 15, min_samples_split: 2, max_features: "sqrt" }
        }
      },
      dataset_summary: {
        total_records: 868,
        genuine_count: 434,
        forged_count: 434,
        missing_values: 0
      },
      training_feature_ranges: {
        variance:  { min: 0.0072, max: 0.3985, mean: 0.1613, std: 0.0742 },
        skewness:  { min: -3.7921, max: 2.5818, mean: -0.5662, std: 0.6670 },
        curtosis:  { min: -1.6877, max: 14.8199, mean: 0.1571, std: 1.5719 },
        entropy:   { min: -3.4200, max: 0.8424, mean: 0.0246, std: 0.5628 },
        r_mean:    { min: 0.0892, max: 0.8757, mean: 0.5641, std: 0.1354 },
        g_mean:    { min: 0.0864, max: 0.8638, mean: 0.5367, std: 0.1275 },
        b_mean:    { min: 0.0592, max: 0.8361, mean: 0.4998, std: 0.1286 },
        r_std:     { min: 0.0233, max: 0.3501, mean: 0.1914, std: 0.0553 },
        g_std:     { min: 0.0500, max: 0.3102, mean: 0.1943, std: 0.0466 },
        b_std:     { min: 0.0455, max: 0.3073, mean: 0.1954, std: 0.0456 }
      },
      test_set_size: 183,
      train_set_size: 685
    };
    renderDashboard(fallbackMetrics);
  }
});

function renderDashboard(data) {
  const models = data.models || {};
  const dt = models.decision_tree || {};
  const rf = models.random_forest || {};
  const bestKey = data.best_model || "random_forest";
  const bestModel = models[bestKey] || rf;

  // 1. Dataset summary
  const summary = data.dataset_summary || {};
  if (summary.total_records) {
    document.getElementById("totalRecords").textContent = summary.total_records.toLocaleString();
  }
  if (summary.genuine_count) {
    document.getElementById("genuineCount").textContent = summary.genuine_count.toLocaleString();
  }
  if (summary.forged_count) {
    document.getElementById("forgedCount").textContent = summary.forged_count.toLocaleString();
  }
  if (data.train_set_size && data.test_set_size) {
    document.getElementById("splitRatio").textContent = `${data.train_set_size} / ${data.test_set_size}`;
  }

  // Update test specimen counts dynamically
  const dtTestCount = document.getElementById("dtTestCount");
  const rfTestCount = document.getElementById("rfTestCount");
  if (dtTestCount && data.test_set_size) dtTestCount.textContent = data.test_set_size;
  if (rfTestCount && data.test_set_size) rfTestCount.textContent = data.test_set_size;

  // 2. Best Model Highlight
  document.getElementById("bestModelTitle").textContent = `${bestModel.model_name} (F1: ${(bestModel.f1_score * 100).toFixed(2)}%)`;
  document.getElementById("bestModelAcc").textContent = `${(bestModel.accuracy * 100).toFixed(2)}%`;
  document.getElementById("bestModelPrec").textContent = `${(bestModel.precision * 100).toFixed(2)}%`;
  document.getElementById("bestModelRec").textContent = `${(bestModel.recall * 100).toFixed(2)}%`;
  document.getElementById("bestModelF1").textContent = `${(bestModel.f1_score * 100).toFixed(2)}%`;

  // 3. Side-by-Side Model Cards
  // Decision Tree
  document.getElementById("dtAcc").textContent = `${(dt.accuracy * 100).toFixed(2)}%`;
  document.getElementById("dtPrec").textContent = `${(dt.precision * 100).toFixed(2)}%`;
  document.getElementById("dtRec").textContent = `${(dt.recall * 100).toFixed(2)}%`;
  document.getElementById("dtF1").textContent = `${(dt.f1_score * 100).toFixed(2)}%`;

  if (dt.confusion_matrix) {
    // [[TN, FP], [FN, TP]]
    document.getElementById("dtTN").textContent = dt.confusion_matrix[0][0];
    document.getElementById("dtFP").textContent = dt.confusion_matrix[0][1];
    document.getElementById("dtFN").textContent = dt.confusion_matrix[1][0];
    document.getElementById("dtTP").textContent = dt.confusion_matrix[1][1];
  }

  if (dt.best_params) {
    const p = dt.best_params;
    document.getElementById("dtParams").textContent =
      `Best Hyperparameters: Criterion=${p.criterion || "gini"}, Max Depth=${p.max_depth || "None"}, Min Split=${p.min_samples_split || 2}`;
  }

  // Random Forest
  document.getElementById("rfAcc").textContent = `${(rf.accuracy * 100).toFixed(2)}%`;
  document.getElementById("rfPrec").textContent = `${(rf.precision * 100).toFixed(2)}%`;
  document.getElementById("rfRec").textContent = `${(rf.recall * 100).toFixed(2)}%`;
  document.getElementById("rfF1").textContent = `${(rf.f1_score * 100).toFixed(2)}%`;

  if (rf.confusion_matrix) {
    // [[TN, FP], [FN, TP]]
    document.getElementById("rfTN").textContent = rf.confusion_matrix[0][0];
    document.getElementById("rfFP").textContent = rf.confusion_matrix[0][1];
    document.getElementById("rfFN").textContent = rf.confusion_matrix[1][0];
    document.getElementById("rfTP").textContent = rf.confusion_matrix[1][1];
  }

  if (rf.best_params) {
    const p = rf.best_params;
    document.getElementById("rfParams").textContent =
      `Best Hyperparameters: N-Estimators=${p.n_estimators || 100}, Max Depth=${p.max_depth || "None"}, Min Split=${p.min_samples_split || 2}, Features=${p.max_features || "sqrt"}`;
  }

  // Highlight the best model card
  if (bestKey === "random_forest") {
    document.getElementById("cardRandomForest").classList.add("best");
    document.getElementById("rfBadge").style.display = "inline-flex";
  } else {
    document.getElementById("cardDecisionTree").classList.add("best");
    document.getElementById("dtBadge").style.display = "inline-flex";
  }

  // 4. Render Chart.js Visualizations
  renderPerformanceChart(dt, rf);
  renderFeatureImportanceChart(dt, rf);

  // 5. Training Distribution Table
  renderDistributionTable(data.training_feature_ranges || {});
}

function renderPerformanceChart(dt, rf) {
  if (typeof Chart === "undefined") {
    console.warn("Chart.js not loaded. Skipping performance chart.");
    return;
  }
  const canvas = document.getElementById("metricsComparisonChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  // Calculate dynamic Y-axis minimum
  const allValues = [
    dt.accuracy * 100, dt.precision * 100, dt.recall * 100, dt.f1_score * 100,
    rf.accuracy * 100, rf.precision * 100, rf.recall * 100, rf.f1_score * 100,
  ];
  const minVal = Math.min(...allValues);
  const yMin = Math.max(0, Math.floor((minVal - 10) / 5) * 5); // Round down to nearest 5, at least 0

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Accuracy", "Precision", "Recall", "F1-Score"],
      datasets: [
        {
          label: "Decision Tree",
          data: [
            (dt.accuracy * 100).toFixed(2),
            (dt.precision * 100).toFixed(2),
            (dt.recall * 100).toFixed(2),
            (dt.f1_score * 100).toFixed(2),
          ],
          backgroundColor: "rgba(16, 185, 129, 0.7)",
          borderColor: "#10b981",
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: "Random Forest",
          data: [
            (rf.accuracy * 100).toFixed(2),
            (rf.precision * 100).toFixed(2),
            (rf.recall * 100).toFixed(2),
            (rf.f1_score * 100).toFixed(2),
          ],
          backgroundColor: "rgba(245, 158, 11, 0.75)",
          borderColor: "#f59e0b",
          borderWidth: 1.5,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#e2e8f0", font: { family: "'IBM Plex Sans', sans-serif" } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}%`,
          },
        },
      },
      scales: {
        y: {
          min: yMin,
          max: 100,
          grid: { color: "rgba(255, 255, 255, 0.06)" },
          ticks: {
            color: "#94a3b8",
            callback: (val) => `${val}%`,
          },
        },
        x: {
          grid: { display: false },
          ticks: { color: "#94a3b8" },
        },
      },
    },
  });
}

function renderFeatureImportanceChart(dt, rf) {
  if (typeof Chart === "undefined") {
    console.warn("Chart.js not loaded. Skipping feature importance chart.");
    return;
  }
  const canvas = document.getElementById("featureImportanceChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const features = Object.keys(rf.feature_importances || dt.feature_importances || {});
  if (features.length === 0) return;

  const dtImp = features.map((f) => ((dt.feature_importances && dt.feature_importances[f]) || 0) * 100);
  const rfImp = features.map((f) => ((rf.feature_importances && rf.feature_importances[f]) || 0) * 100);
  const labels = features.map((f) => f.toUpperCase());

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Decision Tree (%)",
          data: dtImp.map((v) => v.toFixed(2)),
          backgroundColor: "rgba(16, 185, 129, 0.7)",
          borderColor: "#10b981",
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: "Random Forest (%)",
          data: rfImp.map((v) => v.toFixed(2)),
          backgroundColor: "rgba(245, 158, 11, 0.75)",
          borderColor: "#f59e0b",
          borderWidth: 1.5,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      plugins: {
        legend: {
          labels: { color: "#e2e8f0", font: { family: "'IBM Plex Sans', sans-serif" } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}%`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.06)" },
          ticks: {
            color: "#94a3b8",
            callback: (val) => `${val}%`,
          },
        },
        y: {
          grid: { display: false },
          ticks: { color: "#94a3b8" },
        },
      },
    },
  });
}

function renderDistributionTable(ranges) {
  const tbody = document.getElementById("distributionTableBody");
  let html = "";
  for (const [feat, stats] of Object.entries(ranges)) {
    html += `
      <tr>
        <td><strong>${feat.toUpperCase()}</strong></td>
        <td>${stats.min.toFixed(4)}</td>
        <td>${stats.max.toFixed(4)}</td>
        <td>${stats.mean.toFixed(4)}</td>
        <td>${stats.std.toFixed(4)}</td>
      </tr>
    `;
  }
  tbody.innerHTML = html;
}
