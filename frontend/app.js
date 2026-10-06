const $ = id => document.getElementById(id);
const agentDefs = [
  { name: "Coordinator Agent", desc: "Pipeline orchestration & task routing", activeMsg: "Orchestrating workflow" },
  { name: "Data Profiling Agent", desc: "Schema, types, nulls & duplicates detection", activeMsg: "Profiling data structure" },
  { name: "Cleaning Agent", desc: "Outlier handling & feature normalization", activeMsg: "Standardizing features" },
  { name: "EDA & Visualizations Agent", desc: "Histograms, category charts, heatmaps, box plots & trends", activeMsg: "Computing 7 visual analytics sets" },
  { name: "Statistical Agent", desc: "Student's t, Pearson correlations & omnibus normality", activeMsg: "Running statistical significance tests" },
  { name: "ML Multi-Model Suite", desc: "18+ models including Linear & RBF SVM, CART, Boosted Trees", activeMsg: "Training 18 ML algorithms" },
  { name: "Validation Agent", desc: "Integrity, convergence & performance validation", activeMsg: "Verifying benchmarks" },
  { name: "Insight Agent", desc: "Empirical discovery & narrative generation", activeMsg: "Extracting strategic insights" },
  { name: "Gemini AI Conclusion Agent", desc: "Executive analytical synthesis & business recommendations", activeMsg: "Synthesizing AI conclusion" },
  { name: "PDF Report Agent", desc: "Publication-grade executive PDF compilation", activeMsg: "Compiling formal PDF report" }
];
const agentNames = agentDefs.map(a => a.name);

let currentProfile = null;
let currentVisualizations = null;
let currentMLData = null;

// Chart instances
let distChart = null;
let catChart = null;
let corrChart = null;
let mlChart = null;
let boxChart = null;
let scatterChart = null;
let trendChart = null;

// ========================================================
// PAGE DEFINITIONS & METADATA
// ========================================================
const pageMeta = {
  "page-overview": {
    heading: "Overview & Ingestion",
    subheading: "Upload, profile, and validate datasets with automated multi-agent coordination."
  },
  "page-visualizations": {
    heading: "Comprehensive Visual Analytics",
    subheading: "Simultaneous distributions, categorical breakdowns, correlation heatmaps, box plots, scatter correlations, trend lines, and ML benchmarks."
  },
  "page-ml": {
    heading: "Machine Learning Suite",
    subheading: "18+ algorithms evaluated including Support Vector Machines (Linear & RBF SVM), Trees, Ensembles, and Probabilistic models."
  },
  "page-conclusion": {
    heading: "Gemini AI Analysed Conclusion",
    subheading: "Executive synthesis of statistical drivers, risk vectors, and predictive conclusions."
  },
  "page-query": {
    heading: "Ask Your Data (Q&A)",
    subheading: "Natural language query engine powered by Google Gemini dynamic discovery."
  },
  "page-schema": {
    heading: "Data Schema & Profile",
    subheading: "Field types, missing value distributions, and cardinality statistics."
  },
  "page-report": {
    heading: "Executive PDF Report",
    subheading: "Export formal publication-grade PDF report with complete tables and Gemini conclusion."
  }
};

// ========================================================
// LEFT SIDEBAR NAVIGATION & PAGE SWITCHING
// ========================================================
function switchPage(pageId) {
  // Update nav buttons
  document.querySelectorAll(".nav-item").forEach(btn => {
    if (btn.getAttribute("data-page") === pageId) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Update visible view page
  document.querySelectorAll(".view-page").forEach(page => {
    if (page.id === pageId) {
      page.classList.add("active");
    } else {
      page.classList.remove("active");
    }
  });

  // Update header text
  if (pageMeta[pageId]) {
    $("pageHeading").textContent = pageMeta[pageId].heading;
    $("pageSubheading").textContent = pageMeta[pageId].subheading;
  }

  // Trigger chart resize if navigating to visual analytics
  if (pageId === "page-visualizations") {
    setTimeout(() => {
      if (distChart) distChart.resize();
      if (catChart) catChart.resize();
      if (corrChart) corrChart.resize();
      if (mlChart) mlChart.resize();
      if (boxChart) boxChart.resize();
      if (scatterChart) scatterChart.resize();
      if (trendChart) trendChart.resize();
    }, 50);
  }

  window.scrollTo({ top: 0, behavior: "smooth" });
}

// Bind click handlers to left-side navigation buttons
document.querySelectorAll(".nav-item").forEach(item => {
  item.onclick = () => {
    const targetPage = item.getAttribute("data-page");
    if (targetPage) switchPage(targetPage);
  };
});

// ========================================================
// AGENT STATUS RENDERING
// ========================================================
function renderAgents(active = -1, done = -1) {
  $("agents").innerHTML = agentDefs.map((agent, i) => {
    let state = "READY";
    let cls = "ready";
    let checkIcon = "";

    if (i <= done) {
      state = "COMPLETED";
      cls = "done";
      checkIcon = "✓";
    } else if (i === active) {
      state = agent.activeMsg || "RUNNING";
      cls = "active";
      checkIcon = "●";
    } else if (active >= 0) {
      state = "QUEUED";
      cls = "queued";
      checkIcon = "○";
    } else {
      state = "READY";
      cls = "ready";
      checkIcon = "•";
    }

    return `<div class="agent ${cls}">
      <span class="check">${checkIcon}</span>
      <div class="agent-info">
        <span class="agent-name">${agent.name}</span>
        <span class="agent-desc">${agent.desc}</span>
      </div>
      <span class="agent-state">${state}</span>
    </div>`;
  }).join("");
}
renderAgents();

async function api(url, options = {}) {
  const r = await fetch(url, options);
  const data = await r.json().catch(() => ({ detail: "Invalid server response" }));
  if (!r.ok) throw new Error(data.detail || "Request failed");
  return data;
}

// ========================================================
// HEALTH & DYNAMIC GEMINI STATUS
// ========================================================
async function checkHealthAndAI() {
  try {
    const health = await api("/api/health");
    const ai = await api("/api/ai-status");
    
    if (ai.configured && ai.models_count > 0) {
      $("statusText").textContent = `Gemini Online (${ai.models_count} models)`;
    } else if (ai.configured) {
      $("statusText").textContent = "Gemini Ready (.env)";
    } else {
      $("statusText").textContent = "Local Analytics (No Key)";
    }
  } catch (e) {
    $("statusText").textContent = "Backend Offline";
  }
}
checkHealthAndAI();

function escapeHtml(v) {
  return String(v ?? "").replace(/[&<>"']/g, m => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[m]));
}

// Simple markdown formatter for Gemini AI Conclusion
function formatMarkdown(text) {
  if (!text) return "";
  let html = escapeHtml(text);
  // Headings
  html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
  html = html.replace(/^## (.*$)/gim, "<h3>$1</h3>");
  // Bold
  html = html.replace(/\*\*(.*?)\*\*/g, "<b>$1</b>");
  // Numbered list items
  html = html.replace(/^\d+\.\s+(.*$)/gim, "<li>$1</li>");
  // Wrap list items
  html = html.replace(/(<li>.*?<\/li>)/gs, "<ol>$1</ol>");
  // Paragraph breaks
  html = html.split("\n\n").map(p => {
    p = p.trim();
    if (!p) return "";
    if (p.startsWith("<h3>") || p.startsWith("<ol>")) return p;
    return `<p>${p}</p>`;
  }).join("");
  return html;
}

// ========================================================
// DATA PROFILING & SUMMARY
// ========================================================
function showProfile(profile) {
  if (!profile) return;
  currentProfile = profile;

  $("rows").textContent = (profile.rows ?? 0).toLocaleString();
  $("cols").textContent = profile.columns ?? 0;
  $("missing").textContent = (profile.missing_total ?? 0).toLocaleString();
  $("duplicates").textContent = (profile.duplicates ?? 0).toLocaleString();

  const types = {};
  (profile.numeric_columns || []).forEach(x => types[x] = "numeric");
  (profile.categorical_columns || []).forEach(x => types[x] = "categorical");
  (profile.date_columns || []).forEach(x => types[x] = "date");

  const missingMap = profile.missing_by_column || {};
  const uniqueMap = profile.unique_by_column || {};

  $("profileTable").innerHTML = (profile.column_names || []).map(c => `
    <tr>
      <td><b>${escapeHtml(c)}</b></td>
      <td><span class="type-badge ${types[c] || 'other'}">${types[c] || "other"}</span></td>
      <td>${(missingMap[c] ?? 0).toLocaleString()}</td>
      <td>${uniqueMap[c] !== undefined ? (uniqueMap[c]).toLocaleString() : "—"}</td>
    </tr>
  `).join("");

  $("analyzeBtn").disabled = false;
}

// ========================================================
// FILE UPLOAD & AUTOMATIC MULTI-AGENT ORCHESTRATION
// ========================================================
$("fileInput").addEventListener("change", async e => {
  const file = e.target.files[0];
  if (!file) return;

  $("fileName").textContent = "Ingesting " + file.name + "...";
  $("workflowState").textContent = "PROFILING DATASET";
  renderAgents(1, 0);

  try {
    const fd = new FormData();
    fd.append("file", file);
    const data = await api("/api/upload", { method: "POST", body: fd });
    $("fileName").textContent = "Loaded: " + data.name;
    $("activeDatasetBadge").textContent = data.name;
    showProfile(data.profile);
    renderAgents(2, 1);

    // Automatically trigger full analysis so agents never wait idle
    await runFullAnalysis();
  } catch (err) {
    $("fileName").textContent = "Error: " + err.message;
    $("workflowState").textContent = "ERROR";
    renderAgents(-1, -1);
  }
});

// ========================================================
// FULL MULTI-AGENT ANALYSIS PIPELINE
// ========================================================
async function runFullAnalysis() {
  $("workflowState").textContent = "RUNNING MULTI-AGENT PIPELINE";
  $("analyzeBtn").disabled = true;

  let currentStep = 2;
  renderAgents(currentStep, 1);

  const progressInterval = setInterval(() => {
    if (currentStep < agentDefs.length - 1) {
      currentStep++;
      renderAgents(currentStep, currentStep - 1);
    }
  }, 750);

  try {
    const result = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target: null })
    });

    clearInterval(progressInterval);
    renderAgents(-1, agentDefs.length - 1);
    $("workflowState").textContent = "COMPLETE (10/10 AGENTS FINISHED)";
    $("analyzeBtn").disabled = false;

    // 1. Insights
    if (result.insights && result.insights.length) {
      $("insights").innerHTML = result.insights.map(x => `<li>${escapeHtml(x)}</li>`).join("");
    }

    // 2. All Visualizations Rendered
    if (result.eda && result.eda.visualizations) {
      currentVisualizations = result.eda.visualizations;
      initAllCharts(currentVisualizations);
    }

    // 3. Machine Learning Suite (18+ Models)
    if (result.ml) {
      currentMLData = result.ml;
      renderML(result.ml);
      if (result.ml.leaderboard_chart) {
        renderMLChart(result.ml.leaderboard_chart);
      }
    }

    // 4. Gemini AI Analysed Conclusion
    if (result.conclusion) {
      $("conclusionEngineBadge").textContent = result.conclusion.engine || "Gemini AI";
      $("conclusionContent").innerHTML = formatMarkdown(result.conclusion.conclusion || "");
    }

    // 5. PDF Report Link Ready
    $("reportLink").classList.remove("disabled");

    // Seamlessly navigate to Visual Analytics after brief delay
    setTimeout(() => {
      switchPage("page-visualizations");
    }, 700);

  } catch (err) {
    clearInterval(progressInterval);
    $("workflowState").textContent = "ERROR";
    $("analyzeBtn").disabled = false;
    alert("Analysis failed: " + err.message);
  }
}

$("analyzeBtn").onclick = () => runFullAnalysis();

// ========================================================
// ALL VISUALIZATIONS RENDERING
// ========================================================
function initAllCharts(viz) {
  // 1. Feature Distributions
  const distSelect = $("distColSelect");
  distSelect.innerHTML = "";
  const distKeys = Object.keys(viz.distributions || {});
  distKeys.forEach(k => {
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = k;
    distSelect.appendChild(opt);
  });
  if (distKeys.length > 0) {
    renderDistChart(distKeys[0]);
  }
  distSelect.onchange = () => renderDistChart(distSelect.value);

  // 2. Categorical Breakdown
  const catSelect = $("catColSelect");
  catSelect.innerHTML = "";
  const catKeys = Object.keys(viz.categories || {});
  catKeys.forEach(k => {
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = k;
    catSelect.appendChild(opt);
  });
  if (catKeys.length > 0) {
    renderCatChart(catKeys[0]);
  }
  catSelect.onchange = () => renderCatChart(catSelect.value);

  // 3. Correlation Matrix Heatmap / Bar
  if (viz.correlations && viz.correlations.length > 0) {
    renderCorrChart(viz.correlations);
  }

  // 4. Box Plot & Quartile Spread
  const boxSelect = $("boxColSelect");
  if (boxSelect) {
    boxSelect.innerHTML = "";
    const boxKeys = Object.keys(viz.boxplots || {});
    boxKeys.forEach(k => {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = k;
      boxSelect.appendChild(opt);
    });
    if (boxKeys.length > 0) {
      renderBoxChart(boxKeys[0]);
    }
    boxSelect.onchange = () => renderBoxChart(boxSelect.value);
  }

  // 5. 2D Bivariate Scatter Plot
  const scatterSelect = $("scatterPairSelect");
  if (scatterSelect) {
    scatterSelect.innerHTML = "";
    const scatterKeys = Object.keys(viz.scatter || {});
    scatterKeys.forEach(k => {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = k;
      scatterSelect.appendChild(opt);
    });
    if (scatterKeys.length > 0) {
      renderScatterChart(scatterKeys[0]);
    }
    scatterSelect.onchange = () => renderScatterChart(scatterSelect.value);
  }

  // 6. Chronological / Sequential Trend Line
  if (viz.timeseries && viz.timeseries.labels && viz.timeseries.labels.length > 0) {
    renderTrendChart(viz.timeseries);
  }
}

function renderDistChart(col) {
  if (!currentVisualizations || !currentVisualizations.distributions[col]) return;
  const data = currentVisualizations.distributions[col];
  if (distChart) distChart.destroy();

  const ctx = $("distCanvas").getContext("2d");
  distChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [{
        label: `Frequency (${col})`,
        data: data.counts,
        backgroundColor: "rgba(22, 131, 255, 0.7)",
        borderColor: "#1683ff",
        borderWidth: 1.5,
        borderRadius: 5
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cfe6ff" } }
      },
      scales: {
        x: { ticks: { color: "#8cb4dc" }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#8cb4dc" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

function renderCatChart(col) {
  if (!currentVisualizations || !currentVisualizations.categories[col]) return;
  const data = currentVisualizations.categories[col];
  if (catChart) catChart.destroy();

  const ctx = $("catCanvas").getContext("2d");
  catChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: "Record Count",
          data: data.counts,
          backgroundColor: "rgba(66, 165, 245, 0.7)",
          borderColor: "#42a5f5",
          borderWidth: 1.5,
          borderRadius: 6
        },
        ...(data.metric_values ? [{
          label: data.metric_name || "Average Metric",
          data: data.metric_values,
          backgroundColor: "rgba(92, 242, 164, 0.7)",
          borderColor: "#5cf2a4",
          borderWidth: 1.5,
          borderRadius: 6
        }] : [])
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cfe6ff" } }
      },
      scales: {
        x: { ticks: { color: "#8cb4dc" }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#8cb4dc" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

function renderCorrChart(corrs) {
  if (corrChart) corrChart.destroy();
  const ctx = $("corrCanvas").getContext("2d");
  corrChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: corrs.map(c => c.pair),
      datasets: [{
        label: "Pearson Correlation (r)",
        data: corrs.map(c => c.r),
        backgroundColor: corrs.map(c => c.r >= 0 ? "rgba(40, 167, 69, 0.7)" : "rgba(220, 53, 69, 0.7)"),
        borderColor: corrs.map(c => c.r >= 0 ? "#28a745" : "#dc3545"),
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cfe6ff" } }
      },
      scales: {
        x: { min: -1, max: 1, ticks: { color: "#8cb4dc" }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#8cb4dc" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

function renderBoxChart(col) {
  if (!currentVisualizations || !currentVisualizations.boxplots[col]) return;
  const b = currentVisualizations.boxplots[col];
  if (boxChart) boxChart.destroy();

  const ctx = $("boxCanvas").getContext("2d");
  boxChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Min (Whisker)", "Q1 (25th %)", "Median", "Mean", "Q3 (75th %)", "Max (Whisker)"],
      datasets: [{
        label: `Quartile Summary (${col})`,
        data: [b.min, b.q1, b.median, b.mean, b.q3, b.max],
        backgroundColor: [
          "rgba(148, 163, 184, 0.7)",
          "rgba(56, 189, 248, 0.7)",
          "rgba(16, 185, 129, 0.8)",
          "rgba(245, 158, 11, 0.8)",
          "rgba(99, 102, 241, 0.7)",
          "rgba(239, 68, 68, 0.7)"
        ],
        borderColor: [
          "#94a3b8",
          "#38bdf8",
          "#10b981",
          "#f59e0b",
          "#6366f1",
          "#ef4444"
        ],
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cfe6ff" } }
      },
      scales: {
        x: { ticks: { color: "#8cb4dc" }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#8cb4dc" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

function renderScatterChart(pairKey) {
  if (!currentVisualizations || !currentVisualizations.scatter[pairKey]) return;
  const s = currentVisualizations.scatter[pairKey];
  if (scatterChart) scatterChart.destroy();

  const ctx = $("scatterCanvas").getContext("2d");
  scatterChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [{
        label: pairKey,
        data: s.points,
        backgroundColor: "rgba(147, 51, 234, 0.75)",
        borderColor: "#a855f7",
        borderWidth: 1,
        pointRadius: 4.5,
        pointHoverRadius: 7
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cfe6ff" } }
      },
      scales: {
        x: {
          title: { display: true, text: s.x_label, color: "#8cb4dc" },
          ticks: { color: "#8cb4dc" },
          grid: { color: "rgba(255,255,255,0.05)" }
        },
        y: {
          title: { display: true, text: s.y_label, color: "#8cb4dc" },
          ticks: { color: "#8cb4dc" },
          grid: { color: "rgba(255,255,255,0.05)" }
        }
      }
    }
  });
}

function renderTrendChart(ts) {
  if (trendChart) trendChart.destroy();
  if ($("trendInfoBadge")) {
    $("trendInfoBadge").textContent = `${ts.val_col} across ${ts.date_col}`;
  }

  const ctx = $("trendCanvas").getContext("2d");
  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: ts.labels,
      datasets: [{
        label: `${ts.val_col} Trajectory`,
        data: ts.values,
        borderColor: "#38bdf8",
        backgroundColor: "rgba(56, 189, 248, 0.12)",
        borderWidth: 2,
        fill: true,
        tension: 0.35,
        pointRadius: 3,
        pointHoverRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cfe6ff" } }
      },
      scales: {
        x: { ticks: { color: "#8cb4dc", maxRotation: 40, minRotation: 20 }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#8cb4dc" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

function renderMLChart(chartData) {
  if (mlChart) mlChart.destroy();
  const ctx = $("mlCanvas").getContext("2d");
  mlChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: chartData.labels,
      datasets: [{
        label: chartData.metric || "Model Score",
        data: chartData.scores,
        backgroundColor: chartData.scores.map((_, i) => i === 0 ? "rgba(255, 193, 7, 0.85)" : "rgba(22, 131, 255, 0.65)"),
        borderColor: chartData.scores.map((_, i) => i === 0 ? "#ffc107" : "#2b9aff"),
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cfe6ff" } }
      },
      scales: {
        x: { ticks: { color: "#8cb4dc", maxRotation: 40, minRotation: 25 }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#8cb4dc" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

// ========================================================
// MACHINE LEARNING 18+ MODELS LEADERBOARD RENDERING
// ========================================================
function renderML(ml) {
  if (!ml || !ml.models || ml.models.length === 0) {
    $("mlOutput").textContent = ml?.reason || "Machine learning evaluation was skipped.";
    return;
  }

  $("mlTitle").textContent = `Machine Learning Leaderboard: ${escapeHtml(ml.task || "Predictive Modeling")} (${ml.model_count} Models Evaluated)`;
  $("bestModelBadge").textContent = `🏆 Top: ${ml.best_model} (${ml.best_score})`;

  $("mlOutput").innerHTML = `
    <div style="font-size:12px;color:#85b5e3;margin-bottom:12px;">
      Engine: <b>${escapeHtml(ml.engine)}</b> • 100% WDAC Safe • Automated Train/Test Validation
    </div>
    <div class="ml-grid">
      ${ml.models.map((m, idx) => `
        <div class="model-card ${idx === 0 ? 'top-rank' : ''}">
          <div class="model-card-header">
            <b>${escapeHtml(m.name)}</b>
            <span class="model-rank-badge">${idx === 0 ? '🏆 RANK #1' : `RANK #${idx + 1}`}</span>
          </div>
          <div class="model-desc">${escapeHtml(m.description || '')}</div>
          <div>
            ${Object.entries(m.metrics || {}).map(([k, v]) => `
              <span class="metric-badge"><b>${escapeHtml(k)}:</b> ${escapeHtml(v)}</span>
            `).join('')}
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

// ========================================================
// NATURAL LANGUAGE QUERY ASSISTANT
// ========================================================
$("askBtn").onclick = async () => {
  const q = $("question").value.trim();
  if (!q) return;

  $("answer").innerHTML = "<em>Analyzing your dataset...</em>";
  try {
    const r = await api("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    });

    let htmlOutput = `<div>${escapeHtml(r.answer || "")}</div>`;

    if (r.type === "table" && Array.isArray(r.data) && r.data.length > 0) {
      const headers = Object.keys(r.data[0]);
      htmlOutput += `<div class="table-wrap" style="margin-top:10px;"><table><thead><tr>${
        headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")
      }</tr></thead><tbody>${
        r.data.map(row => `<tr>${headers.map(h => `<td>${escapeHtml(row[h])}</td>`).join("")}</tr>`).join("")
      }</tbody></table></div>`;
    }

    if (r.engine) {
      htmlOutput += `<div style="margin-top:8px;font-size:11px;color:#70a2d2;">
        Engine: <b>${escapeHtml(r.engine)}</b> • Model: <b>${escapeHtml(r.model || 'active')}</b>
        ${r.total_discovered ? ` • <i>(${r.total_discovered} Gemini models dynamically detected via API)</i>` : ''}
      </div>`;
    }

    $("answer").innerHTML = htmlOutput;
  } catch (err) {
    $("answer").innerHTML = `<span style="color:#ff6b6b;">Error: ${escapeHtml(err.message)}</span>`;
  }
};
