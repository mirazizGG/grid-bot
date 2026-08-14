const API_BASE = "";

const tabAnalyze = document.getElementById("tab-analyze");
const tabDashboard = document.getElementById("tab-dashboard");
const viewAnalyze = document.getElementById("view-analyze");
const viewDashboard = document.getElementById("view-dashboard");

tabAnalyze.addEventListener("click", () => switchTab("analyze"));
tabDashboard.addEventListener("click", () => switchTab("dashboard"));

function switchTab(name) {
  const isAnalyze = name === "analyze";
  tabAnalyze.classList.toggle("active", isAnalyze);
  tabDashboard.classList.toggle("active", !isAnalyze);
  viewAnalyze.style.display = isAnalyze ? "" : "none";
  viewDashboard.style.display = isAnalyze ? "none" : "";
  if (!isAnalyze) loadDashboard();
}

// ---------- Upload / Analyze ----------

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const dropzoneEmpty = document.getElementById("dropzone-empty");
const dropzoneFilename = document.getElementById("dropzone-filename");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyze-btn");
const errorBox = document.getElementById("error-box");
const resultCard = document.getElementById("result-card");
const symbolSelect = document.getElementById("symbol-select");
const symbolCustom = document.getElementById("symbol-custom");
const timeframeSelect = document.getElementById("timeframe-select");

symbolSelect.addEventListener("change", () => {
  symbolCustom.style.display = symbolSelect.value === "__custom__" ? "" : "none";
});

function currentSymbol() {
  return symbolSelect.value === "__custom__" ? symbolCustom.value.trim().toUpperCase() : symbolSelect.value;
}

let selectedFile = null;

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  dropzoneEmpty.style.display = "none";
  dropzoneFilename.textContent = file.name;
  preview.innerHTML = `<img src="${URL.createObjectURL(file)}" alt="preview">`;
  analyzeBtn.disabled = false;
  errorBox.innerHTML = "";
  resultCard.style.display = "none";
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="spinner"></span> Tahlil qilinmoqda...';
  errorBox.innerHTML = "";

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("symbol", currentSymbol() || "XAUUSD");
  formData.append("timeframe", timeframeSelect.value);

  try {
    const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Noma'lum xato");
    renderResult(data);
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    errorBox.innerHTML = `<div class="error-box">Xato: ${escapeHtml(err.message)}</div>`;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Tahlil qilish";
  }
});

function signalLabel(signal) {
  if (signal === "Buy") return "▲ Buy";
  if (signal === "Sell") return "▼ Sell";
  return "● Hold";
}

function fmtPrice(v) {
  return v === null || v === undefined ? "—" : Number(v).toFixed(2);
}

function renderResult(data) {
  const d = data.detail;
  const custom = d.custom_model;
  const vision = d.vision_ai;

  resultCard.style.display = "";
  resultCard.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
      <span class="badge ${d.final_signal}">${signalLabel(d.final_signal)}</span>
      <span style="color:var(--text-muted); font-size:0.85rem;">
        Ishonch: ${(d.final_confidence * 100).toFixed(0)}% ${d.agreement ? "· ikkala model kelishdi" : "· kelishmovchilik bor"}
      </span>
    </div>

    <div class="result-grid">
      <div class="stat-tile"><div class="label">Entry</div><div class="value">${fmtPrice(d.entry_price)}</div></div>
      <div class="stat-tile"><div class="label">Take Profit</div><div class="value">${fmtPrice(d.tp_price)}</div></div>
      <div class="stat-tile"><div class="label">Stop Loss</div><div class="value">${fmtPrice(d.sl_price)}</div></div>
    </div>
    ${priceSourceTag(d)}

    ${entryStrategyHtml(d)}

    <div class="source-cards">
      <div class="source-card">
        <h3><span class="source-dot custom"></span>Custom Model</h3>
        <span class="badge ${custom.signal}">${signalLabel(custom.signal)}</span>
        <p class="confidence-line">Ishonch: ${(custom.confidence * 100).toFixed(0)}%</p>
        ${ownLevelsHtml(custom)}
      </div>
      <div class="source-card">
        <h3><span class="source-dot gemini"></span>Gemini Vision</h3>
        <span class="badge ${vision.signal}">${signalLabel(vision.signal)}</span>
        <p class="confidence-line">Ishonch: ${(vision.confidence * 100).toFixed(0)}%</p>
        ${ownLevelsHtml(vision)}
        ${vision.trend ? `<p>${escapeHtml(vision.trend)}</p>` : ""}
        ${vision.reasoning ? `<p>${escapeHtml(vision.reasoning)}</p>` : ""}
      </div>
    </div>

    <div class="note-box">${d.agreement ? "✓ " : "⚠ "}${escapeHtml(d.note)}</div>
  `;
}

function ownLevelsHtml(model) {
  if (model.signal === "Hold" || (!model.tp_price && !model.sl_price)) return "";
  return `
    <div class="own-levels">
      <span>Entry <b>${fmtPrice(model.entry_price)}</b></span>
      <span>TP <b>${fmtPrice(model.tp_price)}</b></span>
      <span>SL <b>${fmtPrice(model.sl_price)}</b></span>
    </div>`;
}

function priceSourceTag(d) {
  if (d.final_signal === "Hold") return "";
  if (d.price_source === "mt5_live") {
    return `<div class="price-source-tag live">● MT5 jonli narx asosida (ATR bilan hisoblangan)</div>`;
  }
  return `<div class="price-source-tag">○ Rasmdan taxmin qilingan &mdash; MT5 ulanmagan yoki symbol topilmadi</div>`;
}

function entryTypeLabel(type) {
  if (type === "market") return "Hozir kirish";
  if (type === "wait_pullback") return "Qaytishini kutish";
  if (type === "wait_breakout") return "Daraja buzilishini kutish";
  return "";
}

function entryStrategyHtml(d) {
  if (d.final_signal === "Hold" || !d.entry_note) return "";
  const label = entryTypeLabel(d.entry_type);
  return `
    <div class="entry-box ${d.final_signal}">
      ${label ? `<div class="entry-type">${label}</div>` : ""}
      <div class="entry-note">${escapeHtml(d.entry_note)}</div>
    </div>`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

// ---------- Dashboard ----------

async function loadDashboard() {
  const statsTiles = document.getElementById("stats-tiles");
  const barChart = document.getElementById("bar-chart");
  const historyContainer = document.getElementById("history-container");

  statsTiles.innerHTML = `<div class="empty-state"><span class="spinner"></span> Yuklanmoqda...</div>`;
  barChart.innerHTML = "";
  historyContainer.innerHTML = "";

  const [stats, history] = await Promise.all([
    fetch(`${API_BASE}/stats`).then((r) => r.json()),
    fetch(`${API_BASE}/history?limit=30`).then((r) => r.json()),
  ]);

  statsTiles.innerHTML = `
    <div class="stat-tile"><div class="label">Jami tahlillar</div><div class="value">${stats.total}</div></div>
    <div class="stat-tile"><div class="label">Kelishuv darajasi</div><div class="value">${(stats.agreement_rate * 100).toFixed(0)}%</div></div>
    <div class="stat-tile"><div class="label">Buy</div><div class="value">${stats.by_signal.Buy || 0}</div></div>
    <div class="stat-tile"><div class="label">Sell</div><div class="value">${stats.by_signal.Sell || 0}</div></div>
  `;

  const maxCount = Math.max(1, ...Object.values(stats.by_signal));
  barChart.innerHTML = ["Buy", "Sell", "Hold"]
    .map((sig) => {
      const count = stats.by_signal[sig] || 0;
      const pct = (count / maxCount) * 100;
      return `
        <div class="bar-row">
          <span>${sig}</span>
          <div class="bar-track"><div class="bar-fill ${sig}" style="width:${pct}%"></div></div>
          <span>${count}</span>
        </div>`;
    })
    .join("");

  if (history.length === 0) {
    historyContainer.innerHTML = `
      <div class="empty-state">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>
        <div>Hali tahlil qilingan chart yo'q</div>
        <div class="empty-hint">"Tahlil" bo'limiga o'tib birinchi chartni yuklang</div>
      </div>`;
    return;
  }

  historyContainer.innerHTML = `
    <table class="history-table">
      <thead>
        <tr><th>Rasm</th><th>Vaqt</th><th>Signal</th><th>Ishonch</th><th>Kelishuv</th></tr>
      </thead>
      <tbody>
        ${history
          .map(
            (row) => `
          <tr>
            <td><img class="thumb" src="${row.image_path}" alt=""></td>
            <td class="mono">${new Date(row.timestamp).toLocaleString("uz-UZ")}</td>
            <td><span class="badge ${row.final_signal}">${signalLabel(row.final_signal)}</span></td>
            <td>${(row.final_confidence * 100).toFixed(0)}%</td>
            <td>${row.agreement ? "<span class=\"agree-yes\">✓ mos</span>" : "<span class=\"agree-no\">— farqli</span>"}</td>
          </tr>`
          )
          .join("")}
      </tbody>
    </table>
  `;
}
