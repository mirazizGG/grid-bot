const API_BASE = "";

const tabControl = document.getElementById("tab-control");
const tabAnalyze = document.getElementById("tab-analyze");
const tabDashboard = document.getElementById("tab-dashboard");
const viewAnalyze = document.getElementById("view-analyze");
const viewDashboard = document.getElementById("view-dashboard");
const viewControl = document.getElementById("view-control");
let activeTab = "control";

tabControl.addEventListener("click", () => switchTab("control"));
tabAnalyze.addEventListener("click", () => switchTab("analyze"));
tabDashboard.addEventListener("click", () => switchTab("dashboard"));

function switchTab(name) {
  activeTab = name;
  const isAnalyze = name === "analyze";
  const isDashboard = name === "dashboard";
  tabControl.classList.toggle("active", name === "control");
  tabAnalyze.classList.toggle("active", isAnalyze);
  tabDashboard.classList.toggle("active", isDashboard);
  viewControl.style.display = name === "control" ? "" : "none";
  viewAnalyze.style.display = isAnalyze ? "" : "none";
  viewDashboard.style.display = isDashboard ? "" : "none";
  if (isDashboard) loadDashboard();
  if (name === "control") void loadControlOverview();
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
const chartConfirm = document.getElementById("chart-confirm");
let isAnalyzing = false;
let currentResult = null;
let resultGeneration = 0;
chartConfirm.addEventListener("change", updateAnalyzeButton);

function updateAnalyzeButton() {
  const settings = controlSnapshot?.settings;
  const unavailable = settings && (settings.analysis_enabled === false || settings.gemini_enabled === false);
  analyzeBtn.disabled = isAnalyzing || !selectedFile || !chartConfirm.checked || !!unavailable;
  if (!isAnalyzing) analyzeBtn.textContent = unavailable ? "Tahlil to‘xtatilgan" : "Tahlil qilish";
}

symbolSelect.addEventListener("change", () => {
  symbolCustom.style.display = symbolSelect.value === "__custom__" ? "" : "none";
  resetChartConfirmation();
  loadServices();
});
symbolCustom.addEventListener("input", resetChartConfirmation);
symbolCustom.addEventListener("change", loadServices);
timeframeSelect.addEventListener("change", resetChartConfirmation);

function resetChartConfirmation() {
  chartConfirm.checked = false;
  analyzeBtn.disabled = true;
}

function currentSymbol() {
  return symbolSelect.value === "__custom__" ? symbolCustom.value.trim().toUpperCase() : symbolSelect.value;
}

async function loadServices() {
  const el = document.getElementById("service-status");
  try {
    const response = await fetch(`${API_BASE}/services?symbol=${encodeURIComponent(currentSymbol() || "XAUUSD")}`);
    if (!response.ok) throw new Error("status unavailable");
    const s = await response.json();
    renderServiceStrip(s);
  } catch (_) {
    el.textContent = "Xizmatlar holatini olib bo‘lmadi. Serverni tekshiring.";
  }
}

function renderServiceStrip(s) {
  const mt5 = s.mt5.status === "ready" ? `MT5: ${escapeHtml(s.mt5.symbol)} · ${s.mt5.quote_age_seconds} soniya oldin` : s.mt5.error
    ? `MT5: ${escapeHtml(s.mt5.error)}` : `MT5: ${escapeHtml(s.mt5.symbol)} narxi mavjud emas yoki eskirgan`;
  const gemini = s.gemini.status === "configured" ? "Gemini: kalit sozlangan" : "Gemini: kalit yo‘q";
  const qwen = s.qwen.status === "ready" ? "Qwen: model mavjud" : s.qwen.status === "model_missing" ? "Qwen: model topilmadi" : "Qwen: Ollama ulanmagan";
  const customWarning = s.custom_model?.warning
    ? `<span class="service-item">Custom model: ${escapeHtml(s.custom_model.warning)}</span>` : "";
  document.getElementById("service-status").innerHTML = `<span class="service-item ${s.mt5.status === "ready" ? "ok" : "bad"}">${mt5}</span><span class="service-item ${s.gemini.status === "configured" ? "ok" : "bad"}">${gemini}</span><span class="service-item ${s.qwen.status === "ready" ? "ok" : "bad"}">${qwen}</span>${customWarning}`;
}
loadServices();
setInterval(() => { if (!document.hidden && activeTab !== "control") void loadServices(); }, 60000);

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
  if (isAnalyzing) return;
  selectedFile = file;
  currentResult = null;
  resultGeneration += 1;
  dropzoneEmpty.style.display = "none";
  dropzoneFilename.textContent = file.name;
  preview.innerHTML = `<img src="${URL.createObjectURL(file)}" alt="preview">`;
  resetChartConfirmation();
  errorBox.innerHTML = "";
  resultCard.style.display = "none";
}

analyzeBtn.addEventListener("click", async () => {
  if (isAnalyzing || !selectedFile || !chartConfirm.checked) return;
  if (controlSnapshot?.settings && (controlSnapshot.settings.analysis_enabled === false || controlSnapshot.settings.gemini_enabled === false)) return;
  isAnalyzing = true;
  const generation = ++resultGeneration;
  currentResult = null;
  [fileInput, chartConfirm, symbolSelect, symbolCustom, timeframeSelect].forEach((el) => { el.disabled = true; });
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="spinner"></span> Gemini tahlil qilmoqda...';
  errorBox.innerHTML = "";

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("symbol", currentSymbol() || "XAUUSD");
  formData.append("timeframe", timeframeSelect.value);
  // Faylning o'zgartirilgan vaqti; bu screenshot olingan vaqtni tasdiqlamaydi.
  if (selectedFile.lastModified) formData.append("file_modified_ms", selectedFile.lastModified);

  try {
    const res = await fetch(`${API_BASE}/analyze`, { method: "POST", headers: { "X-Grid-Bot-Request": "1" }, body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Noma'lum xato");
    renderResult(data);
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    void loadQwenResult(data, generation);
  } catch (err) {
    errorBox.innerHTML = `<div class="error-box">Xato: ${escapeHtml(err.message)}</div>`;
  } finally {
    isAnalyzing = false;
    [fileInput, chartConfirm, symbolSelect, symbolCustom, timeframeSelect].forEach((el) => { el.disabled = false; });
    updateAnalyzeButton();
  }
});

async function loadQwenResult(data, generation) {
  if (data.id == null || data.detail.qwen_vision?.status !== "not_requested") return;
  const isCurrent = () => generation === resultGeneration && currentResult?.id === data.id;
  if (!isCurrent()) return;
  renderResult({ ...data, detail: { ...data.detail,
    qwen_vision: { ...data.detail.qwen_vision, status: "running" } } });
  try {
    const response = await fetch(`${API_BASE}/history/${encodeURIComponent(data.id)}/qwen`, {
      method: "POST", headers: { "X-Grid-Bot-Request": "1" },
    });
    const updated = await response.json();
    if (!response.ok) throw new Error(updated.detail || "Qwen tahlilini olib bo‘lmadi");
    if (updated.id !== data.id) throw new Error("Qwen javobi boshqa tahlilga tegishli");
    if (isCurrent()) renderResult(updated);
  } catch (err) {
    if (isCurrent()) renderResult({ ...currentResult, detail: { ...currentResult.detail,
      qwen_vision: { ...currentResult.detail.qwen_vision, status: "error", error: err.message } } });
  }
}

function signalLabel(signal) {
  if (signal === "Buy") return "▲ Sotib olish (Buy)";
  if (signal === "Sell") return "▼ Sotish (Sell)";
  return "● Kutish (Hold)";
}

function fmtPrice(v, digits) {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return "—";
  return Number(v).toFixed(Number.isInteger(digits) && digits >= 0 && digits <= 10 ? digits : 5);
}

function modelComparisonHtml(d) {
  const gemini = d.vision_ai?.signal;
  const waiting = ["not_requested", "pending", "running"].includes(d.qwen_vision?.status);
  const qwen = !waiting && !d.qwen_vision?.error && d.qwen_vision?.status !== "error"
    ? d.qwen_vision?.signal : null;
  const comparable = ["Buy", "Sell", "Hold"].includes(gemini) && ["Buy", "Sell", "Hold"].includes(qwen);
  const text = !comparable ? waiting ? "Qwenning alohida tahlili kutilmoqda" : "Ikki model natijasini solishtirib bo‘lmadi" : gemini === qwen ? "Gemini va Qwen bir xil xulosa berdi" : "Gemini va Qwen xulosalari farq qiladi";
  return `<div class="comparison-box"><strong>${text}</strong><span>Gemini: ${escapeHtml(gemini || "—")} · Qwen: ${escapeHtml(qwen || qwenStatusText(d.qwen_vision))}</span></div>`;
}

function qwenStatusText(qwen) {
  if (!qwen) return "Bu tahlilda Qwen ishlatilmagan.";
  if (qwen.status === "disabled") return "Qwen sozlamalarda o‘chirilgan.";
  if (qwen.error || qwen.status === "error") return "Qwen ishlamadi.";
  if (qwen.status === "not_requested") return "Qwen tahlili hali boshlanmagan.";
  if (qwen.status === "pending") return "Qwen tahlili navbatda.";
  if (qwen.status === "running") return "Qwen alohida tahlil qilmoqda...";
  return "Qwen natijasi hali mavjud emas.";
}

function qwenResultHtml(qwen) {
  if (qwen?.status === "disabled") return `<p>${escapeHtml(qwenStatusText(qwen))}</p>`;
  if (qwen?.error) return `<p class="qwen-error">Qwen ishlamadi: ${escapeHtml(qwen.error)}</p>`;
  if (!qwen || ["not_requested", "pending", "running", "error"].includes(qwen.status) ||
      !["Buy", "Sell", "Hold"].includes(qwen.signal)) return `<p>${escapeHtml(qwenStatusText(qwen))}</p>`;
  const confidence = typeof qwen.confidence === "number" && Number.isFinite(qwen.confidence)
    ? `${(qwen.confidence * 100).toFixed(0)}%` : "—";
  return `<span class="badge ${qwen.signal}">${signalLabel(qwen.signal)}</span>
    <p class="confidence-line">Model bahosi: ${confidence}</p>
    ${qwen.trend ? `<p>${escapeHtml(qwen.trend)}</p>` : ""}
    ${qwen.reasoning ? `<p>${escapeHtml(qwen.reasoning)}</p>` : ""}`;
}

function modelOutcomeHtml(plan) {
  if (!plan) return "Avvalgi tahlil: alohida baholash yo‘q";
  if (plan.version !== 4) return "Eski usuldagi baholash: yangi statistikaga kiritilmagan";
  const timestamp = plan.timestamp ? new Date(plan.timestamp) : null;
  const started = timestamp && !Number.isNaN(timestamp.getTime())
    ? `<br>Reja boshlangan vaqt: ${escapeHtml(timestamp.toLocaleString("uz-UZ"))}` : "";
  if (plan.outcome === "SKIP") return `Baholanmaydi: ${escapeHtml(plan.reason || "reja yo‘q")}${started}`;
  const labels = { TP: "TP urildi", SL: "SL urildi", OPEN: "kuzatilmoqda", EXPIRED: "muddat tugadi", NO_DATA: "MT5 tarixi yetarli emas" };
  return `Gipotetik natija: ${labels[plan.outcome] || "hali tekshirilmagan"}${started}`;
}

function renderResult(data, target = resultCard) {
  const d = data.detail;
  const custom = d.custom_model;
  const vision = d.vision_ai;
  const qwen = d.qwen_vision;
  const digits = d.market_context?.digits;
  const historical = target !== resultCard;
  const keepDetailsOpen = !historical && currentResult?.id === data.id && target.querySelector(".details-panel")?.open;
  if (!historical) currentResult = data;

  target.style.display = "";
  target.innerHTML = `
    <div class="result-heading">
      <div><div class="eyebrow">YAKUNIY XULOSA</div><span class="badge ${d.final_signal}">${signalLabel(d.final_signal)}</span></div>
      <div class="result-meta">${escapeHtml(d.market_context?.symbol || (historical ? "Noma’lum vosita" : currentSymbol()))} · ${escapeHtml(d.market_context?.timeframe || (historical ? "vaqt oralig‘i noma’lum" : timeframeSelect.value))}</div>
    </div>

    ${historical ? '<div class="warn-box">Bu oldingi tahlil. Narx va tavsiyalar hozirgi bozor holatini aks ettirmaydi.</div>' : ""}
    <div class="note-box">${d.agreement ? "✓ " : "⚠ "}${escapeHtml(d.note)}</div>
    ${(d.warnings || []).map((w) => `<div class="warn-box">⚠ ${escapeHtml(w)}</div>`).join("")}
    ${d.final_signal === "Hold" ? "" : `<div class="result-grid">
      <div class="stat-tile"><div class="label">Kirish narxi</div><div class="value">${fmtPrice(d.entry_price, digits)}</div></div>
      <div class="stat-tile"><div class="label">Foyda chegarasi (TP)</div><div class="value">${fmtPrice(d.tp_price, digits)}</div></div>
      <div class="stat-tile"><div class="label">Zarar chegarasi (SL)</div><div class="value">${fmtPrice(d.sl_price, digits)}</div></div>
    </div>`}
    ${priceSourceTag(d, historical)}

    ${entryStrategyHtml(d, historical)}
    <p class="outcome-hint">Signal yordamchi xulosa; foyda kafolatlanmaydi.</p>
    <details class="details-panel">
      <summary>AI tafsilotlari</summary>
      <p class="outcome-hint">${d.final_confidence == null ? "Yakuniy baho mavjud emas." : `Yakuniy model bahosi: ${(d.final_confidence * 100).toFixed(0)}%.`} Model baholari muvaffaqiyat ehtimoli sifatida tasdiqlanmagan.</p>
    ${modelComparisonHtml(d)}
    <p class="outcome-hint">Har bir faraziy reja o‘z boshlanish vaqtidagi narxga asoslanadi; vaqtlar pastda ko‘rsatilgan. Model narxlari taxmin bo‘lib, yakuniy narxlardan farq qilishi mumkin.</p>

    <div class="source-cards">
      <div class="source-card custom-card">
        <h3><span class="source-dot custom"></span>Custom Model</h3>
        <span class="badge ${custom.signal}">${signalLabel(custom.signal)}</span>
        <p class="confidence-line">${d.market_context ? `Model bahosi: ${(custom.confidence * 100).toFixed(0)}%` : "MT5 ma’lumoti mavjud emas"}</p>
        ${ownLevelsGrid(custom, digits)}
        ${custom.trend ? `<p>${escapeHtml(custom.trend)}</p>` : ""}
        ${custom.reasoning ? `<p>${escapeHtml(custom.reasoning)}</p>` : ""}
      </div>
      <div class="source-card qwen-card">
        <h3><span class="source-dot qwen"></span>Qwen · mahalliy model</h3>
        ${qwenResultHtml(qwen)}
        <p class="outcome-hint">${qwen?.model ? escapeHtml(qwen.model) + " · " : ""}Taqqoslash uchun; yakuniy signalga ta’sir qilmaydi.</p>
        <p class="outcome-hint">${modelOutcomeHtml(d.model_comparison?.qwen)}</p>
      </div>
      <div class="source-card">
        <h3><span class="source-dot gemini"></span>Gemini Vision</h3>
        <span class="badge ${vision.signal}">${signalLabel(vision.signal)}</span>
        <p class="confidence-line">Model bahosi: ${(vision.confidence * 100).toFixed(0)}%</p>
        ${ownLevelsGrid(vision, digits)}
        ${vision.trend ? `<p>${escapeHtml(vision.trend)}</p>` : ""}
        ${vision.reasoning ? `<p>${escapeHtml(vision.reasoning)}</p>` : ""}
        ${vision.model ? `<p class="outcome-hint">${escapeHtml(vision.model)}</p>` : ""}
        <p class="outcome-hint">${modelOutcomeHtml(d.model_comparison?.gemini)}</p>
      </div>
    </div>
    </details>
  `;
  if (keepDetailsOpen) target.querySelector(".details-panel").open = true;
}

function ownLevelsGrid(model, digits) {
  if (model.signal === "Hold" || (!model.tp_price && !model.sl_price)) return "";
  return `
    <div class="result-grid own-levels-grid">
      <div class="stat-tile"><div class="label">Entry</div><div class="value">${fmtPrice(model.entry_price, digits)}</div></div>
      <div class="stat-tile"><div class="label">Take Profit</div><div class="value">${fmtPrice(model.tp_price, digits)}</div></div>
      <div class="stat-tile"><div class="label">Stop Loss</div><div class="value">${fmtPrice(model.sl_price, digits)}</div></div>
    </div>`;
}

function priceSourceTag(d, historical = false) {
  if (d.final_signal === "Hold") return "";
  if (d.price_source === "mt5_live") {
    return `<div class="price-source-tag live">● ${historical ? "Tahlil vaqtidagi MT5 narxi" : "MT5 jonli narxi"} asosida hisoblangan</div>`;
  }
  return `<div class="price-source-tag">○ Rasmdan taxmin qilingan &mdash; MT5 ulanmagan yoki symbol topilmadi</div>`;
}

function entryTypeLabel(type) {
  if (type === "market") return "Hozir kirish";
  if (type === "wait_pullback") return "Qaytishini kutish";
  if (type === "wait_breakout") return "Daraja buzilishini kutish";
  return "";
}

function entryStrategyHtml(d, historical = false) {
  if (d.final_signal === "Hold" || !d.entry_note) return "";
  const label = historical && d.entry_type === "market" ? "O‘sha vaqtdagi kirish tavsiyasi" : entryTypeLabel(d.entry_type);
  return `
    <div class="entry-box ${d.final_signal}">
      ${label ? `<div class="entry-type">${label}</div>` : ""}
      <div class="entry-note">${escapeHtml(d.entry_note)}</div>
      ${d.entry_type !== "market" ? '<div class="outcome-hint">Ko‘rsatilgan narxlar tahlil paytidagi narxga asoslangan; kutish sharti bajarilganda qayta tahlil qiling.</div>' : ""}
    </div>`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

// ---------- Dashboard ----------

let dashboardLoading = false;
let historyOffset = 0;
let historyTotal = null;
let historyPageLength = 0;

function outcomeCell(row) {
  const o = row.outcome;
  const hypo = row.outcome_kind === "hold-gipotetik";
  if (!o) return `<span class="outcome-muted">hali tekshirilmagan</span>`;
  if (o === "SKIP") return `<span class="outcome-muted">baholanmaydi</span>`;
  const labels = { TP: "✓ TP urildi", SL: "✗ SL urildi", OPEN: "kuzatilmoqda", EXPIRED: "muddati tugadi", NO_DATA: "ma'lumot yetarli emas" };
  const cls = o === "TP" ? "outcome-tp" : o === "SL" ? "outcome-sl" : "outcome-muted";
  const hint = hypo ? `<div class="outcome-hint">faraziy natija</div>` : "";
  return `<span class="${cls}">${escapeHtml(labels[o] || o)}</span>${hint}${row.outcome_version !== 6 ? '<div class="outcome-hint">eski baholash</div>' : ""}`;
}

function evalTile(title, s) {
  if (!s || s.n === 0) {
    return `<div class="stat-tile"><div class="label">${title}</div><div class="value">—</div>
      <div class="outcome-hint">Hali TP/SL bilan yakunlangan natija yo‘q${s ? ` · ochiq ${s.open} · muddati tugagan ${s.expired} · ma’lumot yo‘q ${s.no_data} · eski ${s.legacy}` : ""}</div></div>`;
  }
  const sign = s.avg_r >= 0 ? "+" : "";
  return `<div class="stat-tile"><div class="label">${title}</div>
    <div class="value">${(s.win_rate * 100).toFixed(0)}%</div>
    <div class="outcome-hint">${s.n} ta TP/SL urilgan · TP ${s.tp} / SL ${s.sl} · ${sign}${s.avg_r.toFixed(2)}R/signal<br>Ochiq ${s.open} · muddati tugagan ${s.expired} · ma’lumot yo‘q ${s.no_data} · eski ${s.legacy}</div></div>`;
}

function renderEvaluation(ev) {
  document.getElementById("eval-tiles").innerHTML =
    evalTile("Berilgan signallar · TP ulushi", ev.signal) +
    evalTile("Hold uchun faraziy natija · TP ulushi", ev["hold-gipotetik"]);
}

function modelTile(title, s) {
  if (!s) return `<div class="stat-tile"><div class="label">${title}</div><div class="value">—</div><div class="outcome-hint">Ma’lumot yo‘q</div></div>`;
  const rate = s.n ? `${(s.win_rate * 100).toFixed(0)}%` : "—";
  return `<div class="stat-tile"><div class="label">${title} · TP ulushi</div><div class="value">${rate}</div>
    <div class="outcome-hint">${s.n} ta TP/SL · TP ${s.tp} / SL ${s.sl}<br>Tekshirilmagan ${s.pending} · ochiq ${s.open} · muddati tugagan ${s.expired}<br>Ma’lumot yo‘q ${s.no_data} · baholanmadi ${s.skip}</div></div>`;
}

document.getElementById("eval-btn").addEventListener("click", async () => {
  const btn = document.getElementById("eval-btn");
  const status = document.getElementById("eval-status");
  btn.disabled = true;
  status.textContent = "Tekshirilmoqda...";
  try {
    const res = await fetch(`${API_BASE}/evaluate`, { method: "POST", headers: { "X-Grid-Bot-Request": "1" } });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Noma'lum xato");
    status.textContent = `Tekshirildi: ${data.checked} ta signal, ${data.model_checked} ta model rejasi`;
    await loadDashboard();
  } catch (err) {
    status.textContent = `Xato: ${err.message}`;
  } finally {
    btn.disabled = false;
  }
});

function updateHistoryPaging() {
  document.getElementById("history-prev").disabled = dashboardLoading || historyOffset === 0;
  document.getElementById("history-next").disabled = dashboardLoading || (historyTotal === null ? historyPageLength < 30 : historyOffset + historyPageLength >= historyTotal);
  document.getElementById("history-filter").disabled = dashboardLoading;
  document.getElementById("history-page").textContent = historyPageLength
    ? `${historyOffset + 1}–${historyOffset + historyPageLength}${historyTotal !== null ? ` / ${historyTotal} ta tahlil` : ""}` : "Natija topilmadi";
}

async function loadDashboard() {
  if (dashboardLoading) return;
  dashboardLoading = true;
  updateHistoryPaging();
  try { await loadDashboardPage(); }
  finally { dashboardLoading = false; updateHistoryPaging(); }
}

async function loadDashboardPage() {
  const statsTiles = document.getElementById("stats-tiles");
  const barChart = document.getElementById("bar-chart");
  const historyContainer = document.getElementById("history-container");

  statsTiles.innerHTML = `<div class="empty-state"><span class="spinner"></span> Yuklanmoqda...</div>`;
  barChart.innerHTML = "";
  historyContainer.innerHTML = "";

  let stats, history;
  const symbol = document.getElementById("history-symbol").value.trim().toUpperCase();
  const signal = document.getElementById("history-signal").value;
  const query = `limit=30&offset=${historyOffset}${symbol ? `&symbol=${encodeURIComponent(symbol)}` : ""}${signal ? `&signal=${encodeURIComponent(signal)}` : ""}`;
  try {
    [stats, history] = await Promise.all([
      fetch(`${API_BASE}/stats`).then((r) => { if (!r.ok) throw new Error("Statistikani olishda xato"); return r.json(); }),
      fetch(`${API_BASE}/history?${query}`).then((r) => {
        if (!r.ok) throw new Error("Tarixni olishda xato");
        const total = r.headers?.get("X-Total-Count");
        historyTotal = total != null && Number.isFinite(Number(total)) ? Number(total) : null;
        return r.json();
      }),
    ]);
  } catch (err) {
    statsTiles.innerHTML = `<div class="error-box">${escapeHtml(err.message)}. Serverni tekshirib, qayta urinib ko‘ring.</div>`;
    return;
  }
  historyPageLength = history.length;

  statsTiles.innerHTML = `
    <div class="stat-tile"><div class="label">Jami tahlillar</div><div class="value">${stats.total}</div></div>
    <div class="stat-tile"><div class="label">Sotib olish · Buy</div><div class="value">${stats.by_signal.Buy || 0}</div></div>
    <div class="stat-tile"><div class="label">Sotish · Sell</div><div class="value">${stats.by_signal.Sell || 0}</div></div>
    <div class="stat-tile"><div class="label">Kutish · Hold</div><div class="value">${stats.by_signal.Hold || 0}</div></div>
  `;

  const excluded = document.getElementById("stats-excluded");
  excluded.hidden = !stats.excluded_stale_images;
  excluded.textContent = stats.excluded_stale_images
    ? `${stats.excluded_stale_images} ta eski rasmli tahlil tarixda bor, lekin yuqoridagi sonlarga qo‘shilmagan.` : "";

  renderEvaluation(stats.evaluation);
  document.getElementById("model-tiles").innerHTML = modelTile("Gemini", stats.model_comparison?.gemini) + modelTile("Qwen", stats.model_comparison?.qwen);

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
    <div class="table-scroll"><table class="history-table">
      <thead>
        <tr><th>Vaqt</th><th>Instrument / vaqt</th><th>Signal</th><th>Natija</th><th></th></tr>
      </thead>
      <tbody>
        ${history
          .map(
            (row) => `
          <tr>
            <td class="mono" data-label="Vaqt">${new Date(row.timestamp).toLocaleString("uz-UZ")}</td>
            <td class="mono" data-label="Vosita / vaqt oralig‘i">${escapeHtml(row.detail.market_context?.symbol || "—")}<br><span class="outcome-hint">${escapeHtml(row.detail.market_context?.timeframe || "")}</span></td>
            <td data-label="Signal"><span class="badge ${row.final_signal}">${signalLabel(row.final_signal)}</span></td>
            <td data-label="Natija">${outcomeCell(row)}</td>
            <td><button type="button" class="history-open" data-row-id="${row.id}">Ochish</button></td>
          </tr>`
          )
          .join("")}
      </tbody>
    </table></div>
  `;
  historyContainer.querySelectorAll(".history-open").forEach((button) => button.addEventListener("click", () => {
    const row = history.find((item) => String(item.id) === button.dataset.rowId);
    if (!row) return;
    document.getElementById("history-time").textContent = new Date(row.timestamp).toLocaleString("uz-UZ");
    document.getElementById("history-image").src = row.image_path;
    renderResult(row, document.getElementById("history-result"));
    document.getElementById("history-dialog").showModal();
  }));
}

document.getElementById("history-close").addEventListener("click", () => document.getElementById("history-dialog").close());
document.getElementById("history-filter").addEventListener("click", () => {
  if (dashboardLoading) return;
  historyOffset = 0;
  void loadDashboard();
});
document.getElementById("history-prev").addEventListener("click", () => {
  if (dashboardLoading || historyOffset === 0) return;
  historyOffset = Math.max(0, historyOffset - 30);
  void loadDashboard();
});
document.getElementById("history-next").addEventListener("click", () => {
  if (dashboardLoading || document.getElementById("history-next").disabled) return;
  historyOffset += 30;
  void loadDashboard();
});

// ---------- Control panel ----------

let controlSnapshot = null;
let controlOverviewLoading = false;
let controlLogsLoading = false;
let controlBackupsLoading = false;
let settingsSaving = false;
let settingsInitialized = false;
let settingsBaseline = {};
let modelConfirmation = null;
const dirtySettings = new Set();
const pendingActions = new Set();
const actionJobs = new Map();
const settingsFields = {
  analysis_enabled: ["setting-analysis-enabled", "boolean"],
  gemini_enabled: ["setting-gemini-enabled", "boolean"],
  qwen_enabled: ["setting-qwen-enabled", "boolean"],
  gemini_model: ["setting-gemini-model", "string"],
  qwen_model: ["setting-qwen-model", "string"],
  qwen_timeout: ["setting-qwen-timeout", "number"],
  min_custom_confidence: ["setting-min-custom-confidence", "number"],
  min_vision_override_confidence: ["setting-min-vision-override-confidence", "number"],
  custom_model_weight: ["setting-custom-model-weight", "number"],
  vision_ai_weight: ["setting-vision-ai-weight", "number"],
  default_symbol: ["setting-default-symbol", "string"],
  default_timeframe: ["setting-default-timeframe", "string"],
};
const serviceButtons = [
  ["control-mt5-check", "mt5", "check"],
  ["control-ollama-start", "ollama", "start"],
  ["control-ollama-stop", "ollama", "stop"],
  ["control-ollama-restart", "ollama", "restart"],
  ["control-backend-restart", "backend", "restart"],
];

async function controlRequest(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: { "X-Grid-Bot-Request": "1", "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch(`${API_BASE}/control/${path}`, options);
  let data;
  try { data = await response.json(); } catch (_) { throw new Error(`Server javobini o‘qib bo‘lmadi (${response.status || "xato"})`); }
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "So‘rovni bajarib bo‘lmadi";
    throw new Error(detail);
  }
  return data;
}

function controlError(error, secret = "") {
  const text = error?.message || "Serverga ulanib bo‘lmadi";
  return secret ? text.split(secret).join("[maxfiy]") : text;
}

function controlFeedback(message, error = false) {
  const el = document.getElementById("control-action-status");
  el.textContent = message;
  el.classList.toggle("error", error);
}

function formatBytes(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "—";
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = value / 1024, index = 0;
  while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
  return `${size.toFixed(size >= 100 ? 0 : 1)} ${units[index]}`;
}

function formatDuration(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "—";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} daqiqa`;
  const hours = Math.floor(minutes / 60);
  return hours < 24 ? `${hours} soat ${minutes % 60} daqiqa` : `${Math.floor(hours / 24)} kun ${hours % 24} soat`;
}

function formatTime(value) {
  const date = value ? new Date(value) : null;
  return date && !Number.isNaN(date.getTime()) ? date.toLocaleString("uz-UZ") : "—";
}

function percent(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(0)}%` : "—";
}

function metricTile(label, value, hint, utilization) {
  const bar = typeof utilization === "number" && Number.isFinite(utilization)
    ? `<div class="metric-track"><span style="width:${Math.min(100, Math.max(0, utilization))}%"></span></div>` : "";
  return `<div class="stat-tile"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div><span class="outcome-hint">${escapeHtml(hint)}</span>${bar}</div>`;
}

function setServiceState(id, label, ready) {
  const el = document.getElementById(id);
  el.textContent = label;
  el.classList.toggle("ready", ready);
  el.classList.toggle("warning", !ready);
}

function isActionBusy(key) {
  const job = actionJobs.get(key);
  return pendingActions.has(key) || ["queued", "running"].includes(job?.status);
}

function renderControlButtons() {
  const capabilities = controlSnapshot?.services?.capabilities || {};
  for (const [id, service, action] of serviceButtons) {
    const capability = capabilities[service]?.[action];
    const button = document.getElementById(id);
    button.disabled = !capability?.enabled || isActionBusy(`service:${service}:${action}`);
    button.title = capability?.reason || (!capability ? "Boshqaruv imkoniyati mavjud emas" : "");
  }
  for (const [id, key] of [["control-evaluate", "job:evaluate"], ["control-backup", "job:backup"],
    ["control-validate-model", "job:validate_model"]]) {
    document.getElementById(id).disabled = !controlSnapshot || isActionBusy(key);
  }
  const models = controlSnapshot?.models || {};
  const modelBusy = isActionBusy("model:activate_candidate") || isActionBusy("model:rollback");
  document.getElementById("control-activate-candidate").disabled = !models.candidate?.available || !!models.candidate?.active || modelBusy;
  document.getElementById("control-rollback").disabled = !models.rollback?.available || modelBusy;
  document.getElementById("control-model-confirm-yes").disabled = modelBusy;
  document.getElementById("control-pause").disabled = !controlSnapshot?.settings || settingsSaving;
}

function renderModels(models = {}) {
  const labels = { active: "Faol model", candidate: "Nomzod model", rollback: "Oldingi model" };
  document.getElementById("control-models").innerHTML = Object.entries(labels).map(([key, label]) => {
    const model = models[key];
    if (!model?.available) return `<div class="model-item"><div><strong>${label}</strong><p>Mavjud emas</p></div><span class="model-score">—</span></div>`;
    const accuracy = typeof model.validation_accuracy === "number" ? percent(model.validation_accuracy * 100) : "—";
    const baseline = typeof model.majority_baseline === "number" ? ` · sodda mezon ${percent(model.majority_baseline * 100)}` : "";
    const status = model.provenance_status === "legacy_unverified" ? "Baholash kelib chiqishi tasdiqlanmagan"
      : model.provenance_status === "verified_candidate" ? "O‘qitish va validatsiya qaydi mavjud" : "Baholash holati noma’lum";
    return `<div class="model-item"><div><strong>${label}${model.active ? " · faol" : ""}</strong><p>${escapeHtml(status)}${escapeHtml(baseline)}</p>${model.warning ? `<p class="model-warning">${escapeHtml(model.warning)}</p>` : ""}</div><div class="model-score">${accuracy}<p>Validatsiya</p></div></div>`;
  }).join("");
}

function jobLabel(kind) {
  const labels = { evaluate: "Natijalarni tekshirish", backup: "Zaxira nusxa", validate_model: "Modelni tekshirish",
    activate_candidate: "Nomzodni faollashtirish", rollback: "Modelni qaytarish" };
  const normalized = typeof kind === "string" && kind.startsWith("model:") ? kind.slice(6) : kind;
  if (labels[normalized]) return labels[normalized];
  if (typeof kind === "string" && kind.includes("ollama")) return "Ollama boshqaruvi";
  if (typeof kind === "string" && kind.includes("mt5")) return "MT5 ulanishini tekshirish";
  if (typeof kind === "string" && kind.includes("backend")) return "Backendni qayta boshlash";
  return kind || "Vazifa";
}

function jobResultText(job) {
  if (job.error) return String(job.error);
  const result = job.result;
  if (!result || typeof result !== "object") return "";
  if (typeof result.message === "string") return result.message;
  if (typeof result.checked === "number") return `${result.checked} ta signal${typeof result.model_checked === "number" ? `, ${result.model_checked} ta model rejasi` : ""} tekshirildi`;
  if (typeof result.name === "string") return result.name;
  if (typeof result.valid === "boolean") return result.valid ? "Tekshiruv muvaffaqiyatli" : "Tekshiruvdan o‘tmadi";
  return "";
}

function renderJobs(jobs = []) {
  for (const [key, previous] of actionJobs) {
    const latest = jobs.find((job) => job.id === previous.id);
    if (latest) actionJobs.set(key, latest);
  }
  const states = { queued: "Navbatda", running: "Bajarilmoqda", succeeded: "Bajarildi", failed: "Xato" };
  document.getElementById("control-jobs").innerHTML = jobs.length ? jobs.slice(0, 8).map((job) => {
    const status = Object.hasOwn(states, job.status) ? job.status : "queued";
    const result = jobResultText(job);
    return `<div class="job-item"><strong>${escapeHtml(jobLabel(job.kind))}</strong><span class="job-status ${status}">${status === "running" ? '<span class="spinner"></span>' : ""}${states[status]}</span><time>${escapeHtml(formatTime(job.finished_at || job.started_at || job.created_at))}</time><span></span>${result ? `<p class="job-result">${escapeHtml(result)}</p>` : ""}</div>`;
  }).join("") : '<p class="muted">Hali fon vazifalari yo‘q.</p>';
  renderControlButtons();
}

function readSetting(key) {
  const [id, type] = settingsFields[key];
  const el = document.getElementById(id);
  if (type === "boolean") return el.checked;
  if (type === "number") return Number(el.value);
  return key === "default_symbol" ? el.value.trim().toUpperCase() : el.value.trim();
}

function updateSettingsState() {
  const state = document.getElementById("control-settings-state");
  state.textContent = settingsSaving ? "Saqlanmoqda…" : dirtySettings.size ? `${dirtySettings.size} ta o‘zgarish saqlanmagan` : settingsInitialized ? "Barcha o‘zgarishlar saqlangan" : "Yuklanmoqda…";
  state.classList.toggle("dirty", dirtySettings.size > 0);
  document.getElementById("control-settings-save").disabled = settingsSaving || !dirtySettings.size;
  document.getElementById("control-settings-reset").disabled = settingsSaving || !dirtySettings.size;
}

function applySettings(settings, force = false) {
  if (settings) {
    document.getElementById("control-settings-warning").hidden = !settings.warning;
    document.getElementById("control-settings-warning").textContent = settings.warning || "";
  }
  if (!settings || (!force && (dirtySettings.size || settingsSaving))) return;
  settingsBaseline = { ...settings };
  for (const [key, [id, type]] of Object.entries(settingsFields)) {
    const el = document.getElementById(id);
    if (type === "boolean") el.checked = settings[key] === true;
    else el.value = settings[key] ?? "";
  }
  document.getElementById("control-api-key-status").textContent = settings.api_key_configured
    ? "Gemini kaliti saqlangan." : "Gemini kaliti hali sozlanmagan.";
  document.getElementById("control-settings-fields").disabled = settingsSaving;
  if (!settingsInitialized && settings.default_symbol && !selectedFile) {
    const symbol = settings.default_symbol;
    const known = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"].includes(symbol);
    symbolSelect.value = known ? symbol : "__custom__";
    symbolCustom.value = known ? "" : symbol;
    symbolCustom.style.display = known ? "none" : "";
    if (settings.default_timeframe) timeframeSelect.value = settings.default_timeframe;
  }
  settingsInitialized = true;
  updateSettingsState();
}

function renderControlOverview(data) {
  const previousBackups = controlSnapshot?.storage?.backup_count;
  controlSnapshot = data;
  const system = data.system || {}, ram = system.ram || {}, disk = system.disk || {};
  const gpus = Array.isArray(system.gpus) ? system.gpus : [];
  const gpuMetrics = gpus.length ? gpus.map((gpu, index) => metricTile(`GPU ${index}`,
    percent(gpu.utilization_percent), `${gpu.name} · ${formatBytes(gpu.memory_used_bytes)} / ${formatBytes(gpu.memory_total_bytes)}`,
    gpu.utilization_percent)).join("") : metricTile("GPU", "—", "GPU ma’lumoti mavjud emas");
  document.getElementById("control-metrics").innerHTML =
    metricTile("CPU", percent(system.cpu_percent), system.scope === "windows_host" ? "Windows kompyuteri" : "Joriy protsessor yuklanishi", system.cpu_percent) +
    metricTile("Operativ xotira", percent(ram.percent), `${formatBytes(ram.used_bytes)} / ${formatBytes(ram.total_bytes)}`, ram.percent) +
    gpuMetrics +
    metricTile("Disk", percent(disk.percent), `${formatBytes(disk.used_bytes)} / ${formatBytes(disk.total_bytes)}`, disk.percent) +
    (system.warning ? `<div class="warn-box">${escapeHtml(system.warning)}</div>` : "");
  const services = data.services || {};
  if (services.mt5 && services.gemini && services.qwen) {
    renderServiceStrip(services);
    setServiceState("control-mt5-state", services.mt5.status === "ready" ? "Ulangan" : services.mt5.error ? "Ulanish xatosi" : "Narx mavjud emas", services.mt5.status === "ready");
    setServiceState("control-gemini-state", services.gemini.status === "configured" ? "Kalit sozlangan" : "Kalit kerak", services.gemini.status === "configured");
    setServiceState("control-qwen-state", services.qwen.status === "ready" ? "Model mavjud" : services.qwen.status === "model_missing" ? "Model topilmadi" : "Ulanmagan", services.qwen.status === "ready");
    document.getElementById("control-mt5-detail").textContent = `${services.mt5.symbol || "—"} · ${services.mt5.status === "ready" ? `${services.mt5.quote_age_seconds} soniya oldingi narx` : services.mt5.error || services.mt5.bridge?.last_error || "MT5 terminali va ulanishini tekshiring"}`;
    const clock = services.mt5.clock_profile;
    if (clock && Number.isInteger(clock.offset_minutes)) {
      const absolute = Math.abs(clock.offset_minutes);
      const offset = `${clock.offset_minutes < 0 ? "−" : "+"}${String(Math.floor(absolute / 60)).padStart(2, "0")}:${String(absolute % 60).padStart(2, "0")}`;
      document.getElementById("control-mt5-detail").textContent += ` · Server UTC${offset} · ${clock.history_mode === "unverified" ? "tarixiy vaqt tasdiqlanmagan" : "tarixiy vaqt sozlangan"}`;
    }
    document.getElementById("control-gemini-detail").textContent = services.gemini.model || "Model ma’lumoti mavjud emas";
    document.getElementById("control-qwen-detail").textContent = services.qwen.model || "Model ma’lumoti mavjud emas";
  }
  const capabilities = services.capabilities || {};
  for (const [id, capability] of [["control-mt5-reason", capabilities.mt5?.check],
    ["control-backend-reason", capabilities.backend?.restart]]) {
    document.getElementById(id).textContent = capability?.reason || "";
  }
  document.getElementById("control-ollama-reason").textContent = [...new Set(Object.values(capabilities.ollama || {}).filter((cap) => !cap.enabled && cap.reason).map((cap) => cap.reason))].join(" · ");
  const analysis = data.analysis || {};
  const paused = analysis.paused ?? data.settings?.analysis_enabled === false;
  document.getElementById("control-analysis-status").textContent = `${paused ? "Yangi tahlillar to‘xtatilgan" : "Tahlil yoqilgan"} · joriy ${analysis.in_flight ?? "—"} / Qwen ${analysis.qwen_in_flight ?? "—"}`;
  document.getElementById("control-pause").textContent = paused ? "Davom ettirish" : "Vaqtincha to‘xtatish";
  setServiceState("control-backend-state", "Server ishlayapti", true);
  const process = system.process || {};
  document.getElementById("control-process").innerHTML = `<span>Ishlash vaqti<strong>${escapeHtml(formatDuration(process.uptime_seconds))}</strong></span><span>Jarayon xotirasi<strong>${escapeHtml(formatBytes(process.rss_bytes))}</strong></span><span>Jarayon ID<strong>${escapeHtml(process.pid ?? "—")}</strong></span>`;
  const storage = data.storage || {};
  document.getElementById("control-storage").innerHTML = `<span>Baza<strong>${escapeHtml(formatBytes(storage.db_bytes))}</strong></span><span>Rasmlar<strong>${escapeHtml(formatBytes(storage.uploads_bytes))} · ${escapeHtml(storage.upload_count ?? "—")} ta</strong></span><span>Zaxiralar<strong>${escapeHtml(storage.backup_count ?? "—")} ta</strong></span>`;
  document.getElementById("control-updated").textContent = `Yangilandi: ${formatTime(data.generated_at)}`;
  renderModels(data.models);
  renderJobs(data.jobs);
  applySettings(data.settings);
  updateAnalyzeButton();
  if (previousBackups === undefined || previousBackups !== storage.backup_count) void loadControlBackups();
}

async function loadControlOverview() {
  if (controlOverviewLoading || document.hidden || activeTab !== "control") return;
  controlOverviewLoading = true;
  document.getElementById("control-refresh").disabled = true;
  try {
    const data = await controlRequest("overview");
    renderControlOverview(data);
    document.getElementById("control-error").innerHTML = "";
    if (document.getElementById("control-logs-panel").open) void loadControlLogs();
  } catch (error) {
    document.getElementById("control-error").innerHTML = `<div class="error-box">${escapeHtml(controlError(error))}</div>`;
    document.getElementById("control-updated").textContent = "Yangilash amalga oshmadi";
    setServiceState("control-backend-state", "Ulanish tekshirilsin", false);
  } finally {
    controlOverviewLoading = false;
    document.getElementById("control-refresh").disabled = false;
  }
}

async function startControlJob(key, path) {
  if (isActionBusy(key)) return;
  pendingActions.add(key);
  renderControlButtons();
  controlFeedback("So‘rov yuborilmoqda…");
  try {
    const job = await controlRequest(path, {});
    actionJobs.set(key, job);
    const jobs = [job, ...(controlSnapshot?.jobs || []).filter((item) => item.id !== job.id)];
    if (controlSnapshot) controlSnapshot.jobs = jobs;
    renderJobs(jobs);
    controlFeedback("Vazifa qabul qilindi. Holati avtomatik yangilanadi.");
    return true;
  } catch (error) {
    controlFeedback(controlError(error), true);
    return false;
  } finally {
    pendingActions.delete(key);
    renderControlButtons();
  }
}

async function saveControlSettings(event) {
  event.preventDefault();
  if (settingsSaving || !dirtySettings.size) return;
  const payload = {};
  for (const key of dirtySettings) if (settingsFields[key]) payload[key] = readSetting(key);
  const keyInput = document.getElementById("setting-gemini-api-key");
  const secret = keyInput.value.trim();
  if (dirtySettings.has("gemini_api_key") && secret) payload.gemini_api_key = secret;
  if (!Object.keys(payload).length) return;
  settingsSaving = true;
  document.getElementById("control-settings-fields").disabled = true;
  updateSettingsState();
  const feedback = document.getElementById("control-settings-feedback");
  feedback.textContent = "Saqlanmoqda…";
  feedback.classList.remove("error");
  try {
    const settings = await controlRequest("settings", payload);
    if (controlSnapshot) controlSnapshot.settings = settings;
    dirtySettings.clear();
    applySettings(settings, true);
    feedback.textContent = "Sozlamalar saqlandi.";
    updateAnalyzeButton();
    void loadControlOverview();
  } catch (error) {
    feedback.textContent = controlError(error, secret);
    feedback.classList.add("error");
  } finally {
    keyInput.value = "";
    dirtySettings.delete("gemini_api_key");
    settingsSaving = false;
    document.getElementById("control-settings-fields").disabled = false;
    updateSettingsState();
    renderControlButtons();
  }
}

async function loadControlBackups() {
  if (controlBackupsLoading) return;
  controlBackupsLoading = true;
  try {
    const data = await controlRequest("backups");
    document.getElementById("control-backups").innerHTML = data.backups?.length ? data.backups.slice(0, 5).map((backup) => {
      const url = typeof backup.download_url === "string" && backup.download_url.startsWith("/control/") && !backup.download_url.startsWith("//") ? backup.download_url : null;
      return `<div class="backup-item"><span>${escapeHtml(backup.name)}<br>${escapeHtml(formatTime(backup.created_at))} · ${escapeHtml(formatBytes(backup.bytes))}</span>${url ? `<a href="${escapeHtml(url)}" download>Yuklash ↓</a>` : ""}</div>`;
    }).join("") : '<p class="muted">Hali zaxira nusxalari yo‘q.</p>';
  } catch (error) {
    document.getElementById("control-backups").textContent = controlError(error);
  } finally { controlBackupsLoading = false; }
}

async function loadControlLogs() {
  if (controlLogsLoading || document.hidden) return;
  controlLogsLoading = true;
  document.getElementById("control-logs-refresh").disabled = true;
  try {
    const source = document.getElementById("control-log-source").value || "server";
    const data = await controlRequest(`logs?source=${encodeURIComponent(source)}&lines=100`);
    document.getElementById("control-logs").textContent = data.available ? data.lines.join("\n") || "Jurnal bo‘sh." : "Bu jurnal hali mavjud emas.";
    document.getElementById("control-logs-status").textContent = data.available ? `Oxirgi ${data.lines.length} qator` : "Mavjud emas";
  } catch (error) {
    document.getElementById("control-logs").textContent = controlError(error);
  } finally {
    controlLogsLoading = false;
    document.getElementById("control-logs-refresh").disabled = false;
  }
}

for (const [key, [id]] of Object.entries(settingsFields)) {
  const changed = () => {
    if (readSetting(key) === settingsBaseline[key]) dirtySettings.delete(key); else dirtySettings.add(key);
    document.getElementById("control-settings-feedback").textContent = "";
    updateSettingsState();
  };
  document.getElementById(id).addEventListener("input", changed);
  document.getElementById(id).addEventListener("change", changed);
}
document.getElementById("setting-gemini-api-key").addEventListener("input", () => {
  if (document.getElementById("setting-gemini-api-key").value.trim()) dirtySettings.add("gemini_api_key");
  else dirtySettings.delete("gemini_api_key");
  updateSettingsState();
});
document.getElementById("control-settings-form").addEventListener("submit", saveControlSettings);
document.getElementById("control-settings-reset").addEventListener("click", () => {
  if (settingsSaving) return;
  dirtySettings.clear();
  document.getElementById("setting-gemini-api-key").value = "";
  document.getElementById("control-settings-feedback").textContent = "";
  applySettings(controlSnapshot?.settings, true);
});
document.getElementById("control-open-settings").addEventListener("click", () => document.getElementById("control-settings-card").scrollIntoView({ behavior: "smooth" }));
document.getElementById("control-refresh").addEventListener("click", () => void loadControlOverview());
document.getElementById("control-go-analyze").addEventListener("click", () => switchTab("analyze"));
document.getElementById("control-logs-refresh").addEventListener("click", () => void loadControlLogs());
document.getElementById("control-log-source").addEventListener("change", () => void loadControlLogs());
document.getElementById("control-logs-panel").addEventListener("toggle", () => {
  if (document.getElementById("control-logs-panel").open) void loadControlLogs();
});
for (const [id, service, action] of serviceButtons) {
  document.getElementById(id).addEventListener("click", () => {
    if (!controlSnapshot?.services?.capabilities?.[service]?.[action]?.enabled) return;
    void startControlJob(`service:${service}:${action}`, `services/${service}/${action}`);
  });
}
for (const [id, kind] of [["control-evaluate", "evaluate"], ["control-backup", "backup"], ["control-validate-model", "validate_model"]]) {
  document.getElementById(id).addEventListener("click", () => void startControlJob(`job:${kind}`, `jobs/${kind}`));
}
document.getElementById("control-pause").addEventListener("click", async () => {
  if (settingsSaving || !controlSnapshot?.settings) return;
  settingsSaving = true;
  updateSettingsState();
  renderControlButtons();
  try {
    const settings = await controlRequest("settings", { analysis_enabled: !controlSnapshot.settings.analysis_enabled });
    controlSnapshot.settings = settings;
    applySettings(settings);
    updateAnalyzeButton();
    controlFeedback(settings.analysis_enabled ? "Yangi tahlillar yoqildi." : "Yangi tahlillar vaqtincha to‘xtatildi.");
    void loadControlOverview();
  } catch (error) { controlFeedback(controlError(error), true); }
  finally {
    settingsSaving = false;
    applySettings(controlSnapshot.settings);
    updateSettingsState();
    renderControlButtons();
  }
});
for (const [id, action] of [["control-activate-candidate", "activate_candidate"], ["control-rollback", "rollback"]]) {
  document.getElementById(id).addEventListener("click", () => {
    if (document.getElementById(id).disabled) return;
    modelConfirmation = action;
    const warning = action === "activate_candidate" ? controlSnapshot?.models?.candidate?.warning : controlSnapshot?.models?.rollback?.warning;
    document.getElementById("control-model-confirm-text").textContent = `${action === "activate_candidate" ? "Nomzod model faol modelni almashtiradi. Validatsiya natijasi amaliy foydani kafolatlamaydi." : "Faol model saqlangan oldingi versiyaga almashtiriladi."}${warning ? ` ${warning}` : ""} Amalni tasdiqlaysizmi?`;
    document.getElementById("control-model-confirm").hidden = false;
  });
}
document.getElementById("control-model-confirm-no").addEventListener("click", () => {
  modelConfirmation = null;
  document.getElementById("control-model-confirm").hidden = true;
});
document.getElementById("control-model-confirm-yes").addEventListener("click", async () => {
  if (!modelConfirmation) return;
  const action = modelConfirmation;
  if (await startControlJob(`model:${action}`, `models/${action}`)) {
    modelConfirmation = null;
    document.getElementById("control-model-confirm").hidden = true;
  }
});
if (document.addEventListener) document.addEventListener("visibilitychange", () => {
  if (!document.hidden && activeTab === "control") void loadControlOverview();
});
setInterval(() => { if (!document.hidden && activeTab === "control") void loadControlOverview(); }, 5000);
void loadControlOverview();
