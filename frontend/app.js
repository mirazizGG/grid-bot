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
const chartConfirm = document.getElementById("chart-confirm");
chartConfirm.addEventListener("change", () => { analyzeBtn.disabled = !selectedFile || !chartConfirm.checked; });

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
    const mt5 = s.mt5.status === "ready" ? `MT5: ${escapeHtml(s.mt5.symbol)} · ${s.mt5.quote_age_seconds} soniya oldin` : `MT5: ${escapeHtml(s.mt5.symbol)} narxi mavjud emas yoki eskirgan`;
    const gemini = s.gemini.status === "configured" ? "Gemini: kalit sozlangan" : "Gemini: kalit yo‘q";
    const qwen = s.qwen.status === "ready" ? "Qwen: model mavjud" : s.qwen.status === "model_missing" ? "Qwen: model topilmadi" : "Qwen: Ollama ulanmagan";
    el.innerHTML = `<span class="service-item ${s.mt5.status === "ready" ? "ok" : "bad"}">${mt5}</span><span class="service-item ${s.gemini.status === "configured" ? "ok" : "bad"}">${gemini}</span><span class="service-item ${s.qwen.status === "ready" ? "ok" : "bad"}">${qwen}</span>`;
  } catch (_) {
    el.textContent = "Xizmatlar holatini olib bo‘lmadi. Serverni tekshiring.";
  }
}
loadServices();
setInterval(loadServices, 60000);

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
  resetChartConfirmation();
  errorBox.innerHTML = "";
  resultCard.style.display = "none";
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile || !chartConfirm.checked) return;
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="spinner"></span> Gemini va Qwen tahlil qilmoqda...';
  errorBox.innerHTML = "";

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("symbol", currentSymbol() || "XAUUSD");
  formData.append("timeframe", timeframeSelect.value);
  // Fayl yaratilgan vaqt -- backend eski skrinshotni aniqlashi uchun (clipboard'dan qo'yilganda 0 bo'lishi mumkin)
  if (selectedFile.lastModified) formData.append("file_modified_ms", selectedFile.lastModified);

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

function fmtPrice(v, digits) {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return "—";
  return Number(v).toFixed(Number.isInteger(digits) && digits >= 0 && digits <= 10 ? digits : 5);
}

function modelComparisonHtml(d) {
  const gemini = d.vision_ai?.signal;
  const qwen = d.qwen_vision?.signal;
  const comparable = ["Buy", "Sell", "Hold"].includes(gemini) && ["Buy", "Sell", "Hold"].includes(qwen);
  const text = !comparable ? "Ikki model natijasini solishtirib bo‘lmadi" : gemini === qwen ? "Gemini va Qwen bir xil xulosa berdi" : "Gemini va Qwen xulosalari farq qiladi";
  return `<div class="comparison-box"><strong>${text}</strong><span>Gemini: ${escapeHtml(gemini || "—")} · Qwen: ${escapeHtml(qwen || "—")}</span></div>`;
}

function modelOutcomeHtml(plan) {
  if (!plan) return "Avvalgi tahlil: alohida baholash yo‘q";
  if (plan.outcome === "SKIP") return `Baholanmaydi: ${escapeHtml(plan.reason || "reja yo‘q")}`;
  const labels = { TP: "TP urildi", SL: "SL urildi", OPEN: "kuzatilmoqda", EXPIRED: "muddat tugadi", NO_DATA: "MT5 tarixi yetarli emas" };
  return `Gipotetik natija: ${labels[plan.outcome] || "hali tekshirilmagan"}`;
}

function renderResult(data, target = resultCard) {
  const d = data.detail;
  const custom = d.custom_model;
  const vision = d.vision_ai;
  const qwen = d.qwen_vision;
  const digits = d.market_context?.digits;

  target.style.display = "";
  target.innerHTML = `
    <div class="result-heading">
      <div><div class="eyebrow">YAKUNIY XULOSA</div><span class="badge ${d.final_signal}">${signalLabel(d.final_signal)}</span></div>
      <div class="result-meta">${escapeHtml(d.market_context?.symbol || currentSymbol())} · ${escapeHtml(d.market_context?.timeframe || timeframeSelect.value)}<br>${d.final_confidence == null ? "Ishonch baholanmagan" : `Model bahosi: ${(d.final_confidence * 100).toFixed(0)}%`}</div>
    </div>

    <div class="result-grid">
      <div class="stat-tile"><div class="label">Entry · kirish</div><div class="value">${fmtPrice(d.entry_price, digits)}</div></div>
      <div class="stat-tile"><div class="label">Take Profit · foyda</div><div class="value">${fmtPrice(d.tp_price, digits)}</div></div>
      <div class="stat-tile"><div class="label">Stop Loss · zarar</div><div class="value">${fmtPrice(d.sl_price, digits)}</div></div>
    </div>
    ${priceSourceTag(d)}

    ${entryStrategyHtml(d)}
    ${modelComparisonHtml(d)}

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
        ${!qwen ? '<p>Bu tahlilda Qwen ishlatilmagan.</p>' : qwen.error
          ? `<p class="qwen-error">Qwen ishlamadi: ${escapeHtml(qwen.error)}</p>`
          : `<span class="badge ${qwen.signal}">${signalLabel(qwen.signal)}</span>
             <p class="confidence-line">Model bahosi: ${(qwen.confidence * 100).toFixed(0)}%</p>
             ${qwen.trend ? `<p>${escapeHtml(qwen.trend)}</p>` : ""}
             ${qwen.reasoning ? `<p>${escapeHtml(qwen.reasoning)}</p>` : ""}`}
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
        <p class="outcome-hint">${modelOutcomeHtml(d.model_comparison?.gemini)}</p>
      </div>
    </div>

    <div class="note-box">${d.agreement ? "✓ " : "⚠ "}${escapeHtml(d.note)}</div>
    ${(d.warnings || []).map((w) => `<div class="warn-box">⚠ ${escapeHtml(w)}</div>`).join("")}
  `;
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
      ${d.entry_type !== "market" ? '<div class="outcome-hint">Ko‘rsatilgan narxlar tahlil paytidagi narxga asoslangan; kutish sharti bajarilganda qayta tahlil qiling.</div>' : ""}
    </div>`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

// ---------- Dashboard ----------

function outcomeCell(row) {
  const o = row.outcome;
  const hypo = row.outcome_kind === "hold-gipotetik";
  if (!o) return `<span class="outcome-muted">hali tekshirilmagan</span>`;
  if (o === "SKIP") return `<span class="outcome-muted">—</span>`;
  const labels = { TP: "✓ TP", SL: "✗ SL", OPEN: "ochiq", EXPIRED: "urilmadi", NO_DATA: "ma'lumot yo'q" };
  const cls = o === "TP" ? "outcome-tp" : o === "SL" ? "outcome-sl" : "outcome-muted";
  const hint = hypo ? `<div class="outcome-hint">agar ${escapeHtml(row.detail.vision_ai.signal)} berilganida</div>` : "";
  return `<span class="${cls}">${labels[o] || o}</span>${hint}${row.outcome_version !== 3 ? '<div class="outcome-hint">eski baholash</div>' : ""}`;
}

function evalTile(title, s) {
  if (!s || s.n === 0) {
    return `<div class="stat-tile"><div class="label">${title}</div><div class="value">—</div>
      <div class="outcome-hint">TP/SL yo‘q${s ? ` · ochiq ${s.open} · urilmadi ${s.expired} · ma’lumot yo‘q ${s.no_data} · eski ${s.legacy}` : ""}</div></div>`;
  }
  const sign = s.avg_r >= 0 ? "+" : "";
  return `<div class="stat-tile"><div class="label">${title}</div>
    <div class="value">${(s.win_rate * 100).toFixed(0)}%</div>
    <div class="outcome-hint">${s.n} ta yakunlangan · TP ${s.tp} / SL ${s.sl} · ${sign}${s.avg_r.toFixed(2)}R/savdo<br>Ochiq ${s.open} · urilmadi ${s.expired} · ma’lumot yo‘q ${s.no_data} · eski ${s.legacy}</div></div>`;
}

function renderEvaluation(ev) {
  document.getElementById("eval-tiles").innerHTML =
    evalTile("Berilgan signallar (win-rate)", ev.signal) +
    evalTile("Hold bo'lsa-da (gipotetik)", ev["hold-gipotetik"]);
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
    const res = await fetch(`${API_BASE}/evaluate`, { method: "POST" });
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

async function loadDashboard() {
  const statsTiles = document.getElementById("stats-tiles");
  const barChart = document.getElementById("bar-chart");
  const historyContainer = document.getElementById("history-container");

  statsTiles.innerHTML = `<div class="empty-state"><span class="spinner"></span> Yuklanmoqda...</div>`;
  barChart.innerHTML = "";
  historyContainer.innerHTML = "";

  let stats, history;
  try {
    [stats, history] = await Promise.all([
      fetch(`${API_BASE}/stats`).then((r) => { if (!r.ok) throw new Error("Statistikani olishda xato"); return r.json(); }),
      fetch(`${API_BASE}/history?limit=30`).then((r) => { if (!r.ok) throw new Error("Tarixni olishda xato"); return r.json(); }),
    ]);
  } catch (err) {
    statsTiles.innerHTML = `<div class="error-box">${escapeHtml(err.message)}. Serverni tekshirib, qayta urinib ko‘ring.</div>`;
    return;
  }

  statsTiles.innerHTML = `
    <div class="stat-tile"><div class="label">Jami tahlillar</div><div class="value">${stats.total}</div></div>
    <div class="stat-tile"><div class="label">Buy</div><div class="value">${stats.by_signal.Buy || 0}</div></div>
    <div class="stat-tile"><div class="label">Sell</div><div class="value">${stats.by_signal.Sell || 0}</div></div>
    <div class="stat-tile"><div class="label">Hold</div><div class="value">${stats.by_signal.Hold || 0}</div></div>
  `;

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
        <tr><th>Rasm</th><th>Vaqt</th><th>Symbol / TF</th><th>Signal</th><th>Qwen</th><th>Model bahosi</th><th>Natija</th><th></th></tr>
      </thead>
      <tbody>
        ${history
          .map(
            (row) => `
          <tr>
            <td><img class="thumb" src="${row.image_path}" alt=""></td>
            <td class="mono">${new Date(row.timestamp).toLocaleString("uz-UZ")}</td>
            <td class="mono">${escapeHtml(row.detail.market_context?.symbol || "—")}<br><span class="outcome-hint">${escapeHtml(row.detail.market_context?.timeframe || "")}</span></td>
            <td><span class="badge ${row.final_signal}">${signalLabel(row.final_signal)}</span></td>
            <td>${row.detail.qwen_vision?.signal ? escapeHtml(row.detail.qwen_vision.signal) : "—"}</td>
            <td>${row.final_confidence == null ? "—" : (row.final_confidence * 100).toFixed(0) + "%"}</td>
            <td>${outcomeCell(row)}</td>
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
