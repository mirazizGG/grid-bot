// Run with: node --test frontend/test_app.cjs
// Mock the DOM and API so request ordering is checked without browser/services.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function escape(value) {
  return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

class Element {
  constructor() {
    this.style = {};
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.events = {};
    this.classes = new Set();
    this.classList = {
      add: (name) => this.classes.add(name), remove: (name) => this.classes.delete(name),
      toggle: (name, enabled) => {
        if (enabled ?? !this.classes.has(name)) this.classes.add(name); else this.classes.delete(name);
      },
    };
    this.panel = { open: false };
  }
  set innerHTML(value) { this.html = value; this.panel = { open: false }; }
  get innerHTML() { return this.html || ""; }
  set textContent(value) { this.text = value; this.html = escape(value); }
  get textContent() { return this.text || ""; }
  addEventListener(name, callback) { this.events[name] = callback; }
  querySelector() { return this.panel; }
  querySelectorAll() { return []; }
  scrollIntoView() {}
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function response(data, ok = true) { return { ok, json: async () => data }; }

function record(id, qwen = { source: "qwen", model: "test", status: "not_requested" }) {
  return { id, timestamp: "2026-11-17T05:00:00Z", detail: {
    final_signal: "Buy", final_confidence: 0.8, agreement: true, note: `primary ${id}`,
    entry_price: 100, tp_price: 102, sl_price: 99, price_source: "mt5_live",
    market_context: { symbol: "XAUUSD", timeframe: "M15", digits: 2 },
    custom_model: { signal: "Buy", confidence: 0.8 },
    vision_ai: { signal: "Buy", confidence: 0.8, model: "gemini-fixture" }, qwen_vision: qwen,
    model_comparison: {
      gemini: { version: 3, outcome: null, timestamp: "2026-11-17T05:00:00Z" },
      qwen: { version: 3, outcome: "SKIP", reason: "not requested" },
    },
  } };
}

test("background Qwen preserves primary controls and the current analysis", async () => {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  element("symbol-select").value = "XAUUSD";
  element("timeframe-select").value = "M15";
  const qwenPending = new Map();
  const calls = [];
  let nextAnalysis = 1;
  let heldAnalysis = null;
  const context = vm.createContext({
    document: { getElementById: element, createElement: () => new Element() },
    URL: { createObjectURL: () => "blob:mock" }, FormData: class { append() {} },
    setInterval() {},
    fetch: async (url, options) => {
      calls.push({ url, options });
      if (url.startsWith("/services")) return response({
        mt5: { status: "ready", symbol: "XAUUSD", quote_age_seconds: 0 },
        gemini: { status: "configured" }, qwen: { status: "ready" },
        custom_model: { warning: "Old <model> warning" },
      });
      if (url === "/analyze") {
        const id = nextAnalysis++;
        if (heldAnalysis) {
          const held = heldAnalysis;
          heldAnalysis = null;
          return held.promise;
        }
        return response(record(id));
      }
      const match = url.match(/^\/history\/(\d+)\/qwen$/);
      if (match) {
        assert.equal(options.method, "POST");
        assert.equal(options.headers["X-Grid-Bot-Request"], "1");
        assert.equal(options.body, undefined);
        const pending = deferred();
        qwenPending.set(Number(match[1]), pending);
        return pending.promise;
      }
      throw new Error(`Unexpected fetch: ${url}`);
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "app.js"), "utf8"), context);
  const read = (expression) => vm.runInContext(expression, context);
  const flush = () => new Promise((resolve) => setImmediate(resolve));
  const upload = (name) => {
    context.mockFile = { name, lastModified: 1 };
    read("setFile(mockFile)");
    element("chart-confirm").checked = true;
  };
  const start = () => element("analyze-btn").events.click();
  await flush();
  assert.match(element("service-status").innerHTML, /Old &lt;model&gt; warning/);

  upload("first.png");
  await start();
  assert.equal(element("analyze-btn").disabled, false, "Qwen must not hold the primary button");
  assert.equal(element("file-input").disabled, false);
  assert.equal(read("currentResult.id"), 1);
  assert.match(element("result-card").innerHTML, /Qwen alohida tahlil/);
  assert.match(element("result-card").innerHTML, /gemini-fixture/);
  assert.doesNotMatch(element("result-card").innerHTML, /NaN/);

  upload("second.png");
  qwenPending.get(1).resolve(response(record(1, {
    signal: "Sell", confidence: 0.9, reasoning: "stale upload result",
  })));
  await flush();
  assert.equal(read("currentResult"), null);
  assert.equal(element("result-card").style.display, "none");
  assert.doesNotMatch(element("result-card").innerHTML, /stale upload result/);

  await start();
  const hold = deferred();
  heldAnalysis = hold;
  const third = start();
  qwenPending.get(2).resolve(response(record(2, {
    signal: "Sell", confidence: 0.9, reasoning: "stale analysis result",
  })));
  await flush();
  assert.equal(read("currentResult"), null);
  assert.doesNotMatch(element("result-card").innerHTML, /stale analysis result/);
  hold.resolve(response(record(3)));
  await third;
  assert.equal(read("currentResult.id"), 3);
  assert.equal(element("analyze-btn").disabled, false);

  element("result-card").panel.open = true;
  const completed = record(3, {
    source: "qwen", status: "completed", signal: "Sell", confidence: 0.7,
    reasoning: "fresh completion",
  });
  completed.detail.model_comparison.qwen = {
    version: 3, outcome: null, timestamp: "2026-11-17T05:02:00Z",
  };
  qwenPending.get(3).resolve(response(completed));
  await flush();
  assert.match(element("result-card").innerHTML, /fresh completion/);
  assert.equal(read("currentResult.detail.final_signal"), "Buy");
  assert.equal(element("result-card").panel.open, true);
  assert.match(element("result-card").innerHTML, /Reja boshlangan vaqt/);
  assert.match(read("modelOutcomeHtml(currentResult.detail.model_comparison.gemini)"),
    /Reja boshlangan vaqt/);

  await start();
  qwenPending.get(4).resolve(response({ detail: "<unsafe> unavailable" }, false));
  await flush();
  assert.equal(read("currentResult.detail.qwen_vision.status"), "error");
  assert.equal(read("currentResult.detail.final_signal"), "Buy");
  assert.match(element("result-card").innerHTML, /&lt;unsafe&gt; unavailable/);
  assert.equal(element("analyze-btn").disabled, false);

  await start();
  qwenPending.get(5).resolve(response(record(99, { signal: "Sell", confidence: 0.9 })));
  await flush();
  assert.equal(read("currentResult.id"), 5, "A mismatched response must not replace the current record");
  assert.equal(read("currentResult.detail.qwen_vision.status"), "error");

  const before = calls.length;
  context.historyRecord = record(7);
  read('renderResult(historyRecord, document.getElementById("history-result"))');
  assert.equal(calls.length, before, "Opening history must not start inference");
  assert.match(element("history-result").innerHTML, /Qwen tahlili hali boshlanmagan/);
  assert.doesNotMatch(element("history-result").innerHTML, /NaN/);
  assert.doesNotMatch(read('outcomeCell({outcome:"TP",outcome_version:5})'), /eski/);
  assert.match(read('modelOutcomeHtml({version:2,outcome:"TP"})'), /Eski/);

  for (const status of ["not_requested", "pending", "running", "error"]) {
    context.model = { status, signal: "Buy", confidence: 0.8 };
    assert.doesNotMatch(read("qwenResultHtml(model)"), /badge|NaN/);
    assert.doesNotMatch(read('modelComparisonHtml({vision_ai:{signal:"Buy"},qwen_vision:model})'),
      /bir xil xulosa/);
  }
});

function controlOverview() {
  return {
    generated_at: "2026-11-17T05:00:00Z",
    services: {
      mt5: { status: "ready", symbol: "XAUUSD", quote_age_seconds: 2 },
      gemini: { status: "configured", model: "gemini-test" },
      qwen: { status: "ready", model: "qwen-test" },
      capabilities: {
        mt5: { check: { enabled: true } },
        ollama: { start: { enabled: true }, stop: { enabled: true }, restart: { enabled: true } },
        backend: { restart: { enabled: false, reason: "Supervisor mavjud emas" } },
      },
    },
    system: { cpu_percent: 42, ram: { used_bytes: 1024 ** 3, total_bytes: 2 * 1024 ** 3, percent: 50 },
      disk: { used_bytes: 4 * 1024 ** 3, total_bytes: 8 * 1024 ** 3, percent: 50 },
      process: { pid: 123, uptime_seconds: 3600, rss_bytes: 1024 ** 2 }, gpus: [] },
    settings: { analysis_enabled: true, gemini_enabled: true, qwen_enabled: true,
      gemini_model: "gemini-test", qwen_model: "qwen-test", qwen_timeout: 120,
      min_custom_confidence: 0.55, min_vision_override_confidence: 0.65,
      custom_model_weight: 0.25, vision_ai_weight: 0.75,
      default_symbol: "XAUUSD", default_timeframe: "M15", api_key_configured: true },
    models: {
      active: { available: true, provenance_status: "legacy_unverified", validation_accuracy: null,
        warning: "Faol model kelib chiqishi tasdiqlanmagan" },
      candidate: { available: true, active: false, provenance_status: "verified_candidate",
        validation_accuracy: 0.4, majority_baseline: 0.5, warning: "Nomzod sodda mezondan past" },
      rollback: { available: false },
    },
    jobs: [], storage: { db_bytes: 1024, uploads_bytes: 2048, upload_count: 3, backup_count: 1 },
    analysis: { paused: false, in_flight: 0, qwen_in_flight: 0 },
  };
}

function createControlHarness(initialSettings = {}) {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  element("symbol-select").value = "XAUUSD";
  element("timeframe-select").value = "M15";
  element("control-log-source").value = "server";
  const calls = [], intervals = [], documentEvents = {};
  const routes = new Map();
  const state = controlOverview();
  Object.assign(state.settings, initialSettings);
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const document = { hidden: false, getElementById: element, createElement: () => new Element(),
    addEventListener: (name, callback) => { documentEvents[name] = callback; } };
  const context = vm.createContext({
    document, URL: { createObjectURL: () => "blob:mock" }, FormData: class { append() {} },
    setInterval: (callback, delay) => { intervals.push({ callback, delay }); },
    fetch: async (url, options = {}) => {
      calls.push({ url, options });
      const route = routes.get(url);
      if (route) return route(options);
      if (url === "/control/overview") return response(clone(state));
      if (url.startsWith("/services")) return response(clone(state.services));
      if (url === "/control/backups") return response({ backups: [
        { name: "backup.db", bytes: 1024, created_at: state.generated_at, download_url: "/control/backups/backup.db" },
        { name: "invalid-link", bytes: 1, created_at: state.generated_at, download_url: "https://bad.example/download" },
      ] });
      if (url.startsWith("/control/logs?")) return response({ available: true, lines: ["<log> token=[REDACTED]"], source: "server" });
      if (url === "/control/settings") {
        const payload = JSON.parse(options.body);
        delete payload.gemini_api_key;
        Object.assign(state.settings, payload);
        return response(clone(state.settings));
      }
      throw new Error(`Unexpected fetch: ${url}`);
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "app.js"), "utf8"), context);
  return { element, calls, intervals, routes, state, document, documentEvents,
    read: (expression) => vm.runInContext(expression, context),
    flush: () => new Promise((resolve) => setImmediate(resolve)) };
}

test("control polling preserves dirty settings, secrets and visibility without overlap", async () => {
  const h = createControlHarness();
  await h.flush();
  assert.match(h.element("control-metrics").innerHTML, /42%/);
  assert.match(h.element("control-metrics").innerHTML, /GPU ma’lumoti mavjud emas/);
  assert.doesNotMatch(h.element("control-metrics").innerHTML, /NaN/);
  assert.equal(h.element("setting-gemini-api-key").value, "");
  assert.match(h.element("control-api-key-status").textContent, /saqlangan/);
  h.element("setting-gemini-model").value = "user-edited-model";
  h.element("setting-gemini-model").events.input();
  h.element("setting-gemini-api-key").value = "private-unsaved-key";
  h.element("setting-gemini-api-key").events.input();
  const held = deferred();
  h.routes.set("/control/overview", () => held.promise);
  const before = h.calls.filter((call) => call.url === "/control/overview").length;
  const first = h.read("loadControlOverview()");
  const duplicate = h.read("loadControlOverview()");
  assert.equal(h.calls.filter((call) => call.url === "/control/overview").length, before + 1);
  assert.equal(h.element("control-refresh").disabled, true);
  h.state.settings.gemini_model = "externally-changed-model";
  held.resolve(response(h.state));
  await Promise.all([first, duplicate]);
  assert.equal(h.element("setting-gemini-model").value, "user-edited-model");
  assert.equal(h.element("setting-gemini-api-key").value, "private-unsaved-key");
  assert.match(h.element("control-settings-state").textContent, /2 ta/);
  h.routes.delete("/control/overview");
  const poll = h.intervals.find((interval) => interval.delay === 5000).callback;
  const count = h.calls.length;
  h.document.hidden = true;
  poll(); await h.flush();
  assert.equal(h.calls.length, count, "A hidden page must not poll");
  h.document.hidden = false;
  h.read('switchTab("analyze")');
  poll(); await h.flush();
  assert.equal(h.calls.length, count, "Other tabs must not poll the control overview");
  h.read('switchTab("control")'); await h.flush();
  assert.ok(h.calls.length > count);
  assert.equal(h.element("setting-gemini-model").value, "user-edited-model");
  h.element("control-settings-reset").events.click();
  assert.equal(h.element("setting-gemini-model").value, "externally-changed-model");
  assert.equal(h.element("setting-gemini-api-key").value, "");
});

test("MT5 authorization errors render safely, take priority over bridge history and clear on recovery", async () => {
  const h = createControlHarness();
  await h.flush();
  const authError = 'MT5 hisobiga kirish tasdiqlanmadi (kod -6). <img src=x onerror="unsafe()">';
  h.state.services.mt5 = { status: "unavailable", symbol: "XAUUSD", quote_age_seconds: null,
    error: authError, bridge: { enabled: true, last_error: "Old unrelated bridge failure" } };
  await h.read("loadControlOverview()");
  assert.equal(h.element("control-mt5-state").textContent, "Ulanish xatosi");
  assert.equal(h.element("control-mt5-state").classes.has("warning"), true);
  assert.equal(h.element("control-mt5-detail").textContent, `XAUUSD · ${authError}`);
  assert.match(h.element("control-mt5-detail").innerHTML, /&lt;img/);
  assert.doesNotMatch(h.element("control-mt5-detail").innerHTML, /<img/);
  assert.doesNotMatch(h.element("control-mt5-detail").textContent, /Old unrelated bridge failure/);
  assert.match(h.element("service-status").innerHTML, /kod -6/);
  assert.match(h.element("service-status").innerHTML, /&lt;img/);
  assert.doesNotMatch(h.element("service-status").innerHTML, /<img/);

  h.state.services.mt5 = { status: "ready", symbol: "XAUUSD", quote_age_seconds: 1,
    error: null, bridge: { enabled: true, last_error: null } };
  await h.read("loadControlOverview()");
  assert.equal(h.element("control-mt5-state").textContent, "Ulangan");
  assert.equal(h.element("control-mt5-state").classes.has("warning"), false);
  assert.equal(h.element("control-mt5-state").classes.has("ready"), true);
  assert.equal(h.element("control-mt5-detail").textContent, "XAUUSD · 1 soniya oldingi narx");
  assert.doesNotMatch(h.element("service-status").innerHTML, /kod -6|unsafe|Ulanish xatosi/);
});

test("initial analysis symbol and timeframe come from saved settings", async () => {
  const h = createControlHarness({ default_symbol: "BTCUSD", default_timeframe: "H1" });
  await h.flush();
  assert.equal(h.element("symbol-select").value, "__custom__");
  assert.equal(h.element("symbol-custom").value, "BTCUSD");
  assert.equal(h.element("symbol-custom").style.display, "");
  assert.equal(h.element("timeframe-select").value, "H1");
  h.element("symbol-custom").value = "user-choice";
  await h.read("loadControlOverview()");
  assert.equal(h.element("symbol-custom").value, "user-choice", "Polling must preserve later chart choices");
});

test("settings save is partial, blocks duplicates and never echoes the write-only key", async () => {
  const h = createControlHarness();
  await h.flush();
  h.element("setting-qwen-timeout").value = "150";
  h.element("setting-qwen-timeout").events.input();
  const secret = "test-secret-must-not-appear";
  h.element("setting-gemini-api-key").value = secret;
  h.element("setting-gemini-api-key").events.input();
  const held = deferred();
  h.routes.set("/control/settings", () => held.promise);
  const first = h.element("control-settings-form").events.submit({ preventDefault() {} });
  const second = h.element("control-settings-form").events.submit({ preventDefault() {} });
  const writes = h.calls.filter((call) => call.url === "/control/settings");
  assert.equal(writes.length, 1);
  assert.equal(writes[0].options.headers["X-Grid-Bot-Request"], "1");
  assert.equal(writes[0].options.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(writes[0].options.body), { qwen_timeout: 150, gemini_api_key: secret });
  assert.equal(h.element("control-settings-fields").disabled, true);
  assert.equal(h.element("control-settings-save").disabled, true);
  await h.read("loadControlOverview()");
  assert.equal(h.element("setting-qwen-timeout").value, "150");
  held.resolve(response({ detail: `Invalid <key> ${secret}` }, false));
  await Promise.all([first, second]);
  assert.equal(h.element("setting-gemini-api-key").value, "");
  assert.doesNotMatch(h.element("control-settings-feedback").textContent, new RegExp(secret));
  assert.match(h.element("control-settings-feedback").textContent, /\[maxfiy\]/);
  assert.match(h.element("control-settings-feedback").innerHTML, /&lt;key&gt;/);
  assert.equal(h.element("control-settings-save").disabled, false);
  h.routes.delete("/control/settings");
  await h.element("control-settings-form").events.submit({ preventDefault() {} });
  const retry = h.calls.filter((call) => call.url === "/control/settings").at(-1);
  assert.deepEqual(JSON.parse(retry.options.body), { qwen_timeout: 150 });
  assert.match(h.element("control-settings-feedback").textContent, /saqlandi/);
  assert.equal(h.element("control-settings-save").disabled, true);
});

test("capability controls, jobs, inline model confirmation, logs and backups use real API state", async () => {
  const h = createControlHarness();
  await h.flush();
  assert.equal(h.element("control-backend-restart").disabled, true);
  assert.match(h.element("control-backend-restart").title, /Supervisor/);
  h.element("control-backend-restart").events.click();
  assert.ok(!h.calls.some((call) => call.url === "/control/services/backend/restart"));
  const held = deferred();
  h.routes.set("/control/services/ollama/start", () => held.promise);
  h.element("control-ollama-start").events.click();
  h.element("control-ollama-start").events.click();
  assert.equal(h.calls.filter((call) => call.url === "/control/services/ollama/start").length, 1);
  const write = h.calls.at(-1).options;
  assert.equal(write.headers["X-Grid-Bot-Request"], "1");
  assert.equal(write.headers["Content-Type"], "application/json");
  assert.equal(h.element("control-ollama-start").disabled, true);
  const job = { id: "service-job", kind: "ollama_start", status: "queued", created_at: h.state.generated_at };
  held.resolve(response(job)); await h.flush();
  assert.match(h.element("control-jobs").innerHTML, /Navbatda/);
  h.element("control-ollama-start").events.click();
  assert.equal(h.calls.filter((call) => call.url === "/control/services/ollama/start").length, 1);
  h.state.jobs = [{ ...job, status: "running" }];
  await h.read("loadControlOverview()");
  assert.match(h.element("control-jobs").innerHTML, /Bajarilmoqda/);
  h.state.jobs = [{ ...job, status: "failed", error: "<unsafe> service failure" }];
  await h.read("loadControlOverview()");
  assert.equal(h.element("control-ollama-start").disabled, false);
  assert.match(h.element("control-jobs").innerHTML, /&lt;unsafe&gt; service failure/);
  h.element("control-activate-candidate").events.click();
  assert.equal(h.element("control-model-confirm").hidden, false);
  assert.match(h.element("control-model-confirm-text").textContent, /sodda mezondan past/);
  assert.ok(!h.calls.some((call) => call.url === "/control/models/activate_candidate"));
  h.element("control-model-confirm-no").events.click();
  assert.equal(h.element("control-model-confirm").hidden, true);
  h.element("control-activate-candidate").events.click();
  h.routes.set("/control/models/activate_candidate", () => Promise.resolve(response({
    id: "model-job", kind: "activate_candidate", status: "queued", created_at: h.state.generated_at,
  })));
  await h.element("control-model-confirm-yes").events.click();
  assert.equal(h.element("control-model-confirm").hidden, true);
  assert.equal(h.element("control-activate-candidate").disabled, true);
  assert.equal(h.calls.filter((call) => call.url === "/control/models/activate_candidate").length, 1);
  await h.read("loadControlLogs()");
  assert.match(h.element("control-logs").textContent, /token=\[REDACTED\]/);
  assert.match(h.element("control-logs").innerHTML, /&lt;log&gt;/);
  assert.match(h.element("control-backups").innerHTML, /href="\/control\/backups\/backup.db"/);
  assert.doesNotMatch(h.element("control-backups").innerHTML, /bad\.example/);
});

test("history filters and pagination use total count without replacing the analysis result", async () => {
  const h = createControlHarness();
  await h.flush();
  const zero = { n: 0, open: 0, expired: 0, no_data: 0, legacy: 0 };
  h.routes.set("/stats", () => Promise.resolve(response({ total: 31,
    by_signal: { Buy: 31, Sell: 0, Hold: 0 }, evaluation: { signal: zero, "hold-gipotetik": zero } })));
  const page = (length, offset) => response(Array.from({ length }, (_, index) => {
    const row = record(offset + index + 1);
    row.final_signal = "Buy";
    return row;
  }));
  const firstUrl = "/history?limit=30&offset=0&symbol=EURUSD&signal=Buy";
  const secondUrl = "/history?limit=30&offset=30&symbol=EURUSD&signal=Buy";
  h.routes.set(firstUrl, () => Promise.resolve({ ...page(30, 0), headers: { get: () => "31" } }));
  h.routes.set(secondUrl, () => Promise.resolve({ ...page(1, 30), headers: { get: () => "31" } }));
  h.element("history-symbol").value = "eurusd";
  h.element("history-signal").value = "Buy";
  h.element("history-filter").events.click(); await h.flush();
  assert.ok(h.calls.some((call) => call.url === firstUrl));
  assert.equal(h.element("history-next").disabled, false);
  assert.equal(h.element("history-prev").disabled, true);
  assert.match(h.element("history-page").textContent, /1–30 \/ 31/);
  h.element("history-next").events.click(); await h.flush();
  assert.ok(h.calls.some((call) => call.url === secondUrl));
  assert.equal(h.element("history-next").disabled, true);
  assert.equal(h.element("history-prev").disabled, false);
  assert.match(h.element("history-page").textContent, /31–31 \/ 31/);
  assert.equal(h.read("currentResult"), null);
});

test("frontend element IDs are unique and all static script references exist", () => {
  const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
  const script = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  for (const match of script.matchAll(/getElementById\("([^"]+)"\)/g)) {
    assert.ok(ids.includes(match[1]), `Missing element: ${match[1]}`);
  }
});
