// ── Detect backend URL (same origin in production, localhost in dev) ──
const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://localhost:5000"
  : "";   // same origin in production

// ── Chart colour palette matching CSS vars ──
const COLORS = {
  load:     "#e6edf3",
  solar:    "#f7c948",
  wind:     "#3fb950",
  battery:  "#a371f7",
  grid:     "#f78166",
  ren:      "#58a6ff",
  spilled:  "#f78166",
  stored:   "#a371f7",
  used:     "#3fb950",
  level:    "#58a6ff",
  ai:       "#f7c948",
};

// ── Chart registry so we can destroy before redraw ──
const charts = {};

function mkChart(id, cfg) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), cfg);
}

// ── Chart.js global defaults ──
Chart.defaults.color         = "#8b949e";
Chart.defaults.borderColor   = "#30363d";
Chart.defaults.font.family   = "Inter, sans-serif";
Chart.defaults.font.size     = 12;

// ── Shared line options factory ──
function lineOpts(labels, datasets, yLabel = "kW") {
  return {
    type: "line",
    data: { labels, datasets: datasets.map(d => ({
      tension: 0.4,
      pointRadius: 2,
      borderWidth: 2,
      fill: false,
      ...d,
    })) },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { usePointStyle: true, padding: 14 } } },
      scales: {
        x: { ticks: { maxTicksLimit: 12 } },
        y: { title: { display: true, text: yLabel } },
      },
    },
  };
}

// ── Gather current control values ──
function getParams() {
  return {
    load_value:      +document.getElementById("load-slider").value,
    use_solar:       document.getElementById("solar-toggle").checked,
    use_wind:        document.getElementById("wind-toggle").checked,
    use_battery:     document.getElementById("battery-toggle").checked,
    use_grid:        document.getElementById("grid-toggle").checked,
    solar_intensity: +document.getElementById("solar-slider").value,
    wind_speed:      +document.getElementById("wind-slider").value,
  };
}

// ── Debounce helper ──
let _timer;
function debounce(fn, ms = 500) {
  clearTimeout(_timer);
  _timer = setTimeout(fn, ms);
}

// ── Render charts from API response ──
function renderCharts(d) {
  const labels = d.t.map(h => `${h}:00`);
  const labelsNN = d.t_nn.map(h => `${h}:00`);

  // 1) Main: Energy Source Distribution
  mkChart("chart-main", lineOpts(labels, [
    { label: "Load",    borderColor: COLORS.load,    data: d.P_load },
    { label: "Solar",   borderColor: COLORS.solar,   data: d.P_solar,   borderDash: [5,3] },
    { label: "Wind",    borderColor: COLORS.wind,    data: d.P_wind,    borderDash: [5,3] },
    { label: "Battery", borderColor: COLORS.battery, data: d.battery_used, borderDash: [5,3] },
    { label: "Grid",    borderColor: COLORS.grid,    data: d.P_from_grid, borderDash: [5,3] },
  ]));

  // 2) Load vs Renewable
  mkChart("chart-lvr", lineOpts(labels, [
    { label: "Load",      borderColor: COLORS.load, data: d.P_load },
    { label: "Renewable", borderColor: COLORS.ren,  data: d.P_ren },
  ]));

  // 3) Grid vs Wasted
  mkChart("chart-gvw", lineOpts(labels, [
    { label: "Grid Usage",    borderColor: COLORS.grid,    data: d.P_from_grid },
    { label: "Wasted Energy", borderColor: COLORS.spilled, data: d.P_spilled, borderDash: [5,3] },
  ]));

  // 4 & 5) Battery (shown / hidden based on toggle)
  const battSec = document.getElementById("battery-section");
  if (d.use_battery) {
    battSec.style.display = "";
    mkChart("chart-bat-behav", lineOpts(labels, [
      { label: "Stored", borderColor: COLORS.stored, data: d.battery_store },
      { label: "Used",   borderColor: COLORS.used,   data: d.battery_used },
    ]));
    mkChart("chart-bat-level", lineOpts(labels, [
      { label: "Battery Level", borderColor: COLORS.level, data: d.battery_levels },
    ]));
  } else {
    battSec.style.display = "none";
  }

  // 6) AI Forecast
  mkChart("chart-ai", lineOpts(
    labels,
    [
      { label: "Actual Load",    borderColor: COLORS.load, data: d.P_load },
      { label: "Predicted Load", borderColor: COLORS.ai,  data: [null, ...d.predicted_nn], borderDash: [5,3] },
    ]
  ));
}

// ── Update summary cards ──
function updateSummary(d) {
  document.getElementById("s-load").textContent    = d.load_value + " kW";
  document.getElementById("s-solar").textContent   = d.use_solar   ? "ON" : "OFF";
  document.getElementById("s-wind").textContent    = d.use_wind    ? "ON" : "OFF";
  document.getElementById("s-battery").textContent = d.use_battery ? "ON" : "OFF";
  document.getElementById("s-grid").textContent    = d.use_grid    ? "ON" : "OFF";

  const banner = document.getElementById("status-banner");
  if (d.total_unmet > 0) {
    banner.className = "banner banner-error";
    banner.textContent = `⚠️ POWER CUT! Unserved Energy = ${d.total_unmet.toFixed(2)} kW`;
  } else {
    banner.className = "banner banner-success";
    banner.textContent = "✅ All demand successfully met";
  }
}

// ── Core: call backend and refresh UI ──
async function runSimulation() {
  try {
    const res = await fetch(`${API_BASE}/api/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getParams()),
    });
    if (!res.ok) throw new Error("Server error " + res.status);
    const data = await res.json();
    renderCharts(data);
    updateSummary(data);
  } catch (err) {
    console.error("Simulation error:", err);
  }
}

// ── Fetch & display weather ──
async function loadWeather() {
  try {
    const res  = await fetch(`${API_BASE}/api/weather`);
    const data = await res.json();
    document.getElementById("temp-val").textContent = `${data.temp} °C`;
    document.getElementById("hum-val").textContent  = `${data.humidity}%`;
  } catch {
    document.getElementById("temp-val").textContent = "30 °C";
    document.getElementById("hum-val").textContent  = "60%";
  }
}

// ── Wire up all controls ──
function wireControls() {
  const loadSlider  = document.getElementById("load-slider");
  const solarSlider = document.getElementById("solar-slider");
  const windSlider  = document.getElementById("wind-slider");

  loadSlider.addEventListener("input", () => {
    document.getElementById("load-display").textContent = loadSlider.value;
    debounce(runSimulation);
  });
  solarSlider.addEventListener("input", () => {
    document.getElementById("solar-display").textContent = solarSlider.value;
    debounce(runSimulation);
  });
  windSlider.addEventListener("input", () => {
    document.getElementById("wind-display").textContent = windSlider.value;
    debounce(runSimulation);
  });

  ["solar-toggle","wind-toggle","battery-toggle","grid-toggle"].forEach(id => {
    document.getElementById(id).addEventListener("change", () => debounce(runSimulation));
  });
}

// ── Bootstrap ──
document.addEventListener("DOMContentLoaded", () => {
  wireControls();
  loadWeather();
  runSimulation();
});
