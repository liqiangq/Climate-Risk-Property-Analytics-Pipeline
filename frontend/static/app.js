const currency = new Intl.NumberFormat("en-NZ", {
  style: "currency",
  currency: "NZD",
  maximumFractionDigits: 0,
});

const number = new Intl.NumberFormat("en-NZ", {
  maximumFractionDigits: 4,
});

const rowsMetric = document.querySelector("#rowsMetric");
const dateMetric = document.querySelector("#dateMetric");
const averagePriceMetric = document.querySelector("#averagePriceMetric");
const medianPriceMetric = document.querySelector("#medianPriceMetric");
const frequencySelect = document.querySelector("#frequencySelect");
const yearSelect = document.querySelector("#yearSelect");
const modelSelect = document.querySelector("#modelSelect");
const refreshButton = document.querySelector("#refreshButton");
const priceIndexChart = document.querySelector("#priceIndexChart");
const floodZoneChart = document.querySelector("#floodZoneChart");
const modelMetrics = document.querySelector("#modelMetrics");
const coefficientRows = document.querySelector("#coefficientRows");
const recordsHead = document.querySelector("#recordsHead");
const recordsBody = document.querySelector("#recordsBody");

function cacheBust() {
  return `t=${Date.now()}`;
}

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || response.statusText);
  }
  return response.json();
}

function setStatus(selector, text) {
  document.querySelector(selector).textContent = text;
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (typeof value === "number") {
    return number.format(value);
  }
  return value;
}

async function loadSummary() {
  const summary = await api("/api/summary");
  rowsMetric.textContent = number.format(summary.rows);
  dateMetric.textContent = `${summary.date_min || "-"} to ${summary.date_max || "-"}`;
  averagePriceMetric.textContent = summary.average_total_price
    ? currency.format(summary.average_total_price)
    : "-";
  medianPriceMetric.textContent = summary.median_total_price
    ? currency.format(summary.median_total_price)
    : "-";

  const years = await api("/api/flood-zones");
  const current = yearSelect.value;
  yearSelect.innerHTML = '<option value="">All years</option>';
  years.forEach((row) => {
    if (row.Year) {
      const option = document.createElement("option");
      option.value = row.Year;
      option.textContent = row.Year;
      yearSelect.appendChild(option);
    }
  });
  yearSelect.value = current;
}

function loadCharts() {
  const frequency = frequencySelect.value;
  const year = yearSelect.value;
  const yearQuery = year ? `&year=${encodeURIComponent(year)}` : "";
  priceIndexChart.src = `/api/charts/price-index.png?frequency=${frequency}&${cacheBust()}`;
  floodZoneChart.src = `/api/charts/flood-zones.png?${cacheBust()}${yearQuery}`;
  setStatus("#priceIndexStatus", frequency);
  setStatus("#floodZoneStatus", year || "all years");
}

async function loadModel() {
  const selected = modelSelect.value;
  setStatus("#modelStatus", "loading");
  const path = selected === "did" ? "/api/model/did" : `/api/model/${selected}`;
  const result = await api(path);
  setStatus("#modelStatus", `${result.rows} rows`);

  modelMetrics.innerHTML = "";
  Object.entries(result.metrics).forEach(([key, value]) => {
    const row = document.createElement("div");
    row.innerHTML = `<span>${key}</span><strong>${formatValue(value)}</strong>`;
    modelMetrics.appendChild(row);
  });

  coefficientRows.innerHTML = "";
  Object.entries(result.coefficients)
    .sort(([left], [right]) => left.localeCompare(right))
    .forEach(([key, value]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${key}</td><td>${formatValue(value)}</td>`;
      coefficientRows.appendChild(tr);
    });
}

async function loadRecords() {
  setStatus("#recordsStatus", "loading");
  const records = await api("/api/records?limit=20");
  setStatus("#recordsStatus", `${records.length} rows`);
  if (!records.length) {
    recordsHead.innerHTML = "";
    recordsBody.innerHTML = "";
    return;
  }

  const columns = Object.keys(records[0]).slice(0, 10);
  recordsHead.innerHTML = `<tr>${columns.map((column) => `<th>${column}</th>`).join("")}</tr>`;
  recordsBody.innerHTML = records
    .map((record) => `<tr>${columns.map((column) => `<td>${formatValue(record[column])}</td>`).join("")}</tr>`)
    .join("");
}

async function loadDashboard() {
  try {
    await loadSummary();
    loadCharts();
    await Promise.all([loadModel(), loadRecords()]);
  } catch (error) {
    setStatus("#modelStatus", error.message);
  }
}

frequencySelect.addEventListener("change", loadCharts);
yearSelect.addEventListener("change", loadCharts);
modelSelect.addEventListener("change", loadModel);
refreshButton.addEventListener("click", loadDashboard);

loadDashboard();

