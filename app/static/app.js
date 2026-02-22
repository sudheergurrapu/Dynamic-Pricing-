const fields = [
  "base_price",
  "demand_index",
  "competitor_price",
  "inventory_level",
  "customer_segment",
  "month",
  "day_of_week",
];

const form = document.getElementById("pricing-form");
const refreshMs = parseInt(document.getElementById("refresh-ms").innerText, 10);

const chartCtx = document.getElementById("priceChart");
const priceChart = new Chart(chartCtx, {
  type: "bar",
  data: {
    labels: ["Static Price", "Optimized Price", "RL-Bounded Price"],
    datasets: [
      {
        label: "Price",
        data: [0, 0, 0],
        backgroundColor: ["#94a3b8", "#38bdf8", "#22c55e"],
      },
    ],
  },
  options: { responsive: true },
});

function payloadFromForm() {
  return {
    base_price: parseFloat(document.getElementById("base_price").value),
    demand_index: parseFloat(document.getElementById("demand_index").value),
    competitor_price: parseFloat(document.getElementById("competitor_price").value),
    inventory_level: parseInt(document.getElementById("inventory_level").value, 10),
    customer_segment: document.getElementById("customer_segment").value,
    month: parseInt(document.getElementById("month").value, 10),
    day_of_week: parseInt(document.getElementById("day_of_week").value, 10),
    historical_demand: [0.7, 0.74, 0.77, 0.83, parseFloat(document.getElementById("demand_index").value)],
  };
}

async function fetchPrediction() {
  const payload = payloadFromForm();
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();

  document.getElementById("optimized_price").innerText = `$${data.optimized_price}`;
  document.getElementById("bounded_price").innerText = `$${data.bounded_price}`;
  document.getElementById("revenue_prediction").innerText = `$${data.revenue_prediction}`;
  document.getElementById("confidence_score").innerText = `${(data.confidence_score * 100).toFixed(2)}%`;
  document.getElementById("uplift").innerText = `${data.expected_revenue_uplift_pct}%`;
  document.getElementById("conversion").innerText = `${data.conversion_rate_improvement_pct}%`;
  document.getElementById("model_name").innerText = data.selected_model;

  document.getElementById("kpi_revenue").innerText = `${data.kpis.revenue_growth}%`;
  document.getElementById("kpi_margin").innerText = `${data.kpis.profit_margin}%`;
  document.getElementById("kpi_conversion").innerText = `${data.kpis.conversion_rate}%`;
  document.getElementById("kpi_turnover").innerText = `${data.kpis.inventory_turnover}`;

  priceChart.data.datasets[0].data = [payload.base_price, data.optimized_price, data.bounded_price];
  priceChart.update();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await fetchPrediction();
});

fields.forEach((name) => {
  document.getElementById(name).addEventListener("change", () => {
    fetchPrediction().catch(console.error);
  });
});

setInterval(() => {
  const demandEl = document.getElementById("demand_index");
  const current = parseFloat(demandEl.value);
  const drift = (Math.random() - 0.5) * 0.08;
  demandEl.value = Math.max(0, Math.min(2, current + drift)).toFixed(2);
  fetchPrediction().catch(console.error);
}, refreshMs);

fetchPrediction().catch(console.error);
