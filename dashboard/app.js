const methods = [
  {
    name: "Full-search",
    ratio: 1,
    capacity: 7.996,
    active: 1.421,
    ber: 0.00402,
    reliableBits: 79960,
    color: "#2563eb",
  },
  {
    name: "Supervised DNN",
    ratio: 0.9258,
    capacity: 7.553,
    active: 1.4215,
    ber: 0.0157,
    reliableBits: 75532,
    color: "#07966f",
  },
  {
    name: "Unsupervised DNN",
    ratio: 0.8997,
    capacity: 7.406,
    active: 1.3045,
    ber: 0.01015,
    reliableBits: 74059,
    color: "#b7791f",
  },
  {
    name: "DQN",
    ratio: 0.8997,
    capacity: 7.216,
    active: 1.598,
    ber: 0.06401,
    reliableBits: 72163,
    color: "#d6455d",
  },
  {
    name: "Random",
    ratio: 0.3058,
    capacity: 2.731,
    active: 4.7945,
    ber: 0.3574,
    reliableBits: 27313,
    color: "#64748b",
  },
  {
    name: "All-max",
    ratio: 0.283,
    capacity: 2.638,
    active: 6,
    ber: 0.3817,
    reliableBits: 26379,
    color: "#94a3b8",
  },
];

const learnedMethods = methods.slice(0, 4);

const ratioPreview = [
  { sample: 1, full: 1, supervised: 1, unsupervised: 1, dqn: 0.9501 },
  { sample: 2, full: 1, supervised: 0.8566, unsupervised: 0.9049, dqn: 1 },
  { sample: 3, full: 1, supervised: 1, unsupervised: 1, dqn: 1 },
  { sample: 4, full: 1, supervised: 0.9459, unsupervised: 1, dqn: 1 },
  { sample: 5, full: 1, supervised: 0.9847, unsupervised: 0.9847, dqn: 0.9847 },
  { sample: 6, full: 1, supervised: 1, unsupervised: 1, dqn: 1 },
  { sample: 7, full: 1, supervised: 0.991, unsupervised: 1, dqn: 0.8858 },
  { sample: 8, full: 1, supervised: 0.8292, unsupervised: 0.8292, dqn: 0.6162 },
];

const samplePowers = [
  { name: "Full-search", ratio: 1, color: "#2563eb", powers: [1, 1, 0.5, 0, 0.75, 0] },
  { name: "Supervised DNN", ratio: 0.98, color: "#07966f", powers: [0.5, 1, 0.25, 0, 0.75, 0] },
  { name: "Unsupervised DNN", ratio: 0.954, color: "#b7791f", powers: [1, 1, 0, 0, 1, 0] },
  { name: "DQN", ratio: 0.746, color: "#d6455d", powers: [1, 0.5, 0.5, 0.25, 0, 0] },
];

function fmt(value, digits = 3) {
  return value.toFixed(digits);
}

function renderCards() {
  document.querySelector("#methodCards").innerHTML = learnedMethods
    .map(
      (method) => `
        <article class="card">
          <div class="method-name">
            <span class="dot" style="background:${method.color}"></span>
            ${method.name}
          </div>
          <div class="big">${fmt(method.ratio)}</div>
          <p class="meta">
            Avg capacity ${fmt(method.capacity)}<br />
            Active links ${fmt(method.active)}
          </p>
        </article>
      `,
    )
    .join("");
}

function renderBars(selector, rows, key, maxValue, formatter = (v) => fmt(v)) {
  document.querySelector(selector).innerHTML = rows
    .map(
      (method) => `
        <div class="bar-row">
          <strong>${method.name}</strong>
          <div class="track">
            <div class="fill" style="width:${Math.min((method[key] / maxValue) * 100, 100)}%; background:${method.color}"></div>
          </div>
          <span class="value">${formatter(method[key])}</span>
        </div>
      `,
    )
    .join("");
}

function renderSmallBars(selector, rows, key, maxValue, formatter = (v) => fmt(v)) {
  document.querySelector(selector).innerHTML = rows
    .map(
      (method) => `
        <div class="small-row">
          <strong>${method.name}</strong>
          <div class="track">
            <div class="fill" style="width:${Math.min((method[key] / maxValue) * 100, 100)}%; background:${method.color}"></div>
          </div>
          <span class="value">${formatter(method[key])}</span>
        </div>
      `,
    )
    .join("");
}

function renderTrace() {
  const svg = document.querySelector("#ratioTrace");
  const width = 940;
  const height = 360;
  const pad = { top: 22, right: 30, bottom: 46, left: 56 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const yMin = 0.25;
  const yMax = 1.02;
  const x = (i) => pad.left + (i / (ratioPreview.length - 1)) * chartW;
  const y = (v) => pad.top + ((yMax - v) / (yMax - yMin)) * chartH;
  const series = [
    { key: "full", name: "Full-search", color: "#2563eb" },
    { key: "supervised", name: "Supervised DNN", color: "#07966f" },
    { key: "unsupervised", name: "Unsupervised DNN", color: "#b7791f" },
    { key: "dqn", name: "DQN", color: "#d6455d" },
  ];

  const grid = [0.25, 0.5, 0.75, 1]
    .map(
      (v) => `
        <line x1="${pad.left}" y1="${y(v)}" x2="${width - pad.right}" y2="${y(v)}" stroke="#d8e0e7" />
        <text x="14" y="${y(v) + 5}" fill="#617080" font-size="13">${v.toFixed(2)}</text>
      `,
    )
    .join("");
  const paths = series
    .map((item) => {
      const d = ratioPreview
        .map((row, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(row[item.key])}`)
        .join(" ");
      const dots = ratioPreview
        .map((row, i) => `<circle cx="${x(i)}" cy="${y(row[item.key])}" r="4.5" fill="${item.color}" />`)
        .join("");
      return `<path d="${d}" fill="none" stroke="${item.color}" stroke-width="3.5" />${dots}`;
    })
    .join("");
  const labels = ratioPreview
    .map((row, i) => `<text x="${x(i)}" y="${height - 15}" text-anchor="middle" fill="#617080" font-size="13">S${row.sample}</text>`)
    .join("");

  svg.innerHTML = `${grid}${paths}${labels}`;
  document.querySelector("#traceLegend").innerHTML = series
    .map((item) => `<span><i style="background:${item.color}"></i>${item.name}</span>`)
    .join("");
}

function powerBg(value) {
  if (value <= 0) return "#edf2f7";
  const alpha = 0.2 + value * 0.44;
  return `rgba(37, 99, 235, ${alpha.toFixed(2)})`;
}

function renderPowers() {
  document.querySelector("#powerList").innerHTML = samplePowers
    .map(
      (method) => `
        <div class="power-method">
          <div class="power-title">
            <span>${method.name}</span>
            <span style="color:${method.color}">ratio ${fmt(method.ratio)}</span>
          </div>
          <div class="power-cells">
            ${method.powers
              .map((p, i) => `<span class="power-cell" style="background:${powerBg(p)}" title="Router ${i + 1}">${p === 0 ? "off" : p.toFixed(2)}</span>`)
              .join("")}
          </div>
        </div>
      `,
    )
    .join("");
}

renderCards();
renderBars("#ratioBars", methods, "ratio", 1);
renderSmallBars("#activeBars", methods, "active", 6, (v) => fmt(v, 2));
renderTrace();
renderPowers();
renderSmallBars("#berBars", methods, "ber", 0.4, (v) => v.toExponential(2));
renderSmallBars("#shannonBars", methods, "reliableBits", 80000, (v) => Math.round(v).toLocaleString());
