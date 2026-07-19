/* Dashboard page script: resolves color role tokens per theme, instantiates
 * Chart.js configs from the embedded JSON payload, and drives the zone-G
 * explainability picker. Fully offline; no external requests. */
(function () {
  "use strict";

  var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;

  var PALETTE = dark
    ? {
        "series-1": "#3987e5", "series-2": "#008300", "series-3": "#d55181",
        "series-4": "#c98500", "series-5": "#199e70", "series-6": "#d95926",
        "series-7": "#9085e9", "series-8": "#e66767",
        "status-good": "#0ca30c", "status-warning": "#fab219",
        "status-serious": "#ec835a", "status-critical": "#d03b3b",
        muted: "#898781", ink: "#c3c2b7", grid: "#2c2c2a"
      }
    : {
        "series-1": "#2a78d6", "series-2": "#008300", "series-3": "#e87ba4",
        "series-4": "#eda100", "series-5": "#1baf7a", "series-6": "#eb6834",
        "series-7": "#4a3aa7", "series-8": "#e34948",
        "status-good": "#0ca30c", "status-warning": "#fab219",
        "status-serious": "#ec835a", "status-critical": "#d03b3b",
        muted: "#898781", ink: "#52514e", grid: "#e1e0d9"
      };

  function resolveColors(node) {
    if (Array.isArray(node)) {
      return node.map(resolveColors);
    }
    if (node && typeof node === "object") {
      var out = {};
      for (var key in node) {
        out[key] = resolveColors(node[key]);
      }
      return out;
    }
    if (typeof node === "string" && PALETTE[node]) {
      return PALETTE[node];
    }
    return node;
  }

  var dataNode = document.getElementById("dashboard-data");
  var payload = {};
  try {
    payload = JSON.parse(dataNode.textContent);
  } catch (error) {
    return;
  }

  if (window.Chart) {
    Chart.defaults.color = PALETTE.ink;
    Chart.defaults.borderColor = PALETTE.grid;
    Chart.defaults.font.family =
      'system-ui, -apple-system, "Segoe UI", sans-serif';
    var charts = payload.charts || {};
    Object.keys(charts).forEach(function (panelId) {
      var canvas = document.getElementById("chart-" + panelId);
      if (!canvas) {
        return;
      }
      try {
        new Chart(canvas, resolveColors(charts[panelId]));
      } catch (error) {
        /* one broken chart must not kill the rest of the page */
      }
    });
  }

  /* ---- zone G: explainability picker ---- */
  var root = document.getElementById("explain-root");
  var forecast = payload.forecast;
  if (!root || !forecast) {
    return;
  }
  var inputs = payload.inputs || {};
  var productSelect = document.getElementById("explain-product");
  var timeSelect = document.getElementById("explain-time");
  var variableSelect = document.getElementById("explain-variable");
  var detail = document.getElementById("explain-detail");

  function points(product) {
    if (product === "minutely") {
      return forecast.minutely || [];
    }
    return (product === "daily" ? forecast.daily : forecast.hourly) || [];
  }

  function pointKey(product, point) {
    return product === "daily" ? point.date_local : point.valid_time;
  }

  function pointValues(product, point) {
    if (product !== "minutely") {
      return point.values || {};
    }
    var values = {};
    [
      "temp_c", "humidity_pct", "dew_point_c", "wind_speed_ms",
      "precip_intensity_mmh", "pop"
    ].forEach(function (name) {
      if (point[name] !== null && point[name] !== undefined) {
        values[name] = point[name];
      }
    });
    return values;
  }

  function option(select, value, label) {
    var node = document.createElement("option");
    node.value = value;
    node.textContent = label;
    select.appendChild(node);
  }

  function fillTimes() {
    timeSelect.textContent = "";
    points(productSelect.value).forEach(function (point, index) {
      var label = productSelect.value === "daily"
        ? point.date_local + " (D+" + point.lead_days + ")"
        : productSelect.value === "minutely"
          ? point.valid_time + " (+" + point.minutes_ahead + "m)"
          : point.valid_time + " (+" + point.lead_hours + "h)";
      option(timeSelect, String(index), label);
    });
  }

  function fillVariables() {
    var point = points(productSelect.value)[Number(timeSelect.value) || 0];
    variableSelect.textContent = "";
    if (!point) {
      return;
    }
    Object.keys(pointValues(productSelect.value, point)).sort().forEach(function (name) {
      option(variableSelect, name, name);
    });
  }

  function row(dt, dd) {
    return "<dt>" + dt + "</dt><dd>" + dd + "</dd>";
  }

  function escapeText(value) {
    var node = document.createElement("span");
    node.textContent = value === null || value === undefined ? "—" : String(value);
    return node.innerHTML;
  }

  function renderDetail() {
    var point = points(productSelect.value)[Number(timeSelect.value) || 0];
    var name = variableSelect.value;
    if (!point || !name) {
      detail.innerHTML = "<p class=\"empty-state\">no served point selected</p>";
      return;
    }
    var quantiles = (point.quantiles || {})[name];
    var quantileText = quantiles
      ? Object.keys(quantiles).sort().map(function (level) {
          return "q" + level + "=" + quantiles[level];
        }).join("  ")
      : "point-only (no distributional method selected)";
    var productInputs = inputs[productSelect.value] || {};
    var pointInputs = productInputs[pointKey(productSelect.value, point)] || {};
    var providers = pointInputs[name] || {};
    var providerText = Object.keys(providers).sort().map(function (source) {
      var entry = providers[source];
      return source + "=" + entry.value +
        (entry.age_hours === null || entry.age_hours === undefined
          ? ""
          : " (fetched " + entry.age_hours + "h before issue)");
    }).join("; ") || "no provider inputs recorded for this variable";
    detail.innerHTML =
      "<dl>" +
      row("value", escapeText(pointValues(productSelect.value, point)[name])) +
      row("method", escapeText((point.methods || {})[name])) +
      row("selection reason",
          escapeText(
            productSelect.value === "minutely"
              ? "minutely interpolation / anchoring path"
              : (point.selection_reasons || {})[name])) +
      row("quantiles", escapeText(quantileText)) +
      row("issued at", escapeText(forecast.issued_at)) +
      row("document status", escapeText(
        forecast.status + (forecast.status_reason ? " — " + forecast.status_reason : ""))) +
      row("release ids", escapeText((forecast.release_ids || []).join(", ") || "none (degraded)")) +
      row("dataset fingerprint", escapeText(forecast.dataset_fingerprint)) +
      row("anchor observation at", escapeText(forecast.observation_at)) +
      row("provider inputs", escapeText(providerText)) +
      "</dl>";
  }

  option(productSelect, "hourly", "hourly");
  if ((forecast.minutely || []).length) {
    option(productSelect, "minutely", "minutely");
  }
  if ((forecast.daily || []).length) {
    option(productSelect, "daily", "daily");
  }
  productSelect.addEventListener("change", function () {
    fillTimes();
    fillVariables();
    renderDetail();
  });
  timeSelect.addEventListener("change", function () {
    fillVariables();
    renderDetail();
  });
  variableSelect.addEventListener("change", renderDetail);
  fillTimes();
  fillVariables();
  renderDetail();
})();
