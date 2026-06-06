// ============================================================
// CoughAI — dashboard.js
// Polls backend /device/history every 3 seconds
// ============================================================

const POLL_INTERVAL = 3000;
const HISTORY_URL   = "/device/history";

const dot         = document.getElementById("status-dot");
const statusText  = document.getElementById("status-text");
const statTotal   = document.getElementById("stat-total");
const statHigh    = document.getElementById("stat-high");
const statMed     = document.getElementById("stat-med");
const statLow     = document.getElementById("stat-low");
const latestCard  = document.getElementById("latest-card");
const latestClass = document.getElementById("latest-class");
const latestConf  = document.getElementById("latest-conf");
const latestRisk  = document.getElementById("latest-risk");
const latestTime  = document.getElementById("latest-time");
const latestDev   = document.getElementById("latest-device");
const latestProbs = document.getElementById("latest-probs");
const historyList = document.getElementById("history-list");

let lastTimestamp = null;

// ── helpers ──────────────────────────────────────────────
function fmtTime(iso) {
    if (!iso) return "—";
    try {
        const d = new Date(iso);
        const diff = (Date.now() - d.getTime()) / 1000;
        if (diff < 60)    return `${Math.floor(diff)}s ago`;
        if (diff < 3600)  return `${Math.floor(diff/60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff/3600)}h ago`;
        return d.toLocaleString();
    } catch { return iso; }
}

function setOnline(ok) {
    if (ok) {
        dot.classList.add("online");
        statusText.textContent = "Device online";
    } else {
        dot.classList.remove("online");
        statusText.textContent = "Disconnected";
    }
}

// ── render ───────────────────────────────────────────────
function renderLatest(item) {
    if (!item) {
        latestCard.style.display = "none";
        return;
    }
    latestCard.style.display = "block";

    const cls = (item.classification || "").toLowerCase();
    latestClass.className = `latest-class ${cls}`;
    latestClass.textContent = cls.toUpperCase() || "—";

    latestConf.textContent = `Confidence ${item.confidence?.toFixed(1) ?? "?"}%`;
    latestTime.textContent = fmtTime(item.timestamp);
    latestDev.textContent  = item.device_id || "—";

    const risk = (item.risk_level || "LOW").toUpperCase();
    latestRisk.className = `latest-risk-chip ${risk}`;
    latestRisk.textContent = `${risk} RISK`;

    // probability bars
    latestProbs.innerHTML = "";
    const probs = item.probabilities || [];
    probs.forEach(p => {
        const row = document.createElement("div");
        row.className = "prob-row";
        const name = document.createElement("span");
        name.className = "prob-name";
        name.textContent = p.label;
        const trk = document.createElement("div");
        trk.className = "prob-track";
        const fill = document.createElement("div");
        const c = (p.label || "").toLowerCase();
        fill.className = `prob-fill ${c}`;
        fill.style.width = "0%";
        trk.appendChild(fill);
        const pct = document.createElement("span");
        pct.className = "prob-pct";
        pct.textContent = `${(p.score * 100).toFixed(1)}%`;
        row.append(name, trk, pct);
        latestProbs.appendChild(row);
        requestAnimationFrame(() => requestAnimationFrame(() => {
            fill.style.width = `${p.score * 100}%`;
        }));
    });
}

function renderStats(items) {
    statTotal.textContent = items.length;
    let h = 0, m = 0, l = 0;
    for (const it of items) {
        const r = (it.risk_level || "").toUpperCase();
        if (r === "HIGH")        h++;
        else if (r === "MEDIUM") m++;
        else if (r === "LOW")    l++;
    }
    statHigh.textContent = h;
    statMed.textContent  = m;
    statLow.textContent  = l;
}

function renderHistory(items) {
    if (!items.length) {
        historyList.innerHTML = `<div class="history-empty">No screenings yet — waiting for edge device…</div>`;
        return;
    }

    const rows = [`
        <div class="history-row head">
            <div>Time</div>
            <div>Classification</div>
            <div class="col-device">Device</div>
            <div class="col-risk">Risk</div>
        </div>
    `];

    for (const it of items) {
        const cls = (it.classification || "").toLowerCase();
        const risk = (it.risk_level || "LOW").toUpperCase();
        rows.push(`
            <div class="history-row">
                <div>${fmtTime(it.timestamp)}</div>
                <div class="history-class">
                    <span class="dot ${cls}"></span>
                    ${cls.toUpperCase()} · ${it.confidence?.toFixed(1) ?? "?"}%
                </div>
                <div class="col-device" style="color:var(--text3);font-size:0.82rem">${it.device_id || "—"}</div>
                <div class="col-risk">
                    <span class="latest-risk-chip ${risk}" style="font-size:0.72rem">${risk}</span>
                </div>
            </div>
        `);
    }
    historyList.innerHTML = rows.join("");
}

// ── poll ─────────────────────────────────────────────────
async function poll() {
    try {
        const res = await fetch(HISTORY_URL, { cache: "no-store" });
        if (!res.ok) throw new Error(res.status);
        const data = await res.json();
        const items = data.items || [];

        setOnline(true);
        renderStats(items);
        renderHistory(items);

        const top = items[0];
        if (top && top.timestamp !== lastTimestamp) {
            lastTimestamp = top.timestamp;
            renderLatest(top);
        } else if (!items.length) {
            latestCard.style.display = "none";
        } else {
            // refresh time label only
            latestTime.textContent = fmtTime(top.timestamp);
        }
    } catch (err) {
        console.warn("poll failed:", err);
        setOnline(false);
    }
}

poll();
setInterval(poll, POLL_INTERVAL);
