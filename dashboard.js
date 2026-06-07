// ============================================================
// CoughAI — dashboard.js  (เว็บล้วน · เก็บประวัติเสียงไอที่อัด/อัปโหลดบนเว็บ)
// ดึงประวัติจาก /history ทุก 4 วินาที แล้วแสดงผลแบบเรียลไทม์
// ============================================================

const POLL_INTERVAL = 4000;
const HISTORY_URL   = "/history";

// ── label → ไทย + สี ──────────────────────────────────────
const TH = {
    covid:       { name: "โควิด-19",  cls: "covid" },
    healthy:     { name: "สุขภาพดี",  cls: "healthy" },
    symptomatic: { name: "มีอาการ",   cls: "symptomatic" },
};
const RISK_TH = { HIGH: "เสี่ยงสูง", MEDIUM: "เฝ้าระวัง", LOW: "ปกติ" };

function thClass(label) {
    const key = (label || "").toLowerCase();
    return TH[key] || { name: (label || "—"), cls: "" };
}

// ── elements ──────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const liveDot    = $("live-dot");
const liveText   = $("live-text");
const statTotal  = $("stat-total");
const statHigh   = $("stat-high");
const statMed    = $("stat-med");
const statLow    = $("stat-low");
const latestCard = $("latest-card");
const latestClass= $("latest-class");
const latestConf = $("latest-conf");
const latestRisk = $("latest-risk");
const latestTime = $("latest-time");
const latestProbs= $("latest-probs");
const histList   = $("hist-list");
const refreshBtn = $("refresh-btn");

let lastTimestamp = null;

// ── เวลาแบบไทย ────────────────────────────────────────────
function fmtTime(iso) {
    if (!iso) return "—";
    try {
        const d = new Date(iso);
        const diff = (Date.now() - d.getTime()) / 1000;
        if (diff < 10)    return "เมื่อสักครู่";
        if (diff < 60)    return `${Math.floor(diff)} วินาทีที่แล้ว`;
        if (diff < 3600)  return `${Math.floor(diff / 60)} นาทีที่แล้ว`;
        if (diff < 86400) return `${Math.floor(diff / 3600)} ชั่วโมงที่แล้ว`;
        return d.toLocaleString("th-TH", {
            day: "numeric", month: "short",
            hour: "2-digit", minute: "2-digit",
        });
    } catch { return iso; }
}

function setLive(ok) {
    if (ok) {
        liveDot.classList.add("on");
        liveText.textContent = "เชื่อมต่อแล้ว";
    } else {
        liveDot.classList.remove("on");
        liveText.textContent = "เชื่อมต่อไม่ได้";
    }
}

// ── render: ผลล่าสุด ──────────────────────────────────────
function renderLatest(item) {
    if (!item) { latestCard.style.display = "none"; return; }
    latestCard.style.display = "block";

    const t = thClass(item.classification);
    latestClass.className = `latest-class ${t.cls}`;
    latestClass.textContent = t.name;

    const conf = (typeof item.confidence === "number") ? item.confidence.toFixed(1) : "?";
    latestConf.textContent = `ความมั่นใจ ${conf}%`;
    latestTime.textContent = fmtTime(item.timestamp);

    const risk = (item.risk_level || "LOW").toUpperCase();
    latestRisk.className = `risk-chip ${risk}`;
    latestRisk.textContent = RISK_TH[risk] || "ปกติ";

    // probability bars (เรียงมาก→น้อย)
    latestProbs.innerHTML = "";
    const probs = [...(item.probabilities || [])].sort((a, b) => b.score - a.score);
    probs.forEach((p) => {
        const t2 = thClass(p.label);
        const pct = (p.score * 100).toFixed(1);
        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
            <span class="prob-name">${t2.name}</span>
            <div class="prob-track"><div class="prob-fill ${t2.cls}"></div></div>
            <span class="prob-pct">${pct}%</span>`;
        latestProbs.appendChild(row);
        const fill = row.querySelector(".prob-fill");
        requestAnimationFrame(() => requestAnimationFrame(() => {
            fill.style.width = `${pct}%`;
        }));
    });
}

// ── render: สถิติ ─────────────────────────────────────────
function renderStats(items) {
    statTotal.textContent = items.length;
    let h = 0, m = 0, l = 0;
    for (const it of items) {
        const r = (it.risk_level || "").toUpperCase();
        if (r === "HIGH") h++;
        else if (r === "MEDIUM") m++;
        else l++;
    }
    statHigh.textContent = h;
    statMed.textContent  = m;
    statLow.textContent  = l;
}

// ── render: ประวัติ ───────────────────────────────────────
function renderHistory(items) {
    if (!items.length) {
        histList.innerHTML = `
            <div class="hist-empty">
                <p>ยังไม่มีประวัติ — ลองอัดหรืออัปโหลดเสียงไอที่หน้า App</p>
                <a href="/app">ไปหน้า App →</a>
            </div>`;
        return;
    }

    const rows = [`
        <div class="hist-row head">
            <div>เวลา</div>
            <div>ผลวิเคราะห์</div>
            <div class="hist-risk">ระดับ</div>
        </div>`];

    for (const it of items) {
        const t = thClass(it.classification);
        const risk = (it.risk_level || "LOW").toUpperCase();
        const conf = (typeof it.confidence === "number") ? it.confidence.toFixed(1) : "?";
        rows.push(`
            <div class="hist-row">
                <div class="hist-time">${fmtTime(it.timestamp)}</div>
                <div class="hist-class">
                    <span class="dot ${t.cls}"></span>${t.name} · ${conf}%
                </div>
                <div class="hist-risk"><span class="mini-chip ${risk}">${RISK_TH[risk] || "ปกติ"}</span></div>
            </div>`);
    }
    histList.innerHTML = rows.join("");
}

// ── poll ──────────────────────────────────────────────────
async function poll() {
    try {
        const res = await fetch(HISTORY_URL, { cache: "no-store" });
        if (!res.ok) throw new Error(res.status);
        const data = await res.json();
        const items = data.items || [];

        setLive(true);
        renderStats(items);
        renderHistory(items);

        const top = items[0];
        if (top && top.timestamp !== lastTimestamp) {
            lastTimestamp = top.timestamp;
            renderLatest(top);
        } else if (!items.length) {
            latestCard.style.display = "none";
        } else if (top) {
            latestTime.textContent = fmtTime(top.timestamp);
        }
    } catch (err) {
        console.warn("poll failed:", err);
        setLive(false);
    }
}

refreshBtn.addEventListener("click", () => {
    refreshBtn.classList.add("spin");
    poll().finally(() => setTimeout(() => refreshBtn.classList.remove("spin"), 600));
});

poll();
setInterval(poll, POLL_INTERVAL);
