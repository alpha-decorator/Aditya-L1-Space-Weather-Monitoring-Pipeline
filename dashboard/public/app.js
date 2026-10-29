"use strict";

const API = "/api/snapshot";
const REFRESH_MS = 30_000;
const P_MIN = 0.001;                // forecast axis floor (0.1 %)
const CLASSES = [["A", 1e-8], ["B", 1e-7], ["C", 1e-6], ["M", 1e-5], ["X", 1e-4]];
const MIN_FLUX = { C1: 1e-6, M1: 1e-5 };
const BAR_COLORS = { A: "#4a3352", B: "#6b4a78", C: "#a8558f", M: "#e2569b", X: "#f6ead0" };
const DEMO = new URLSearchParams(location.search).get("demo");   // ?demo=watch | major | flare

let snap = null;
let rangeH = 24;
let lastAlarm = null;
let soundOn = false;
const charts = {};

// ---------------------------------------------------------------- utilities
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pad = (n) => String(n).padStart(2, "0");
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const hhmm = (ms) => { const d = new Date(ms); return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`; };
const dayLabel = (ms) => { const d = new Date(ms); return `${MON[d.getUTCMonth()]} ${d.getUTCDate()}`; };
const dateTime = (ms) => `${dayLabel(ms)} ${hhmm(ms)}`;
const mins = (ms) => Math.round(ms / 60e3);

function goesClass(f) {
  if (f == null || !(f > 0)) return "—";
  if (f < 1e-8) return "<A1";
  let [letter, base] = CLASSES[0];
  for (const [l, b] of CLASSES) if (f >= b) [letter, base] = [l, b];
  return `${letter}${(Math.floor((f / base) * 10) / 10).toFixed(1)}`;
}
function pct(p) {
  if (p == null) return "—";
  if (p < 0.001) return "<0.1%";
  if (p < 0.01) return `${(p * 100).toFixed(1)}%`;
  return `${Math.round(p * 100)}%`;
}
const verLabel = (m) => (m.version === "base" ? "Original" : `v${m.version.slice(4, 6)}.${m.version.slice(6, 8)}`);

const ICON = {
  good: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="currentColor"/><path d="M4.8 8.2l2.1 2.1 4.3-4.6" stroke="#fff" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  warning: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5l7 12.5H1z" fill="currentColor" stroke="currentColor" stroke-linejoin="round"/><path d="M8 6v3.6" stroke="#0b0b0b" stroke-width="1.7" stroke-linecap="round"/><circle cx="8" cy="11.8" r="1" fill="#0b0b0b"/></svg>',
  serious: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5l7 12.5H1z" fill="currentColor" stroke="currentColor" stroke-linejoin="round"/><path d="M8 6v3.6" stroke="#fff" stroke-width="1.7" stroke-linecap="round"/><circle cx="8" cy="11.8" r="1" fill="#fff"/></svg>',
  critical: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="currentColor"/><path d="M8 4.2v4.6" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/><circle cx="8" cy="11.5" r="1.05" fill="#fff"/></svg>',
  neutral: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>',
};
const pill = (tone, label) => `<span class="pill ${tone}">${ICON[tone]}${esc(label)}</span>`;
const STATE_LABEL = { QUIET: "Quiet", FLARE: "Flare in progress", DECAY: "Flare fading" };

// ---------------------------------------------------------------- illustration
function rng(seed) { return () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646; }
const sparkle = (x, y, r, fill) =>
  `<path d="M${x},${y - r}Q${x + r * .16},${y - r * .16} ${x + r},${y}Q${x + r * .16},${y + r * .16} ${x},${y + r}Q${x - r * .16},${y + r * .16} ${x - r},${y}Q${x - r * .16},${y - r * .16} ${x},${y - r}Z" fill="${fill}"/>`;

function drawSky() {
  const W = 1600, H = 1000, R = rng(7);
  let s = "";
  for (let i = 0; i < 220; i++) {
    const c = R() < 0.82 ? "#f6ead0" : R() < 0.5 ? "#f26a2e" : "#f29bc4";
    s += `<circle cx="${(R() * W).toFixed(0)}" cy="${(R() * H).toFixed(0)}" r="${(R() * 1.3 + 0.3).toFixed(2)}" fill="${c}" opacity="${(R() * 0.6 + 0.15).toFixed(2)}"/>`;
  }
  for (let i = 0; i < 9; i++) s += sparkle(Math.round(R() * W), Math.round(R() * H), Math.round(5 + R() * 7), "rgba(246,234,208,.55)");
  const sky = $("#sky");
  sky.setAttribute("viewBox", `0 0 ${W} ${H}`);
  sky.setAttribute("preserveAspectRatio", "xMidYMid slice");
  sky.innerHTML = s;
}

const TONE_HEX = { good: "#f26a2e", warning: "#fab219", serious: "#ec835a", critical: "#e04848" };

function drawSun(alarm, flaring) {
  const glow = TONE_HEX[alarm.tone];
  const spikes = [];
  for (let i = 0; i < 48; i++) {
    const a = (i / 48) * Math.PI * 2, r = i % 2 ? 100 : 124;
    spikes.push(`${(Math.cos(a) * r).toFixed(1)},${(Math.sin(a) * r).toFixed(1)}`);
  }
  const R = rng(11);
  let speck = "";
  for (let i = 0; i < 160; i++) {
    const a = R() * Math.PI * 2, d = Math.sqrt(R()) * 80;
    speck += `<circle cx="${(Math.cos(a) * d).toFixed(1)}" cy="${(Math.sin(a) * d).toFixed(1)}" r="${(R() * 1.6 + .5).toFixed(2)}" fill="${R() < .7 ? "#f6a04d" : "#f26a2e"}" opacity=".55"/>`;
  }
  const prominence = flaring
    ? `<path d="M40,-72 C70,-120 120,-110 104,-62" fill="none" stroke="#e2569b" stroke-width="9" stroke-linecap="round"/>
       <path d="M40,-72 C70,-120 120,-110 104,-62" fill="none" stroke="#f6ead0" stroke-width="3" stroke-linecap="round" opacity=".8"/>` : "";
  $("#sun").innerHTML = `
    <defs>
      <radialGradient id="g-glow"><stop offset="0" stop-color="${glow}" stop-opacity=".55"/><stop offset=".55" stop-color="${glow}" stop-opacity=".16"/><stop offset="1" stop-color="${glow}" stop-opacity="0"/></radialGradient>
      <radialGradient id="g-disc" cx=".38" cy=".35"><stop offset="0" stop-color="#fff6e2"/><stop offset=".7" stop-color="#f6ead0"/><stop offset="1" stop-color="#f3c89a"/></radialGradient>
      <clipPath id="c-disc"><circle r="84"/></clipPath>
    </defs>
    <g class="breathe"><circle r="158" fill="url(#g-glow)"/></g>
    <circle r="142" fill="none" stroke="rgba(246,234,208,.16)" stroke-width="1" stroke-dasharray="2 7"/>
    <g class="spin"><polygon points="${spikes.join(" ")}" fill="#f26a2e"/></g>
    <circle r="98" fill="#f6a04d"/>
    <circle r="84" fill="url(#g-disc)"/>
    <g clip-path="url(#c-disc)">${speck}<path d="M-84,30 C-30,10 30,60 84,26 L84,90 L-84,90Z" fill="#f29bc4" opacity=".28"/></g>
    ${prominence}
    ${sparkle(-128, -108, 9, "#f6ead0")}${sparkle(132, 92, 7, "#f6ead0")}${sparkle(-140, 70, 5, "#f29bc4")}
    <g class="orbit-sat"><g transform="translate(142,0) rotate(90)">
      <rect x="-24" y="-6" width="16" height="12" rx="1.5" fill="#e2569b"/><path d="M-20,-6v12M-16,-6v12M-12,-6v12M-24,0h16" stroke="#9d2f68" stroke-width="1"/>
      <rect x="8" y="-6" width="16" height="12" rx="1.5" fill="#e2569b"/><path d="M12,-6v12M16,-6v12M20,-6v12M8,0h16" stroke="#9d2f68" stroke-width="1"/>
      <rect x="-8" y="-1.5" width="16" height="3" fill="#c4b2ec"/>
      <circle r="7" fill="#f6ead0"/><circle r="3.2" fill="#f29bc4"/>
    </g></g>`;
}

const ILL = {
  sat: `<circle cx="32" cy="32" r="30" fill="#2a1530"/>${sparkle(14, 14, 4, "#f6ead0")}
        <g transform="rotate(-30 32 32)"><rect x="8" y="27" width="16" height="10" rx="1.5" fill="#e2569b"/><rect x="40" y="27" width="16" height="10" rx="1.5" fill="#e2569b"/>
        <path d="M12 27v10M16 27v10M20 27v10M44 27v10M48 27v10M52 27v10" stroke="#9d2f68"/><rect x="24" y="30.5" width="16" height="3" fill="#c4b2ec"/><circle cx="32" cy="32" r="6.5" fill="#f6ead0"/><circle cx="32" cy="32" r="3" fill="#f29bc4"/></g>`,
  scope: `<circle cx="32" cy="32" r="30" fill="#2a1530"/><path d="M8 50h48v6H8z" fill="#3b1d40"/><path d="M14 50a18 18 0 0 1 36 0z" fill="#f6ead0"/>
          <path d="M30 32.5a18 18 0 0 1 6 0v17.5h-6z" fill="#9085e9"/><path d="M14 50h36" stroke="#e2569b" stroke-width="2"/>${sparkle(48, 16, 4, "#f6ead0")}<circle cx="18" cy="18" r="2" fill="#f26a2e"/>`,
  orb: `<circle cx="32" cy="32" r="30" fill="#2a1530"/><circle cx="32" cy="32" r="13" fill="#f26a2e"/><path d="M19 30h26M20 36h24" stroke="#f6a04d" stroke-width="2.5"/>
        <ellipse cx="32" cy="32" rx="24" ry="7" fill="none" stroke="#c4b2ec" stroke-width="2" transform="rotate(-18 32 32)"/>${sparkle(50, 14, 4, "#f6ead0")}`,
  loop: `<circle cx="32" cy="32" r="30" fill="#2a1530"/><path d="M20 26a13 13 0 0 1 23-4" fill="none" stroke="#e2569b" stroke-width="4" stroke-linecap="round"/><path d="M45 14l-1 9-9-2" fill="none" stroke="#e2569b" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
         <path d="M44 38a13 13 0 0 1-23 4" fill="none" stroke="#f26a2e" stroke-width="4" stroke-linecap="round"/><path d="M19 50l1-9 9 2" fill="none" stroke="#f26a2e" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>${sparkle(32, 32, 5, "#f6ead0")}`,
};

// ---------------------------------------------------------------- sound + notifications
let audio = null;
function beep(pattern) {
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    let t = audio.currentTime + 0.02;
    for (const [freq, dur] of pattern) {
      const o = audio.createOscillator(), g = audio.createGain();
      o.frequency.value = freq; o.type = "sine";
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.25, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(g).connect(audio.destination);
      o.start(t); o.stop(t + dur + 0.02);
      t += dur + 0.08;
    }
  } catch (e) { /* audio unavailable */ }
}
function setSound(on) {
  soundOn = on;
  const b = $("#sound-btn");
  b.setAttribute("aria-pressed", String(on));
  b.title = on ? "Alarm sound on: you'll hear a sound and get a notification when an alarm starts" : "Alarm sound off";
  try { localStorage.setItem("sound", on ? "1" : "0"); } catch (e) { /* storage blocked */ }
}
$("#sound-btn").addEventListener("click", async () => {
  setSound(!soundOn);
  if (soundOn) {
    beep([[880, 0.15]]);
    if ("Notification" in window && Notification.permission === "default") {
      try { await Notification.requestPermission(); } catch (e) { /* ignore */ }
    }
  }
});
try { if (localStorage.getItem("sound") === "1") setSound(true); } catch (e) { /* ignore */ }

function raiseAlarm(a) {
  if (!soundOn) return;
  beep(a.rank >= 3 ? [[988, 0.18], [740, 0.18], [988, 0.18], [740, 0.18]] : [[880, 0.2], [660, 0.25]]);
  if ("Notification" in window && Notification.permission === "granted") {
    try { new Notification(a.chip, { body: a.text, tag: "flare-alarm" }); } catch (e) { /* ignore */ }
  }
}

// ---------------------------------------------------------------- data
async function load() {
  try {
    const r = await fetch(API, { cache: "no-store" });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
    snap = DEMO ? demo(j, DEMO) : j;
    render();
    setBanner(j.error ? `The latest refresh failed, showing the previous update (${j.error}).` : null);
  } catch (e) {
    setBanner(snap ? `Could not refresh (${e.message}). Showing data from ${hhmm(snap.data_end)} UTC.`
                   : `Could not load data (${e.message}). Retrying every 30 seconds.`);
    $("#live").className = "live stale";
    if (!snap) $("#live-text").textContent = "Offline";
  }
}

function setBanner(msg) {
  const b = $("#banner");
  if (!msg) { b.hidden = true; return; }
  b.hidden = false;
  b.innerHTML = `${ICON.warning}<span>${esc(msg)}</span>`;
}

// Preview the alarm states on quiet days: ?demo=watch | major | flare
function demo(s, kind) {
  s = structuredClone(s);
  const t = s.data_end;
  if (kind === "flare") {
    const fl = { id: "demo", start: t - 9 * 60e3, detect: t - 5 * 60e3, peak_time: t - 60e3, end: null, ongoing: true,
                 peak: 2.4e-5, class: "M2.4", confirmed: true, detect_delay_min: 4 };
    s.flares.unshift(fl);
    s.now = { ...s.now, state: "FLARE", flux: 2.4e-5, class: "M2.4", ongoing: fl };
    for (const k of ["C1", "M1"]) s.forecasts[k].prob = null;
  } else {
    const k = kind === "major" ? "M1" : "C1";
    const f = s.forecasts[k];
    f.prob = kind === "major" ? 0.12 : 0.55;
    f.alert = true;
    f.alerts_48h.push({ start: t - 4 * 60e3, end: t });
    if (kind === "major") { s.forecasts.C1.prob = 0.62; s.forecasts.C1.alert = true; s.forecasts.C1.alerts_48h.push({ start: t - 6 * 60e3, end: t }); }
  }
  return s;
}

// ---------------------------------------------------------------- alarm logic
function currentAlarm(s) {
  const n = s.now, C = s.forecasts.C1, M = s.forecasts.M1;
  if (n.state === "FLARE") {
    const cls = n.ongoing?.class || n.class;
    const major = cls && "MX".includes(cls[0]);
    return {
      rank: major ? 4 : 3, tone: major ? "critical" : "serious", big: cls || "Flare",
      chip: major ? "Major flare in progress" : "Flare in progress",
      text: n.ongoing ? `Started ${hhmm(n.ongoing.start)} UTC, brightest so far at ${hhmm(n.ongoing.peak_time)} UTC.` : "Detected by the nowcast.",
    };
  }
  if (M.alert) {
    const since = M.alerts_48h.at(-1)?.start;
    return {
      rank: 2, tone: "serious", big: "Major watch", chip: "M-class flare alarm",
      text: `${pct(M.prob)} chance of an M-class or bigger flare within 15 min${since ? ` (alarm since ${hhmm(since)} UTC)` : ""}. Most of these pass without a flare: be ready, not alarmed.`,
    };
  }
  if (C.alert) {
    const since = C.alerts_48h.at(-1)?.start;
    return {
      rank: 1, tone: "warning", big: "Flare watch", chip: "C-class flare alarm",
      text: `${pct(C.prob)} chance of a C-class or bigger flare within 15 min${since ? ` (alarm since ${hhmm(since)} UTC)` : ""}.`,
    };
  }
  if (n.state === "DECAY") {
    return { rank: 0, tone: "good", big: "Flare ending", chip: "No alarm", text: "The last flare is fading. No new flare expected in the next 15 minutes." };
  }
  return { rank: 0, tone: "good", big: "All clear", chip: "No alarm", text: "The Sun is quiet: no flare happening and none expected in the next 15 minutes." };
}

function alarmLog(s) {
  const end = s.data_end;
  const rows = [];
  for (const k of ["C1", "M1"]) {
    for (const a of s.forecasts[k].alerts_48h) {
      const ongoing = a.end >= end;
      const hit = s.flares.find((f) => f.detect >= a.start && f.detect <= a.end + 15 * 60e3 && (f.peak || 0) >= MIN_FLUX[k]);
      let result;
      if (hit) result = pill("good", `Flare followed: ${hit.class} at ${hhmm(hit.detect)}`);
      else if (a.end + 15 * 60e3 > end) result = pill("neutral", "Waiting");
      else result = `<span class="dim">No flare (false alarm)</span>`;
      const dur = ongoing ? "still on" : `${Math.max(1, mins(a.end - a.start + 60e3))} min`;
      rows.push({ t: a.start, event: pill(k === "M1" ? "serious" : "warning", k === "M1" ? "M-flare alarm" : "C-flare alarm"), detail: dur, result });
    }
  }
  for (const f of s.flares) {
    if (!f.class || !"CMX".includes(f.class[0])) continue;
    const warned = s.forecasts.C1.alerts_48h.concat(s.forecasts.M1.alerts_48h)
      .filter((a) => a.start < f.detect && a.end >= f.detect - 15 * 60e3)
      .reduce((m, a) => Math.min(m, a.start), Infinity);
    rows.push({
      t: f.detect,
      event: pill("MX".includes(f.class[0]) ? "critical" : "serious", `Flare ${f.class}`),
      detail: f.ongoing ? "ongoing" : `peak ${hhmm(f.peak_time)}, ${mins(f.end - f.start)} min`,
      result: Number.isFinite(warned) ? `Warned ${mins(f.detect - warned)} min ahead` : `<span class="dim">No advance warning</span>`,
    });
  }
  return rows.sort((a, b) => b.t - a.t);
}

// ---------------------------------------------------------------- render
function render() {
  const s = snap;
  const live = $("#live");
  live.className = `live ${s.stale ? "stale" : "ok"}`;
  $("#live-text").textContent = s.stale ? `Data delayed · ${hhmm(s.data_end)} UTC` : `Live · data ${hhmm(s.data_end)} UTC`;
  live.title = `Data up to ${dateTime(s.data_end)} UTC (${s.data_age_min} min old). Checked every 30 s.`;

  // stats
  $("#st-class").textContent = s.now.class || "—";
  $("#st-state").textContent = STATE_LABEL[s.now.state];
  const day = s.flares.filter((f) => f.start >= s.data_end - 24 * 3600e3);
  const big = day.reduce((a, f) => (!a || (f.peak || 0) > (a.peak || 0) ? f : a), null);
  $("#st-count").innerHTML = `${day.length}${big ? ` <span class="dim" style="font-size:13px;font-weight:500">max ${esc(big.class)}</span>` : ""}`;

  // hero / alarm board
  const a = currentAlarm(s);
  $("#hero").className = `hero ${a.tone}`;
  $("#tone-chip").className = `tone-chip ${a.tone}`;
  $("#tone-chip").innerHTML = `${ICON[a.tone]}${esc(a.chip)}`;
  $("#big").textContent = a.big;
  $("#lede").textContent = a.text;
  drawSun(a, s.now.state === "FLARE");
  if (lastAlarm && a.rank > lastAlarm.rank) raiseAlarm(a);
  lastAlarm = a;
  document.title = a.rank > 0 ? `⚠ ${a.chip} · FlareWatch` : "FlareWatch";

  renderModelPill(s);
  renderRings(s);
  renderAlarmLog(s);
  $("#legend-prob").innerHTML = [["s-c1", "C-class or bigger"], ["s-m1", "M-class or bigger"]]
    .map(([c, l]) => `<span><i style="background:var(--${c})"></i>${l}</span>`).join("");
  $("#legend-bars").innerHTML = ["B", "C", "M", "X"]
    .map((l) => `<span><i class="sq" style="background:${BAR_COLORS[l]}"></i>${l}-class</span>`).join("");
  drawCharts();
  renderRecentTable(s);
  renderModel(s);
}

function renderModelPill(s) {
  const C = s.forecasts.C1.model, M = s.forecasts.M1.model;
  const same = C.version === M.version;
  $("#pill-version").textContent = same ? verLabel(C) : `C ${verLabel(C)} · M ${verLabel(M)}`;
  const last = (s.learning || [])[0];
  $("#pill-trained").textContent = `data up to ${dayLabel(Math.min(C.train_end, M.train_end))}`
    + (last ? ` · checked ${dayLabel(Date.parse(last.run_at.replace(" ", "T") + "Z"))}` : "");
  $("#version").innerHTML = `Model <b>${esc(same ? verLabel(C) : `C ${verLabel(C)} / M ${verLabel(M)}`)}</b> · <a href="#model">Retraining log ›</a>`;
}

function ring(id, frac, value, label, sub, color) {
  const r = 36, c = 2 * Math.PI * r, f = Math.max(0, Math.min(1, frac || 0));
  $(id).innerHTML = `<div class="ring"><svg viewBox="0 0 86 86"><circle class="track" cx="43" cy="43" r="${r}"/>`
    + `<circle class="arc" cx="43" cy="43" r="${r}" stroke="${color}" stroke-dasharray="${c.toFixed(1)}" stroke-dashoffset="${(c * (1 - f)).toFixed(1)}"/></svg>`
    + `<div class="val">${value}</div></div><div class="ring-label"><b>${label}</b>${sub}</div>`;
}

function renderRings(s) {
  const C = s.forecasts.C1, M = s.forecasts.M1;
  // probability rings fill toward the alarm level: a full ring means "alarm"
  ring("#ring-C1", C.prob == null ? 0 : C.prob / C.threshold, C.prob == null ? "—" : pct(C.prob), "C flare, 15 min", `alarm at ${pct(C.threshold)}`, "var(--s-c1)");
  ring("#ring-M1", M.prob == null ? 0 : M.prob / M.threshold, M.prob == null ? "—" : pct(M.prob), "M flare, 15 min", `alarm at ${pct(M.threshold)}`, "var(--s-m1)");
  ring("#ring-caught", C.skill.pod, `${Math.round(C.skill.pod * 100)}%`, "Flares caught", "C model, tested", "var(--s-flux)");
  ring("#ring-false", C.skill.far, `${Math.round(C.skill.far * 100)}%`, "False alarms", "C model, tested", "var(--muted)");
}

function renderAlarmLog(s) {
  const log = alarmLog(s);
  $("#alarm-log").innerHTML = log.length
    ? `<thead><tr><th>UTC</th><th>Event</th><th>Details</th><th>What happened</th></tr></thead><tbody>`
      + log.map((r) => `<tr><td>${dateTime(r.t)}</td><td>${r.event}</td><td>${esc(r.detail)}</td><td>${r.result}</td></tr>`).join("")
      + `</tbody>`
    : `<tbody><tr><td class="empty">No alarms and no C-class or bigger flares in the last 48 hours.</td></tr></tbody>`;
}

// ---------------------------------------------------------------- charts
function windowed() {
  const S = snap.series;
  const t1 = snap.data_end, t0 = t1 - rangeH * 3600e3;
  let i0 = S.t.findIndex((t) => t > t0);
  if (i0 < 0) i0 = 0;
  return { S, i0, t0, t1 };
}

function timeTicks(t0, t1, width) {
  const steps = [1, 2, 3, 6, 12].map((h) => h * 3600e3);
  const want = Math.max(2, Math.floor(width / 85));
  const step = steps.find((s) => (t1 - t0) / s <= want) || steps[steps.length - 1];
  const out = [];
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) out.push(t);
  return out;
}

function linePath(ts, vs, x, y, i0, ok = (v) => v != null) {
  let d = "", pen = false;
  for (let i = i0; i < ts.length; i++) {
    const v = vs[i];
    if (!ok(v)) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${x(ts[i]).toFixed(1)},${y(v).toFixed(1)}`;
    pen = true;
  }
  return d;
}

function frame(el, padL, padR) {
  const W = el.clientWidth, H = el.clientHeight;
  const m = { l: padL, r: padR, t: 12, b: 22 };
  return { W, H, m, iw: W - m.l - m.r, ih: H - m.t - m.b };
}

function axisX(F, x, t0, t1) {
  let g = "";
  for (const t of timeTicks(t0, t1, F.iw)) {
    const lab = new Date(t).getUTCHours() === 0 ? dayLabel(t) : hhmm(t);
    g += `<text class="tick" x="${x(t)}" y="${F.m.t + F.ih + 16}" text-anchor="middle">${lab}</text>`;
  }
  return g;
}

function alarmSpans(F, x, t0, t1) {
  let g = "";
  for (const k of ["C1", "M1"]) {
    for (const a of snap.forecasts[k].alerts_48h) {
      if (a.end < t0) continue;
      const xa = x(Math.max(a.start, t0)), xb = x(Math.min(a.end + 60e3, t1));
      g += `<rect x="${xa.toFixed(1)}" y="${F.m.t}" width="${Math.max(2, xb - xa).toFixed(1)}" height="${F.ih}" fill="rgba(250,178,25,.13)"/>`;
    }
  }
  return g;
}

function drawFlux() {
  const el = $("#chart-flux");
  const { S, i0, t0, t1 } = windowed();
  const F = frame(el, 8, 30);
  const x = (t) => F.m.l + ((t - t0) / (t1 - t0)) * F.iw;
  let lo = Infinity, hi = 0;
  for (let i = i0; i < S.t.length; i++) { const v = S.xrsb[i]; if (v > 0) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
  const e0 = Math.min(-7, Math.floor(Math.log10(lo || 1e-7))), e1 = Math.max(-5, Math.ceil(Math.log10((hi || 1e-6) * 1.3)));
  const y = (v) => F.m.t + (1 - (Math.log10(v) - e0) / (e1 - e0)) * F.ih;
  const base = F.m.t + F.ih;

  let g = `<defs><linearGradient id="g-area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#e2569b" stop-opacity=".55"/><stop offset="1" stop-color="#e2569b" stop-opacity="0"/></linearGradient></defs>`;
  g += alarmSpans(F, x, t0, t1);
  const visible = snap.flares.filter((f) => (f.end ?? t1) >= t0);
  for (const f of visible) {
    const a = x(Math.max(f.start, t0)), b = x(Math.min(f.end ?? t1, t1));
    g += `<rect x="${a.toFixed(1)}" y="${F.m.t}" width="${Math.max(2, b - a).toFixed(1)}" height="${F.ih}" fill="rgba(242,106,46,.10)"/>`;
  }
  for (let e = e0; e <= e1; e++) {
    const yy = y(10 ** e);
    g += `<line x1="${F.m.l}" x2="${F.m.l + F.iw}" y1="${yy}" y2="${yy}" stroke="var(--grid)"/>`;
    const cl = CLASSES.find(([, b]) => Math.abs(Math.log10(b) - e) < 0.01);
    if (cl && e < e1) g += `<text class="cls" x="${F.m.l + F.iw + 12}" y="${(y(10 ** e) + y(10 ** (e + 1))) / 2 + 4}">${cl[0]}</text>`;
  }
  // area: break at gaps
  let area = "", seg = [];
  const flush = () => { if (seg.length > 1) area += `M${seg[0][0]},${base}L${seg.map((p) => p.join(",")).join("L")}L${seg.at(-1)[0]},${base}Z`; seg = []; };
  for (let i = i0; i < S.t.length; i++) {
    const v = S.xrsb[i];
    if (!(v > 0)) { flush(); continue; }
    seg.push([x(S.t[i]).toFixed(1), y(v).toFixed(1)]);
  }
  flush();
  g += `<path d="${area}" fill="url(#g-area)"/>`;
  g += `<path d="${linePath(S.t, S.xrsb, x, y, i0, (v) => v > 0)}" fill="none" stroke="var(--s-flux)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  g += axisX(F, x, t0, t1);
  const biggest = visible.reduce((a, f) => (!a || f.peak > a.peak ? f : a), null);
  let lastX = -1e9;
  for (const f of [...visible].sort((a, b) => a.peak_time - b.peak_time)) {
    if (!(f.class && ("CMX".includes(f.class[0]) || f === biggest))) continue;
    const px = x(Math.min(Math.max(f.peak_time, t0), t1));
    if (px - lastX < 36) continue;
    lastX = px;
    const py = Math.max(F.m.t + 14, y(f.peak) - 12);
    g += `<rect x="${px - 19}" y="${py - 11}" width="38" height="16" rx="8" fill="rgba(246,234,208,.92)"/><text x="${px}" y="${py + 1}" text-anchor="middle" style="font-size:10.5px;font-weight:600;fill:#2a1530">${esc(f.class)}</text>`;
  }
  const last = S.xrsb[S.t.length - 1];
  if (last > 0) g += `<circle cx="${x(t1)}" cy="${y(last)}" r="5" fill="var(--s-flux)" stroke="var(--plum-800)" stroke-width="2"/>`;
  mount(el, F, g, x, y, [{ key: "xrsb", color: "s-flux", ok: (v) => v > 0 }], "Sun's X-ray brightness chart");
}

function drawProb() {
  const el = $("#chart-prob");
  const { S, i0, t0, t1 } = windowed();
  const F = frame(el, 40, 62);
  const x = (t) => F.m.l + ((t - t0) / (t1 - t0)) * F.iw;
  const y = (p) => F.m.t + (1 - (Math.log10(Math.max(p, P_MIN)) + 3) / 3) * F.ih;
  let g = alarmSpans(F, x, t0, t1);
  for (const [p, l] of [[0.001, "0.1%"], [0.01, "1%"], [0.1, "10%"], [1, "100%"]]) {
    g += `<line x1="${F.m.l}" x2="${F.m.l + F.iw}" y1="${y(p)}" y2="${y(p)}" stroke="var(--grid)"/>`;
    g += `<text class="tick" x="${F.m.l - 8}" y="${y(p) + 4}" text-anchor="end">${l}</text>`;
  }
  g += axisX(F, x, t0, t1);
  const ends = [];
  for (const [k, c, name] of [["C1", "s-c1", "C"], ["M1", "s-m1", "M"]]) {
    const f = snap.forecasts[k];
    const ty = y(f.threshold);
    g += `<line x1="${F.m.l}" x2="${F.m.l + F.iw}" y1="${ty}" y2="${ty}" stroke="var(--${c})" stroke-width="1" opacity=".55"/>`;
    g += `<text class="tick halo" x="${F.m.l + F.iw - 4}" y="${ty - 5}" text-anchor="end">${name} alarm level</text>`;
    g += `<path d="${linePath(S.t, S[`p_${k}`], x, y, i0)}" fill="none" stroke="var(--${c})" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    const lp = S[`p_${k}`][S.t.length - 1];
    if (lp != null) {
      g += `<circle cx="${x(t1)}" cy="${y(lp)}" r="4" fill="var(--${c})" stroke="var(--plum-800)" stroke-width="2"/>`;
      ends.push({ y: y(lp), text: `${name} ${pct(lp)}` });
    }
  }
  if (ends.length === 2 && Math.abs(ends[0].y - ends[1].y) < 14) {
    const [a, b] = ends[0].y <= ends[1].y ? ends : [ends[1], ends[0]];
    const mid = (a.y + b.y) / 2;
    a.ly = mid - 8; b.ly = mid + 8;
  }
  for (const e of ends) {
    const ly = e.ly ?? e.y;
    if (e.ly != null) g += `<path d="M${x(t1) + 5},${e.y}L${x(t1) + 10},${ly}" stroke="var(--muted)" fill="none"/>`;
    g += `<text class="end-lbl" x="${x(t1) + 12}" y="${ly + 4}">${e.text}</text>`;
  }
  mount(el, F, g, x, y, [
    { key: "p_C1", color: "s-c1", ok: (v) => v != null },
    { key: "p_M1", color: "s-m1", ok: (v) => v != null },
  ], "Flare chance chart");
}

function drawBars() {
  const el = $("#chart-bars");
  const t1 = snap.data_end, t0 = t1 - 48 * 3600e3, step = 6 * 3600e3;
  const bins = [];
  for (let b = 0; b < 8; b++) bins.push({ s: t0 + b * step, e: t0 + (b + 1) * step, n: 0, max: null });
  for (const f of snap.flares) {
    const b = bins.find((x) => f.start >= x.s && f.start < x.e);
    if (!b) continue;
    b.n++;
    if (!b.max || (f.peak || 0) > (b.max.peak || 0)) b.max = f;
  }
  const F = frame(el, 26, 8);
  const top = Math.max(2, ...bins.map((b) => b.n));
  const y = (v) => F.m.t + 14 + (1 - v / top) * (F.ih - 14);
  const bw = F.iw / bins.length, w = Math.min(24, bw * 0.55);
  let g = "";
  for (let v = 0; v <= top; v += Math.max(1, Math.ceil(top / 4))) {
    g += `<line x1="${F.m.l}" x2="${F.m.l + F.iw}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text class="tick" x="${F.m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  }
  bins.forEach((b, i) => {
    const cx = F.m.l + bw * (i + 0.5);
    if (b.n) {
      const yt = y(b.n), yb = y(0), r = Math.min(4, (yb - yt) / 2);
      const col = BAR_COLORS[(b.max?.class || "B")[0]] || BAR_COLORS.B;
      g += `<path d="M${cx - w / 2},${yb}V${yt + r}Q${cx - w / 2},${yt} ${cx - w / 2 + r},${yt}H${cx + w / 2 - r}Q${cx + w / 2},${yt} ${cx + w / 2},${yt + r}V${yb}Z" fill="${col}"/>`;
      g += `<text class="lbl" x="${cx}" y="${yt - 6}" text-anchor="middle">${b.n}</text>`;
    }
    g += `<rect class="bar-hit" data-i="${i}" x="${cx - bw / 2}" y="${F.m.t}" width="${bw}" height="${F.ih}" fill="transparent"/>`;
    if (i % 2 === 0) g += `<text class="tick" x="${cx}" y="${F.m.t + F.ih + 16}" text-anchor="middle">${new Date(b.s).getUTCHours() === 0 ? dayLabel(b.s) : hhmm(b.s)}</text>`;
  });
  el.innerHTML = `<svg viewBox="0 0 ${F.W} ${F.H}" role="img" aria-label="Flares per 6 hours">${g}</svg>`;
  el.querySelectorAll(".bar-hit").forEach((r) => {
    const show = (ev) => {
      const b = bins[+r.dataset.i], p = ev.touches ? ev.touches[0] : ev;
      tooltip(`<div class="t">${dateTime(b.s)}–${hhmm(b.e)} UTC</div><div class="r"><span>Flares</span><b>${b.n}</b></div>`
        + `<div class="r"><span>Biggest</span><b>${b.max ? esc(b.max.class) : "—"}</b></div>`, p);
    };
    r.addEventListener("mousemove", show);
    r.addEventListener("touchstart", show, { passive: true });
    r.addEventListener("mouseleave", hideHover);
  });
}

function mount(el, F, body, x, y, keys, label) {
  el.innerHTML = `<svg viewBox="0 0 ${F.W} ${F.H}" role="img" aria-label="${label}">${body}<g class="hover"></g><rect class="hit" x="${F.m.l}" y="0" width="${F.iw}" height="${F.H}" fill="transparent"/></svg>`;
  charts[el.id] = { el, F, x, y, keys };
  const hit = el.querySelector(".hit");
  const move = (ev) => {
    const p = ev.touches ? ev.touches[0] : ev;
    const r = el.getBoundingClientRect();
    const { t0, t1 } = windowed();
    showHover(t0 + ((p.clientX - r.left - F.m.l) / F.iw) * (t1 - t0), p);
  };
  hit.addEventListener("mousemove", move);
  hit.addEventListener("touchstart", move, { passive: true });
  hit.addEventListener("touchmove", move, { passive: true });
  hit.addEventListener("mouseleave", hideHover);
  hit.addEventListener("touchend", () => setTimeout(hideHover, 1500));
}

function nearestIndex(t) {
  const ts = snap.series.t;
  let lo = 0, hi = ts.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ts[mid] < t) lo = mid; else hi = mid; }
  return Math.abs(ts[lo] - t) <= Math.abs(ts[hi] - t) ? lo : hi;
}

function tooltip(html, pt) {
  const tip = $("#tooltip");
  tip.innerHTML = html;
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let left = pt.clientX + 14, top = pt.clientY - h - 12;
  if (left + w > innerWidth - 8) left = pt.clientX - w - 14;
  if (top < 8) top = pt.clientY + 16;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${top}px`;
}

function showHover(tHover, pt) {
  const S = snap.series;
  const i = nearestIndex(tHover);
  const t = S.t[i];
  for (const c of Object.values(charts)) {
    const xx = c.x(t);
    let h = `<line x1="${xx}" x2="${xx}" y1="${c.F.m.t}" y2="${c.F.m.t + c.F.ih}" stroke="rgba(246,234,208,.35)"/>`;
    for (const k of c.keys) {
      const v = S[k.key][i];
      if (k.ok(v)) h += `<circle cx="${xx}" cy="${c.y(v)}" r="5" fill="var(--${k.color})" stroke="#f6ead0" stroke-width="2"/>`;
    }
    c.el.querySelector(".hover").innerHTML = h;
  }
  const st = ["QUIET", "FLARE", "DECAY"][S.state[i]];
  const row = (c, l, v) => `<div class="r"><span>${c ? `<i style="background:var(--${c})"></i>` : ""}${l}</span><b>${v}</b></div>`;
  tooltip(`<div class="t">${dateTime(t)} UTC</div>`
    + row("s-flux", "X-ray level", goesClass(S.xrsb[i]))
    + row(null, "Status", STATE_LABEL[st])
    + row("s-c1", "C chance", st === "FLARE" ? "—" : pct(S.p_C1[i]))
    + row("s-m1", "M chance", st === "FLARE" ? "—" : pct(S.p_M1[i])), pt);
}

function hideHover() {
  $("#tooltip").hidden = true;
  for (const c of Object.values(charts)) c.el.querySelector(".hover").innerHTML = "";
}

function drawCharts() {
  if (!snap) return;
  drawFlux();
  drawProb();
  drawBars();
}

// ---------------------------------------------------------------- tables + model
function renderRecentTable(s) {
  const S = s.series, n = S.t.length;
  let rows = "";
  for (let i = n - 1; i >= Math.max(0, n - 60); i--) {
    const st = ["QUIET", "FLARE", "DECAY"][S.state[i]];
    rows += `<tr><td>${hhmm(S.t[i])}</td><td>${goesClass(S.xrsb[i])}</td><td>${STATE_LABEL[st]}</td>`
      + `<td class="num">${pct(S.p_C1[i])}</td><td class="num">${pct(S.p_M1[i])}</td></tr>`;
  }
  $("#recent-table").innerHTML = `<thead><tr><th>UTC</th><th>X-ray level</th><th>Status</th><th class="num">C chance</th><th class="num">M chance</th></tr></thead><tbody>${rows}</tbody>`;
}

function plainReason(e) {
  if (e.promoted) return `New version did better on ${Math.round(e.unseen_days)} unseen days, now live`;
  if (/unseen/.test(e.reason)) return "Waiting for enough new days to test it fairly";
  if (/CSI worse/.test(e.reason)) return "New version raised worse alarms, kept the current one";
  return "New version was not better, kept the current one";
}

function renderModel(s) {
  const kv = (k, v) => `<div class="kv"><div class="k">${k}</div><div class="v">${v}</div></div>`;
  $("#model-cards").innerHTML = [["C1", "C-class or bigger", "s-c1"], ["M1", "M-class or bigger", "s-m1"]].map(([k, name, c]) => {
    const f = s.forecasts[k], sk = f.skill, m = f.model;
    const rule = f.rule_m > 1 ? `≥ ${pct(f.threshold)} for ${f.rule_m} of ${f.rule_n} min` : `≥ ${pct(f.threshold)}`;
    return `<article class="mcard"><div class="mcard-head"><b><i style="background:var(--${c})"></i>${name}</b>`
      + `<span>${verLabel(m)} · data up to ${dayLabel(m.train_end)} · alarm ${rule}</span></div>`
      + `<div class="kvs">${kv("Flares caught", `${Math.round(sk.pod * 100)}%`)}${kv("Alarms that were false", `${Math.round(sk.far * 100)}%`)}`
      + `${kv("False alarms per day", sk.false_alarms_per_day.toFixed(1))}${kv("Typical warning", `${sk.lead_median_min} min`)}</div>`
      + `<p class="mnote">Tested on ${sk.flares} flares, ${esc(sk.period)}.${k === "M1" ? " Treat an M alarm as \"be ready\": most pass without a flare." : " Most alarms fire in the first minutes of a flare's rise."}</p></article>`;
  }).join("");
  const rows = s.learning || [];
  $("#learning-log").innerHTML = rows.length
    ? `<thead><tr><th>Run (UTC)</th><th>Model</th><th>Result</th><th>Why</th></tr></thead><tbody>`
      + rows.map((e) => `<tr><td>${esc(e.run_at.slice(0, 16))}</td><td>${e.target === "M1" ? "M-class" : "C-class"}</td>`
        + `<td>${e.promoted ? pill("good", "Updated") : pill("neutral", "Kept current")}</td>`
        + `<td class="wrap">${esc(plainReason(e))}</td></tr>`).join("") + `</tbody>`
    : `<tbody><tr><td class="empty">No retraining runs yet. Run <code>python -m continual daily</code> or start the server with <code>--retrain-daily</code>.</td></tr></tbody>`;
}

// ---------------------------------------------------------------- wiring
function tickClock() {
  const d = new Date();
  $("#st-clock").textContent = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
}

document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
  rangeH = +b.dataset.range;
  document.querySelectorAll(".seg button").forEach((o) => o.classList.toggle("on", o === b));
  hideHover();
  drawFlux();
  drawProb();
}));

// highlight the rail item of the section in view
const navs = [...document.querySelectorAll(".rail .nav[data-sec]")];
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    const id = e.target.id === "hero" ? "top" : e.target.id;
    navs.forEach((n) => n.classList.toggle("on", n.dataset.sec === id));
  }
}, { rootMargin: "-45% 0px -50% 0px" });
["hero", "alarms", "charts", "model"].forEach((id) => io.observe(document.getElementById(id)));

let rz;
new ResizeObserver(() => { clearTimeout(rz); rz = setTimeout(drawCharts, 80); }).observe($("main"));

document.querySelectorAll("svg.ill").forEach((el) => { el.innerHTML = ILL[el.dataset.ill]; });
drawSky();
drawSun({ tone: "good" }, false);
tickClock();
setInterval(tickClock, 1000);
load();
setInterval(load, REFRESH_MS);
