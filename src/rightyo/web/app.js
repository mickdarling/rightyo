/* SPDX-License-Identifier: AGPL-3.0-or-later */
(() => {
  "use strict";
  // The local capability stays in this closure. Never store or log it.
  const token = location.hash.slice(1);
  history.replaceState(null, "", location.pathname + location.search);
  const authorized = /^[A-Za-z0-9_-]{24,128}$/.test(token);
  const $ = (id) => document.getElementById(id);
  const activePhases = new Set(["starting", "listening", "replaying", "finishing", "stopping"]);
  const phaseNames = {
    idle: "Idle · microphone off", starting: "Starting · awaiting audio",
    listening: "Listening · microphone on", replaying: "Replaying · microphone off",
    finishing: "Finishing decisions · microphone off", complete: "Demo complete · microphone off",
    error: "Session stopped · incomplete", offline: "Disconnected · context cleared",
    stopping: "Stopping · clearing context",
  };
  const decisionNames = {attend: "For the agent", ignore: "Other conversation", uncertain: "Uncertain"};
  let retainedTurns = [];
  let phase = "idle";
  let connected = false;
  let demoAvailable = false;
  let busy = false;
  let polling = false;
  let generation = 0;
  let mustStop = false;
  let closed = false;

  function timeLabel(milliseconds) {
    const seconds = Math.max(0, Math.floor(Number(milliseconds || 0) / 1000));
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }

  function message(text, error = false) {
    $("message").textContent = text;
    $("message").dataset.error = String(error);
  }

  function controls() {
    const active = activePhases.has(phase);
    $("start-button").disabled = !connected || active || busy;
    $("demo-button").disabled = !connected || !demoAvailable || active || busy;
    $("stop-button").disabled = !connected || busy;
    $("settings-fields").disabled = active || busy;
    $("confidence").disabled = active || busy || !$("use-jev").checked;
    $("max-requests").disabled = active || busy || !$("use-jev").checked;
    $("copy-button").disabled = !connected || retainedTurns.length === 0 || busy;
    $("source-help").textContent = demoAvailable
      ? "The demo uses configured audio and leaves your microphone off."
      : "No audio demo is configured. Start microphone to try live speech.";
  }

  function setPhase(value) {
    phase = Object.hasOwn(phaseNames, value) ? value : "error";
    $("phase-pill").dataset.phase = phase;
    $("phase-label").textContent = phaseNames[phase];
  }

  function clearContext() {
    retainedTurns = [];
    $("transcript").replaceChildren();
    $("empty-state").hidden = false;
    $("turn-count").textContent = "0";
    $("context-span").textContent = "0:00";
    $("attention-status").textContent = "Decisions off";
    $("attention-dot").dataset.state = "off";
    $("attention-counts").textContent = "Enable Jev to test addressedness.";
    $("jev-model-tag").textContent = "OFF";
    $("request-count").textContent = `0 / ${$("max-requests").value}`;
    $("copy-button").disabled = true;
  }

  async function api(path, body, keepalive = false) {
    const controller = new AbortController();
    const timer = keepalive ? null : setTimeout(() => controller.abort(), 5000);
    try {
      const response = await fetch(path, {
        method: body === undefined ? "GET" : "POST",
        headers: {Authorization: `Bearer ${token}`, "Content-Type": "application/json"},
        body: body === undefined ? undefined : JSON.stringify(body),
        cache: "no-store", credentials: "omit", referrerPolicy: "no-referrer",
        signal: controller.signal, keepalive,
      });
      if (!response.ok) throw new Error("Local control request failed");
      return await response.json();
    } finally {
      if (timer !== null) clearTimeout(timer);
    }
  }

  function disconnect() {
    generation += 1;
    connected = false;
    mustStop = true;
    clearContext();
    setPhase("offline");
    message("Connection lost. Context was cleared. Capture stops when the local server receives Stop, or its 15-second browser lease expires.", true);
    controls();
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  function stage(id, backend, question) {
    // Labels come from the server's backend summary (kind, hosted, service name only).
    if (!backend || typeof backend !== "object") return;
    const hosted = backend.hosted === true;
    $(`${id}-tag`).textContent = hosted ? "HOSTED" : "LOCAL";
    $(`${id}-name`).textContent = String(backend.service || backend.kind || "Unknown");
    $(`${id}-detail`).textContent = `${question} · ${hosted ? "audio sent to the hosted service" : "on this Mac"}`;
  }

  function hostedSpeech(state) {
    return (state.transcriber && state.transcriber.hosted === true) || (state.diarizer && state.diarizer.hosted === true);
  }

  function render(state) {
    stage("diarizer", state.diarizer, "Who is speaking");
    stage("transcriber", state.transcriber, "What was said");
    $("locality-label").textContent = hostedSpeech(state) ? "Audio leaves this Mac · hosted speech configured" : "On this Mac";
    const transcriber = state.transcriber && typeof state.transcriber === "object" ? state.transcriber : null;
    $("footer-locality").textContent = transcriber && transcriber.hosted === true
      ? `Hosted transcription (${String(transcriber.service || transcriber.kind || "hosted")}) · temporary context`
      : "Local transcription · temporary context";
    connected = true;
    demoAvailable = state.demo_available === true;
    setPhase(state.phase);
    retainedTurns = Array.isArray(state.turns) ? state.turns : [];
    const decisions = state.decisions || {};
    const retention = state.retention || {};
    const count = {attend: 0, ignore: 0, uncertain: 0};
    const fragment = document.createDocumentFragment();
    const colors = new Map();
    const scroll = $("transcript");
    const atBottom = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 50;
    for (const turn of retainedTurns) {
      const item = element("li", "turn");
      const header = element("div", "turn-header");
      const speaker = turn.speaker_id || "Speaker unknown";
      if (turn.speaker_id && !colors.has(speaker)) colors.set(speaker, colors.size % 4);
      const dot = element("span", "speaker-dot");
      dot.dataset.color = turn.speaker_id ? String(colors.get(speaker)) : "unknown";
      dot.setAttribute("aria-hidden", "true");
      header.append(dot, element("span", "speaker-name", speaker),
        element("span", "turn-time", `${timeLabel(turn.start_ms)}–${timeLabel(turn.end_ms)}`));
      const decision = decisions[turn.utterance_id];
      if (decision && Object.hasOwn(count, decision.label)) {
        count[decision.label] += 1;
        const badge = element("span", "decision-badge", decisionNames[decision.label]);
        badge.dataset.label = decision.label;
        header.append(badge);
      }
      item.append(header, element("p", "turn-text", turn.text));
      const notes = [];
      if (turn.overlap) notes.push("Overlapping speech");
      if (!turn.speaker_id) notes.push("Speaker attribution uncertain");
      if (decision) {
        const recipient = decision.recipient_speaker_id || ({system: "agent", other_human: "another human", unknown: "unknown"}[decision.recipient_kind]) || "unknown";
        notes.push(`Recipient: ${recipient}`);
        const confidence = Number(decision.confidence);
        if (Number.isFinite(confidence)) notes.push(`Decision score ${confidence.toFixed(2)}`);
        if (decision.policy_abstained) notes.push("Confidence or recipient policy abstained");
      }
      if (notes.length) item.append(element("p", "turn-meta", notes.join(" · ")));
      fragment.append(item);
    }
    scroll.replaceChildren(fragment);
    if (atBottom) scroll.scrollTop = scroll.scrollHeight;
    $("empty-state").hidden = retainedTurns.length > 0;
    $("turn-count").textContent = String(retainedTurns.length);
    const span = retainedTurns.length
      ? Number(retention.newest_end_ms || 0) - Number(retention.oldest_start_ms || 0) : 0;
    $("context-span").textContent = timeLabel(span);
    $("request-count").textContent = `${state.jev_requests || 0} / ${state.jev_request_limit || 20}`;
    const status = state.decision_status || "off";
    $("attention-dot").dataset.state = status;
    $("jev-model-tag").textContent = status === "off" ? "OFF" : "OPT-IN";
    $("attention-status").textContent = {
      off: "Decisions off", ready: "Jev decisions enabled",
      unavailable: "Jev unavailable · transcription continues",
      "budget-exhausted": "Jev request limit reached",
    }[status] || "Decisions unavailable";
    $("attention-counts").textContent = status === "off" ? "Enable Jev to test addressedness."
      : `${count.attend} attend · ${count.ignore} ignore · ${count.uncertain} uncertain${state.pending_decisions ? ` · ${state.pending_decisions} pending` : ""}`;
    if (state.error) message(state.error, true);
    else if (state.phase === "idle") message("Ready when you are. The microphone is off.");
    else if (state.phase === "starting") message(`${hostedSpeech(state) ? "Connecting to hosted speech service" : "Loading local models"}. If macOS asks for microphone access, approve only when ready to record this session.`);
    else if (state.phase === "complete") message("Audio demo finished. Review the retained text, or Stop to clear it.");
    else {
      const boundary = Number(retention.boundary_overlap_ms || 0);
      const notice = boundary > 0 ? " A turn crossing the retention boundary is kept whole." : "";
      message(`Recent text is retained locally for ${Math.round(Number(retention.retention_ms || 300000) / 1000)} seconds.${notice}`);
    }
    controls();
  }

  async function poll() {
    if (!authorized || closed || polling || busy) return;
    polling = true;
    const stamp = generation;
    try {
      // A dropped connection must stop the old session before renewing its lease.
      if (mustStop) {
        await api("/api/stop", {});
        if (stamp !== generation || closed) return;
        mustStop = false;
      }
      const state = await api("/api/state");
      if (stamp === generation && !closed && !busy) render(state);
    } catch {
      if (stamp === generation && !closed) disconnect();
    } finally {
      polling = false;
    }
  }

  async function start(mode) {
    if (!connected || busy || activePhases.has(phase)) return;
    const retention = Number($("retention-seconds").value);
    const confidence = Number($("confidence").value);
    const requests = Number($("max-requests").value);
    if (!Number.isInteger(retention) || retention < 60 || retention > 600
      || !Number.isFinite(confidence) || confidence < 0 || confidence > 1
      || !Number.isInteger(requests) || requests < 1 || requests > 100) {
      message("Choose valid retention, confidence, and request limits before starting.", true);
      return;
    }
    generation += 1;
    const stamp = generation;
    busy = true;
    clearContext();
    setPhase("starting");
    message(mode === "microphone" ? "Starting your explicitly enabled microphone session…" : "Starting the configured audio replay. Microphone stays off.");
    controls();
    try {
      await api("/api/start", {mode, use_jev: $("use-jev").checked,
        retention_seconds: retention, confidence, max_requests: requests});
      if (stamp === generation && !closed) {
        busy = false;
        await poll();
      }
    } catch {
      if (stamp === generation && !closed) disconnect();
    } finally {
      busy = false;
      controls();
    }
  }

  async function stop() {
    generation += 1;
    const stamp = generation;
    busy = true;
    clearContext();
    setPhase("stopping");
    message("Clearing context and stopping capture…");
    controls();
    try {
      await api("/api/stop", {});
      if (stamp === generation && !closed) {
        setPhase("idle");
        message("Session stopped. Retained context was cleared.");
      }
    } catch {
      if (stamp === generation && !closed) disconnect();
    } finally {
      busy = false;
      controls();
    }
  }

  $("start-button").addEventListener("click", () => start("microphone"));
  $("demo-button").addEventListener("click", () => start("demo"));
  $("stop-button").addEventListener("click", stop);
  $("use-jev").addEventListener("change", controls);
  $("retention-seconds").addEventListener("input", () => {
    const minutes = Number($("retention-seconds").value) / 60;
    $("retention-label").textContent = `${minutes} minute${minutes === 1 ? "" : "s"}`;
  });
  $("copy-button").addEventListener("click", async () => {
    if (!connected || !retainedTurns.length) return;
    const stamp = generation;
    const text = retainedTurns.map((turn) =>
      `[${timeLabel(turn.start_ms)}–${timeLabel(turn.end_ms)}] ${turn.speaker_id || "Speaker unknown"}${turn.overlap ? " (overlap)" : ""}: ${turn.text}`
    ).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      if (stamp === generation) message("Retained context copied. This clipboard copy remains until you replace it.");
    } catch {
      if (stamp === generation) message("Clipboard access was unavailable. Select transcript text to copy it manually.", true);
    }
  });
  window.addEventListener("offline", () => {
    disconnect();
    if (authorized && !closed) void api("/api/stop", {}, true).catch(() => {});
  });
  window.addEventListener("online", poll);
  window.addEventListener("pagehide", () => {
    closed = true;
    generation += 1;
    clearContext();
    if (authorized) void api("/api/stop", {}, true).catch(() => {});
  });
  if (!authorized) {
    clearContext();
    message("Open the private launch link printed by your local RightyO server. Refreshing this page clears its session access token.", true);
    controls();
  } else {
    void poll();
    setInterval(poll, 750);
  }
})();
