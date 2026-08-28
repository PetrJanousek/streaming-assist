(() => {
  const profileEl = document.getElementById("profile");
  const formEl = document.getElementById("composer");
  const inputEl = document.getElementById("message");
  const sendEl = document.getElementById("send");
  const transcriptEl = document.getElementById("transcript");
  const statusEl = document.getElementById("status");
  const constraintsEl = document.getElementById("constraints");
  const pillsEl = document.getElementById("constraint-pills");
  const stageItems = Array.from(document.querySelectorAll("#stages li"));

  const STAGE_ORDER = [
    "start",
    "constraints",
    "candidates",
    "validated",
    "picks",
    "reply",
    "done",
  ];

  const state = {
    sessionId: null,
    token: localStorage.getItem("assist.token") || "dev-adult",
    busy: false,
  };

  function setStatus(text, isError) {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("error", Boolean(isError));
  }

  function setBusy(busy) {
    state.busy = busy;
    sendEl.disabled = busy;
    inputEl.disabled = busy;
    document.querySelectorAll(".chip").forEach((btn) => {
      btn.disabled = busy;
    });
  }

  function markStage(stage) {
    const idx = STAGE_ORDER.indexOf(stage);
    stageItems.forEach((item) => {
      const itemStage = item.getAttribute("data-stage");
      const itemIdx = STAGE_ORDER.indexOf(itemStage);
      item.classList.toggle("active", itemStage === stage);
      item.classList.toggle("done", idx > itemIdx && itemIdx >= 0);
    });
  }

  function renderPills(constraints) {
    pillsEl.replaceChildren();
    if (!constraints || typeof constraints !== "object") {
      constraintsEl.hidden = true;
      return;
    }
    const labels = [];
    if (constraints.media_type) labels.push(String(constraints.media_type));
    for (const genre of constraints.genres_include || []) labels.push(String(genre));
    for (const mood of constraints.moods || []) labels.push(String(mood));
    if (constraints.year_min || constraints.year_max) {
      labels.push(`${constraints.year_min || "..."}-${constraints.year_max || "..."}`);
    }
    if (constraints.duration_max_min) {
      labels.push(`under ${constraints.duration_max_min} min`);
    }
    if (constraints.local_originals_only) labels.push("local originals");
    if (constraints.people_count) labels.push(`${constraints.people_count} people`);
    if (labels.length === 0) {
      constraintsEl.hidden = true;
      return;
    }
    constraintsEl.hidden = false;
    for (const label of labels) {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = label;
      pillsEl.appendChild(pill);
    }
  }

  function addUserBubble(text) {
    const wrap = document.createElement("article");
    wrap.className = "bubble user";
    wrap.innerHTML = `<p class="who">You</p><p class="reply"></p>`;
    wrap.querySelector(".reply").textContent = text;
    transcriptEl.appendChild(wrap);
    wrap.scrollIntoView({ block: "end" });
  }

  function addTurnCard() {
    const wrap = document.createElement("article");
    wrap.className = "bubble turn";
    wrap.innerHTML = `
      <p class="who">Assist</p>
      <p class="reply" data-role="reply">Working...</p>
      <div class="cards" data-role="cards"></div>
      <div class="chips" data-role="chips"></div>
    `;
    transcriptEl.appendChild(wrap);
    wrap.scrollIntoView({ block: "end" });
    return wrap;
  }

  function renderSkeletons(container, count) {
    container.replaceChildren();
    const n = Math.max(0, Math.min(Number(count) || 0, 8));
    for (let i = 0; i < n; i += 1) {
      const card = document.createElement("div");
      card.className = "card skeleton";
      card.setAttribute("aria-hidden", "true");
      container.appendChild(card);
    }
  }

  function renderCards(container, cards, { validated }) {
    container.replaceChildren();
    for (const card of cards || []) {
      const el = document.createElement("article");
      el.className = "card";
      const title = document.createElement("h3");
      title.textContent = card.title || card.reason_short || "Title";
      const meta = document.createElement("p");
      const bits = [];
      if (card.release_year) bits.push(String(card.release_year));
      if (card.media_type) bits.push(String(card.media_type));
      if (card.reason_short && card.reason_short !== card.title) {
        bits.push(String(card.reason_short));
      }
      meta.textContent = bits.join(" · ");
      if (validated && card.catalog_id) {
        el.dataset.catalogId = String(card.catalog_id);
      }
      el.appendChild(title);
      el.appendChild(meta);
      container.appendChild(el);
    }
  }

  function renderChips(container, chips, onTap) {
    container.replaceChildren();
    for (const chip of chips || []) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip";
      btn.textContent = chip.label;
      btn.dataset.chipId = chip.id;
      btn.addEventListener("click", () => onTap(chip));
      container.appendChild(btn);
    }
  }

  async function loadProfiles() {
    try {
      const res = await fetch("/dev/profiles");
      const body = await res.json();
      const profiles = body.profiles || [];
      profileEl.replaceChildren();
      for (const row of profiles) {
        const opt = document.createElement("option");
        opt.value = row.token;
        opt.textContent = row.label || row.token;
        profileEl.appendChild(opt);
      }
      if (![...profileEl.options].some((o) => o.value === state.token)) {
        state.token = profiles[0] ? profiles[0].token : "dev-adult";
      }
      profileEl.value = state.token;
    } catch (_err) {
      profileEl.replaceChildren();
      const opt = document.createElement("option");
      opt.value = "dev-adult";
      opt.textContent = "dev-adult";
      profileEl.appendChild(opt);
    }
  }

  function parseSseChunk(buffer, onEvent) {
    let rest = buffer;
    let idx = rest.indexOf("\n\n");
    while (idx !== -1) {
      const raw = rest.slice(0, idx);
      rest = rest.slice(idx + 2);
      let event = "message";
      const dataLines = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length) {
        try {
          onEvent(event, JSON.parse(dataLines.join("\n")));
        } catch (_err) {
          setStatus("Bad stream frame.", true);
        }
      }
      idx = rest.indexOf("\n\n");
    }
    return rest;
  }

  async function sendTurn(message) {
    if (state.busy) return;
    const label =
      message.type === "chip" ? message.label || "chip" : (message.text || "").trim();
    if (!label) return;
    addUserBubble(label);
    const turnEl = addTurnCard();
    const replyEl = turnEl.querySelector('[data-role="reply"]');
    const cardsEl = turnEl.querySelector('[data-role="cards"]');
    const chipsEl = turnEl.querySelector('[data-role="chips"]');

    setBusy(true);
    setStatus("Streaming...");
    markStage("start");

    const body = {
      session_id: state.sessionId,
      message:
        message.type === "chip"
          ? { type: "chip", chip_id: message.chip_id }
          : { type: "text", text: message.text },
    };

    let res;
    try {
      res = await fetch("/v1/assist/turn/stream", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${state.token}`,
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
      });
    } catch (_err) {
      setBusy(false);
      setStatus("Network error.", true);
      replyEl.textContent = "The assist service is unreachable.";
      return;
    }

    const ctype = res.headers.get("content-type") || "";
    if (!res.ok || !ctype.includes("text/event-stream")) {
      let detail = `HTTP ${res.status}`;
      try {
        const err = await res.json();
        detail = (err.error && err.error.message) || detail;
      } catch (_err) {
        /* keep status text */
      }
      setBusy(false);
      setStatus(detail, true);
      replyEl.textContent = detail;
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const onChip = (chip) => {
      sendTurn({ type: "chip", chip_id: chip.id, label: chip.label });
    };

    const onEvent = (_event, data) => {
      const stage = data.stage || _event;
      markStage(stage);
      if (data.session_id) state.sessionId = data.session_id;
      if (stage === "constraints") renderPills(data.constraints);
      if (stage === "candidates") renderSkeletons(cardsEl, data.count || 4);
      if (stage === "validated") renderCards(cardsEl, data.cards || [], { validated: false });
      if (stage === "picks") renderCards(cardsEl, data.picks || [], { validated: true });
      if (stage === "reply") {
        replyEl.textContent = data.reply || "";
        renderChips(chipsEl, data.chips || [], onChip);
      }
      if (stage === "done") {
        if (data.session_id) state.sessionId = data.session_id;
        if (data.reply) replyEl.textContent = data.reply;
        if (data.chips && data.chips.length) renderChips(chipsEl, data.chips, onChip);
        if (data.picks && data.picks.length && !cardsEl.querySelector(".card:not(.skeleton)")) {
          renderCards(cardsEl, data.picks, { validated: true });
        }
        if (data.meta && data.meta.degraded) {
          setStatus(data.meta.degraded_reason || "degraded", true);
        } else {
          setStatus("");
        }
      }
    };

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        buffer = parseSseChunk(buffer, onEvent);
      }
      if (buffer.trim()) parseSseChunk(`${buffer}\n\n`, onEvent);
    } catch (_err) {
      setStatus("Stream dropped.", true);
    } finally {
      setBusy(false);
      markStage("done");
    }
  }

  profileEl.addEventListener("change", () => {
    state.token = profileEl.value;
    state.sessionId = null;
    localStorage.setItem("assist.token", state.token);
    setStatus("Profile switched. New session on next search.");
  });

  formEl.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = "";
    sendTurn({ type: "text", text });
  });

  loadProfiles();
  inputEl.focus();
})();
