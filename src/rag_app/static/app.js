// RAG Chat UI — vanilla JS, no build step. Talks to the FastAPI backend
// served alongside this page. Markdown/escaping helpers live in markdown.js.
(() => {
  const { escapeHtml: esc, renderMarkdown, cleanPassage } = window;

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  let uid = 0;
  const nextId = () => "m" + (++uid);
  const state = {
    chats: [],            // [{ id, title, updated (ms) }]
    docs: [],             // [{ id, filename, type, at, chunks }]
    current: null,        // open chat id, a "tmp-n" key for a new chat still waiting for its id, or null
    messages: {},         // chat key -> [message]
    // Requests still streaming, keyed by chat key. Per chat (not one global
    // flag) so switching to another chat while one streams doesn't block
    // asking there; switching back shows the live message, since it lives
    // in state.messages and is re-rendered from there.
    inFlight: new Map(),
    pendingDeletes: new Map(), // chat id -> timer (undo window)
    scope: null,
    topK: 5,
    sel: null,            // { mid, n } source open in the inspector
    tab: "src",
    uploads: [],
    query: "",
    health: null,
  };
  try { state.topK = Math.min(20, Math.max(1, Number(localStorage.getItem("rag-topk")) || 5)); } catch {}

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  const $ = (s, el = document) => el.querySelector(s);
  const app = $("#app"), thread = $("#thread"), scroller = $("#scroller");
  const ta = $("#composer-input"), sendBtn = $("#send"), live = $("#live");
  const icon = (name, cls = "ic") => `<svg class="${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const fmt = (n) => Number(n || 0).toLocaleString("en-US");
  const plural = (n, word) => `${fmt(n)} ${word}${n === 1 ? "" : "s"}`;
  const announce = (msg) => { live.textContent = ""; setTimeout(() => (live.textContent = msg), 30); };
  const docByName = (f) => state.docs.find((d) => d.filename === f);
  const totalChunks = () => state.docs.reduce((a, d) => a + d.chunks, 0);
  const isTmp = (key) => typeof key === "string" && key.startsWith("tmp-");
  const layout = () => app.dataset.layout;
  const isOverlayPanel = () => layout() !== "wide";
  function ago(ms) {
    const s = Math.max(0, (Date.now() - ms) / 1000);
    if (s < 60) return "just now";
    if (s < 3600) return Math.floor(s / 60) + " min ago";
    if (s < 86400) return Math.floor(s / 3600) + " h ago";
    const d = Math.floor(s / 86400);
    return d === 1 ? "yesterday" : d + " days ago";
  }
  // Each document keeps one highlighter hue everywhere, assigned in upload
  // order so existing documents never change colour when a new one arrives.
  function docColor(filename) {
    const byAge = [...state.docs].sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
    const i = byAge.findIndex((d) => d.filename === filename);
    return i < 0 ? "--c:var(--border-strong)" : `--c:var(--doc-${(i % 8) + 1})`;
  }
  const md = (text, m) => renderMarkdown(text, { msgId: m.id, count: (m.sources || []).length, colorFor: (n) => docColor(m.sources[n - 1].filename) });
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); } catch {
      const t = document.createElement("textarea"); t.value = text; document.body.appendChild(t); t.select();
      try { document.execCommand("copy"); } catch {} t.remove();
    }
  }

  // ---------------------------------------------------------------------------
  // Layout: one source of truth for drawer/panel behaviour
  // ---------------------------------------------------------------------------
  function syncInert() {
    const L = layout();
    const drawer = app.classList.contains("drawer-open");
    const panel = app.classList.contains("panel-open");
    $("#sidebar").inert = L === "narrow" && !drawer;
    $("#panel").inert = !panel;
    $("#main").inert = (L === "narrow" && drawer) || (L !== "wide" && panel);
    $("#open-nav").setAttribute("aria-expanded", String(drawer));
    $("#docs-btn").setAttribute("aria-expanded", String(panel && state.tab === "docs"));
  }
  new ResizeObserver(() => {
    const w = app.clientWidth;
    const L = w >= 1100 ? "wide" : w >= 760 ? "medium" : "narrow";
    if (L === app.dataset.layout) return;
    const prev = app.dataset.layout;
    app.dataset.layout = L;
    if (L !== "narrow") app.classList.remove("drawer-open");
    if (prev === "wide") app.classList.remove("panel-open");
    closePops();
    syncInert();
  }).observe(app);

  function openDrawer() { app.classList.add("drawer-open"); syncInert(); setTimeout(() => $("#new-chat").focus(), 60); }
  function closeDrawer() { app.classList.remove("drawer-open"); syncInert(); }

  let panelReturn = null;
  function openPanel(tab, trigger) {
    if (trigger) panelReturn = trigger;
    setTab(tab);
    app.classList.add("panel-open");
    if (layout() === "narrow") closeDrawer();
    syncInert();
    if (isOverlayPanel()) setTimeout(() => $(tab === "src" ? "#tab-src" : "#tab-docs").focus(), 60);
  }
  function closePanel() {
    app.classList.remove("panel-open");
    syncInert();
    markActive(null);
    if (panelReturn && document.contains(panelReturn)) panelReturn.focus();
    panelReturn = null;
  }
  const panelShows = (tab) => app.classList.contains("panel-open") && state.tab === tab;
  function setTab(tab) {
    state.tab = tab;
    $("#tab-src").setAttribute("aria-selected", String(tab === "src"));
    $("#tab-docs").setAttribute("aria-selected", String(tab === "docs"));
    $("#tab-src").tabIndex = tab === "src" ? 0 : -1;
    $("#tab-docs").tabIndex = tab === "docs" ? 0 : -1;
    $("#pane-src").hidden = tab !== "src";
    $("#pane-docs").hidden = tab !== "docs";
    if (tab === "src") renderInspector(); else renderDocs();
    syncInert();
  }

  // ---------------------------------------------------------------------------
  // Deep links: #/c/<chat id>
  // ---------------------------------------------------------------------------
  function setHash(id, replace = false) {
    const h = id ? `#/c/${encodeURIComponent(id)}` : "#/";
    if (location.hash === h || (!id && !location.hash)) return;
    history[replace ? "replaceState" : "pushState"](null, "", h);
  }
  function route() {
    const m = location.hash.match(/^#\/c\/(.+)$/);
    if (m) openChat(decodeURIComponent(m[1]), { fromHistory: true });
    else newChat({ fromHistory: true, focus: false });
  }
  window.addEventListener("popstate", route);

  // ---------------------------------------------------------------------------
  // Sidebar: chats
  // ---------------------------------------------------------------------------
  function groupOf(t) {
    const today = new Date(); today.setHours(0, 0, 0, 0);
    const day = 86400000, t0 = today.getTime();
    if (t >= t0) return "Today";
    if (t >= t0 - day) return "Yesterday";
    if (t >= t0 - 7 * day) return "Previous 7 days";
    if (t >= t0 - 30 * day) return "Previous 30 days";
    return "Older";
  }
  async function loadChats() {
    try {
      const res = await fetch("/chats");
      if (!res.ok) return;
      const data = await res.json();
      state.chats = (data.chats || [])
        .filter((c) => !state.pendingDeletes.has(c.id))
        .map((c) => ({ id: c.id, title: c.title, updated: parseFloat(c.updated_at) * 1000 || 0 }));
      renderChats();
      renderTitle();
    } catch {
      // Keep the last-known list; the status pill already reports connectivity.
    }
  }
  function renderChats() {
    const q = state.query.trim().toLowerCase();
    const list = state.chats.filter((c) => !q || c.title.toLowerCase().includes(q));
    const el = $("#chat-groups");
    if (!list.length) {
      el.innerHTML = q ? `<p class="list-empty">No chats match “${esc(state.query.trim())}”.</p>` : `<p class="list-empty">No chats yet. Ask a question to start one.</p>`;
      return;
    }
    const groups = new Map();
    list.forEach((c) => { const g = groupOf(c.updated); if (!groups.has(g)) groups.set(g, []); groups.get(g).push(c); });
    const hi = (t) => {
      if (!q) return esc(t);
      const i = t.toLowerCase().indexOf(q);
      return esc(t.slice(0, i)) + "<mark>" + esc(t.slice(i, i + q.length)) + "</mark>" + esc(t.slice(i + q.length));
    };
    el.innerHTML = [...groups].map(([g, cs], gi) => `
      <section class="group" aria-labelledby="g${gi}">
        <h2 id="g${gi}">${g}</h2>
        <ul class="chat-list">${cs.map((c) => `
          <li class="chat-item">
            <button type="button" class="chat-link" data-id="${esc(c.id)}" title="${esc(c.title)}" ${c.id === state.current ? 'aria-current="page"' : ""}>${hi(c.title)}</button>
            <button type="button" class="icon-btn del" data-del="${esc(c.id)}" aria-label="Delete chat “${esc(c.title)}”">${icon("trash", "ic ic-sm")}</button>
          </li>`).join("")}
        </ul>
      </section>`).join("");
  }
  function renderTitle() {
    const chat = state.chats.find((c) => c.id === state.current);
    $("#chat-title").textContent = chat ? chat.title : "New chat";
  }
  $("#chat-groups").addEventListener("click", (e) => {
    const del = e.target.closest("[data-del]");
    if (del) return deleteChat(del.dataset.del);
    const link = e.target.closest(".chat-link");
    if (link) openChat(link.dataset.id);
  });
  $("#chat-search").addEventListener("input", (e) => { state.query = e.target.value; renderChats(); });

  // Delete is deferred for an undo window; the DELETE request goes out after 5s
  // (or immediately if the page is closed before then).
  function deleteChat(id) {
    const idx = state.chats.findIndex((c) => c.id === id);
    if (idx < 0) return;
    const [chat] = state.chats.splice(idx, 1);
    const wasCurrent = state.current === id;
    if (wasCurrent) newChat({ focus: false });
    renderChats();
    announce(`Deleted ${chat.title}`);
    const timer = setTimeout(() => commitDelete(id), 5000);
    state.pendingDeletes.set(id, timer);
    toast(`Deleted “${chat.title}”`, {
      action: "Undo", duration: 5000,
      onAction() {
        clearTimeout(timer);
        state.pendingDeletes.delete(id);
        state.chats.splice(Math.min(idx, state.chats.length), 0, chat);
        renderChats();
        if (wasCurrent) openChat(id);
        announce(`Restored ${chat.title}`);
      },
    });
  }
  async function commitDelete(id) {
    state.pendingDeletes.delete(id);
    // A stream still running for this chat has nowhere to be saved any more.
    state.inFlight.get(id)?.controller.abort();
    delete state.messages[id];
    try {
      const res = await fetch(`/chats/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!res.ok && res.status !== 404) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      console.error("deleteChat failed:", err);
      toast("Couldn't delete that chat on the server. It will reappear after a refresh.");
      loadChats();
    }
  }
  window.addEventListener("pagehide", () => {
    for (const [id, timer] of state.pendingDeletes) {
      clearTimeout(timer);
      fetch(`/chats/${encodeURIComponent(id)}`, { method: "DELETE", keepalive: true });
    }
  });

  let openSeq = 0;
  async function openChat(id, { fromHistory = false } = {}) {
    if (!fromHistory) setHash(id);
    if (layout() === "narrow") closeDrawer();
    if (id === state.current && state.messages[id]) return;
    const seq = ++openSeq;
    state.current = id;
    renderTitle();
    renderChats();
    refreshComposer();
    // A chat that's still streaming is shown from local state: the server
    // only saves the answer once the stream finishes.
    if (state.inFlight.has(id)) { renderThread(); scroller.scrollTop = scroller.scrollHeight; return; }
    if (state.messages[id]) renderThread();
    else {
      thread.classList.remove("is-empty");
      thread.innerHTML = `<div class="skeleton" aria-busy="true" aria-label="Loading chat"><span style="width:38%;align-self:flex-end"></span><span style="width:92%"></span><span style="width:84%"></span><span style="width:60%"></span></div>`;
    }
    try {
      const res = await fetch(`/chats/${encodeURIComponent(id)}`);
      if (seq !== openSeq) return;
      if (res.status === 404) {
        toast("That chat no longer exists.");
        newChat({ focus: false });
        setHash(null, true);
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (seq !== openSeq || state.inFlight.has(id)) return; // superseded, or a send started meanwhile
      state.messages[id] = fromServer(data.messages || []);
      if (!state.chats.some((c) => c.id === id)) state.chats.unshift({ id, title: data.title, updated: Date.now() });
      renderTitle();
      renderThread();
      scroller.scrollTop = scroller.scrollHeight;
    } catch (err) {
      console.error("openChat failed:", err);
      if (seq !== openSeq) return;
      thread.classList.remove("is-empty");
      thread.innerHTML = `<div class="err" role="alert">${icon("alert")}<div><h3>Couldn't load this chat</h3><p>The server didn't respond. Check that the API is running, then try again.</p><div class="row"><button type="button" class="btn" data-reload="${esc(id)}">${icon("retry", "ic ic-sm")}Try again</button></div></div></div>`;
    }
  }
  thread.addEventListener("click", (e) => {
    const b = e.target.closest("[data-reload]");
    if (b) { delete state.messages[b.dataset.reload]; state.current = null; openChat(b.dataset.reload, { fromHistory: true }); }
  });

  function fromServer(list) {
    let question = "";
    return list.map((x) => {
      if (x.role === "user") { question = x.content; return { id: nextId(), role: "user", text: x.content }; }
      return {
        id: nextId(), role: "ai", status: "done", question, md: x.content, shown: x.content,
        sources: x.sources || [], provider: x.llm_provider || "", model: x.llm_model || "", stored: true,
      };
    });
  }

  function newChat({ focus = true, fromHistory = false } = {}) {
    ++openSeq;
    state.current = null;
    if (!fromHistory) setHash(null);
    renderTitle();
    renderChats();
    renderThread();
    refreshComposer();
    if (layout() === "narrow") closeDrawer();
    if (focus) ta.focus();
  }

  // ---------------------------------------------------------------------------
  // Thread
  // ---------------------------------------------------------------------------
  function findMsg(mid) {
    for (const k in state.messages) { const m = state.messages[k].find((x) => x.id === mid); if (m) return m; }
    return null;
  }
  function renderThread() {
    const msgs = state.current ? state.messages[state.current] || [] : [];
    hideTip();
    thread.innerHTML = "";
    thread.classList.toggle("is-empty", !msgs.length);
    if (!msgs.length) return renderEmpty();
    msgs.forEach((m) => thread.appendChild(m.role === "user" ? userEl(m) : aiEl(m)));
    if (state.sel) markActive(state.sel);
  }
  const prettyName = (f) => f.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
  function suggestions() {
    const templates = [
      (n) => `What are the key points of ${n}?`,
      (n) => `Summarize ${n}`,
      (n) => `What does ${n} cover?`,
      (n) => `What are the main conclusions of ${n}?`,
    ];
    return [...state.docs].sort((a, b) => Date.parse(b.at) - Date.parse(a.at)).slice(0, 4)
      .map((d, i) => ({ q: templates[i](prettyName(d.filename)), doc: d }));
  }
  function renderEmpty() {
    const docs = state.docs;
    const latest = docs.reduce((a, d) => (!a || Date.parse(d.at) > Date.parse(a.at) ? d : a), null);
    const sugg = suggestions();
    thread.innerHTML = `
      <section class="empty" aria-labelledby="empty-h">
        <div class="empty-head">
          <h2 id="empty-h">Chat with your documents</h2>
          <p>Ask a question in plain language. RAG Chat finds the relevant passages in your files and writes an answer that cites each one, so you can check it.</p>
        </div>
        <ol class="how" aria-label="How answers are made">
          <li><span class="how-n">1</span><div><strong>Search</strong><span>Finds matching passages across your ${docs.length ? plural(docs.length, "document") : "documents"}, by meaning and by keyword.</span></div></li>
          <li><span class="how-n">2</span><div><strong>Rerank</strong><span>A cross-encoder scores the candidates and keeps the best ${state.topK}.</span></div></li>
          <li><span class="how-n">3</span><div><strong>Answer</strong><span>The model writes from those passages and cites each one, like this <span class="cite" style="${latest ? docColor(latest.filename) : ""};cursor:default">1</span></span></div></li>
        </ol>
        ${docs.length ? `
        <div class="kb-line">${icon("library", "ic ic-sm")}<span>Searching <span class="mono">${docs.length}</span> ${docs.length === 1 ? "document" : "documents"} · <span class="mono">${fmt(totalChunks())}</span> passages · last added ${ago(Date.parse(latest.at))}</span><button type="button" class="linkish" data-docs>Manage documents</button></div>
        <div class="try-label">Try asking</div>
        <div class="suggest">${sugg.map((s, i) => `
          <button type="button" class="sugg" data-sugg="${i}"><span class="q">${esc(s.q)}</span><span class="doc" style="${docColor(s.doc.filename)}"><span class="doc-sw" aria-hidden="true"></span><span>${esc(s.doc.filename)}</span></span></button>`).join("")}
        </div>` : `
        <label class="dropzone" for="file-input">${icon("upload")}<span><strong>Add your first document</strong> · drop a file here or <span class="u">browse</span></span><small>PDF, TXT or MD · up to 20 MB each</small></label>`}
      </section>`;
  }
  thread.addEventListener("click", (e) => {
    const s = e.target.closest("[data-sugg]");
    if (s) { const sg = suggestions()[+s.dataset.sugg]; if (sg) ask(sg.q, { scope: sg.doc }); return; }
    if (e.target.closest("[data-docs]")) openPanel("docs", e.target.closest("[data-docs]"));
  });

  function userEl(m) {
    const el = document.createElement("div");
    el.className = "msg-user";
    el.textContent = m.text;
    return el;
  }
  function aiEl(m) {
    const el = document.createElement("article");
    el.className = "msg-ai";
    el.dataset.mid = m.id;
    el.setAttribute("aria-label", "Answer from the assistant");
    el.innerHTML = `<div class="ai-head"><span class="ai-mark">${icon("mark")}</span>Assistant</div><div class="ai-status"></div><div class="prose"></div><div class="ai-sources"></div><div class="msg-actions"></div>`;
    paintAi(el, m);
    return el;
  }
  const elFor = (mid) => thread.querySelector(`.msg-ai[data-mid="${mid}"]`);

  function paintAi(el, m) {
    const live = m.status === "streaming";
    el.setAttribute("aria-busy", String(live));

    // Retrieval pipeline: live progress while streaming, then kept as a receipt of how the answer was made.
    const st = $(".ai-status", el);
    const n = (m.sources || []).length;
    const files = new Set((m.sources || []).map((x) => x.filename)).size;
    const answered = n ? `Answered from ${plural(n, "passage")} in ${plural(files, "document")}` : "No relevant passages found";
    if (m.status === "error") st.innerHTML = "";
    else if (m.cached && !live) {
      st.innerHTML = `<ol class="steps receipt" aria-label="How this answer was made"><li class="step done">${icon("check", "ic ic-sm")}Reused the saved answer to this question</li><li class="step">${n ? `${plural(n, "passage")} in ${plural(files, "document")}` : "No passages"}</li></ol>`;
    } else {
      const order = ["search", "rerank", "write"];
      const at = live || m.status === "stopped" ? order.indexOf(m.stage) : 3;
      const where = m.where || "your documents";
      const cand = m.cand ? plural(m.cand, "passage") : "passages";
      const labels = [
        [`Searching ${where}`, `Searched ${where}`],
        m.reranked === false ? null : [`Reranking ${cand}`, `Reranked ${cand}`],
        ["Writing answer", answered],
      ];
      const steps = labels.map((l, i) => {
        if (!l || (!live && i >= at)) return "";
        const cls = i < at ? "done" : i === at ? "active" : "";
        const mark = i < at ? icon("check", "ic ic-sm") : i === at ? '<span class="spin" aria-hidden="true"></span>' : '<span class="pending-dot" aria-hidden="true"></span>';
        return `<li class="step ${cls}">${mark}${i < at ? l[1] : l[0]}</li>`;
      }).join("");
      st.innerHTML = steps ? `<ol class="steps ${live ? "" : "receipt"}" aria-label="${live ? "Progress" : "How this answer was made"}">${steps}</ol>` : "";
    }

    const prose = $(".prose", el);
    if (m.status === "error") prose.innerHTML = errorHtml(m);
    else if (m.shown) prose.innerHTML = md(m.shown, m) + (live ? '<span class="cursor" aria-hidden="true"></span>' : "");
    else if (m.status === "stopped") prose.innerHTML = '<p class="stopped-note">Stopped before the answer started.</p>';
    else prose.innerHTML = "";

    // Sources arrive before the text, so citations work while it streams.
    const src = $(".ai-sources", el);
    if (n && m.status !== "error") {
      const key = m.sources.map((s) => s.filename).join("|");
      if (src.dataset.key !== key) {
        src.dataset.key = key;
        const groups = new Map();
        m.sources.forEach((s, i) => { if (!groups.has(s.filename)) groups.set(s.filename, []); groups.get(s.filename).push(i + 1); });
        src.innerHTML = `<div class="src-label">Sources</div><div class="src-row">${[...groups].map(([f, nums]) => `
          <div class="src-file" style="${docColor(f)}"><span class="doc-sw" aria-hidden="true"></span><span class="name" title="${esc(f)}">${esc(f || "Unknown file")}</span><span class="nums">${nums.map((k) => `<button type="button" class="src-num" style="${docColor(f)}" data-mid="${m.id}" data-n="${k}" aria-label="Source ${k}, ${esc(f)}">${k}</button>`).join("")}</span></div>`).join("")}</div>`;
      }
    } else { src.innerHTML = ""; delete src.dataset.key; }

    const acts = $(".msg-actions", el);
    if (m.status === "done" || m.status === "stopped") {
      const meta = [m.provider, m.model, m.latency ? (m.latency / 1000).toFixed(1) + " s" : null].filter(Boolean).map(esc).join(" · ");
      acts.innerHTML = `
        ${m.shown ? `<button type="button" class="act" data-copy="${m.id}">${icon("copy", "ic ic-sm")}<span>Copy</span></button>` : ""}
        ${n ? `<button type="button" class="act" data-allsrc="${m.id}">${icon("library", "ic ic-sm")}<span>${plural(n, "source")}</span></button>` : ""}
        ${m.status === "stopped" ? '<span class="tag">Stopped</span>' : ""}
        ${m.cached ? '<span class="tag accent">Cached answer</span>' : ""}
        <span class="meta">${meta}</span>`;
    } else acts.innerHTML = "";
    if (state.sel && state.sel.mid === m.id) markActive(state.sel);
  }

  const ERRORS = {
    rate: ["Too many questions at once", "The server accepts 20 questions per minute. Wait a few seconds, then try again."],
    network: ["Can't reach the server", "Check that the API is running, then try again."],
    server: ["The server couldn't answer", "It returned an error. Try again; if it keeps failing, check the API logs."],
    llm: ["The language model didn't answer", "The model provider returned an error. Try again in a moment."],
    incomplete: ["The answer was cut off", "The connection closed before the answer finished. Try again."],
  };
  function errorHtml(m) {
    const [title, body] = ERRORS[m.error] || ERRORS.server;
    const detail = m.errorText ? `<p class="mono" style="margin-top:6px;font-size:12px">${esc(m.errorText.slice(0, 300))}</p>` : "";
    return `<div class="err" role="alert">${icon("alert")}<div><h3>${title}</h3><p>${body}</p>${detail}<div class="row">
      <button type="button" class="btn" data-retry="${m.id}">${icon("retry", "ic ic-sm")}Try again</button>
      <button type="button" class="btn" data-edit="${m.id}">${icon("edit", "ic ic-sm")}Edit question</button></div></div></div>`;
  }

  thread.addEventListener("click", async (e) => {
    const copy = e.target.closest("[data-copy]");
    if (copy) {
      const m = findMsg(copy.dataset.copy);
      await copyText(m.md.replace(/\*\*|`/g, ""));
      copy.innerHTML = `${icon("check", "ic ic-sm")}<span>Copied</span>`;
      announce("Answer copied");
      setTimeout(() => (copy.innerHTML = `${icon("copy", "ic ic-sm")}<span>Copy</span>`), 1600);
      return;
    }
    const codeCopy = e.target.closest(".code-copy");
    if (codeCopy) {
      await copyText(codeCopy.closest(".code-block").querySelector("code").textContent);
      codeCopy.textContent = "Copied";
      setTimeout(() => (codeCopy.textContent = "Copy"), 1600);
      return;
    }
    const all = e.target.closest("[data-allsrc]");
    if (all) return openSource(all.dataset.allsrc, 1, all);
    const cite = e.target.closest(".cite[data-mid], .src-num[data-mid]");
    if (cite) return openSource(cite.dataset.mid, +cite.dataset.n, cite);
    const retry = e.target.closest("[data-retry]");
    if (retry) return retryMsg(retry.dataset.retry);
    const edit = e.target.closest("[data-edit]");
    if (edit) return editMsg(edit.dataset.edit);
  });

  // ---------------------------------------------------------------------------
  // Asking: POST /chat/stream, Server-Sent Events
  // ---------------------------------------------------------------------------
  function bumpChat(id) {
    const c = state.chats.find((x) => x.id === id);
    if (!c) return;
    c.updated = Date.now();
    state.chats.splice(state.chats.indexOf(c), 1);
    state.chats.unshift(c);
    renderChats();
  }
  // A new chat has no id until the server's "chat" event; it runs under a
  // temporary key that is swapped for the real id everywhere once known.
  function rekey(tmp, id) {
    state.messages[id] = state.messages[tmp];
    delete state.messages[tmp];
    const run = state.inFlight.get(tmp);
    state.inFlight.delete(tmp);
    if (run) { run.key = id; state.inFlight.set(id, run); }
    if (state.current === tmp) { state.current = id; setHash(id, true); }
  }

  function ask(text, { scope } = {}) {
    const q = text.trim();
    if (!q) return;
    if (!state.current) {
      state.current = "tmp-" + (++uid);
      state.messages[state.current] = [];
      renderTitle();
    }
    const key = state.current;
    if (state.inFlight.has(key)) return;
    state.messages[key].push({ id: nextId(), role: "user", text: q });
    if (!isTmp(key)) bumpChat(key);
    renderThread();
    stream(key, q, scope === undefined ? state.scope : scope);
  }

  async function stream(key, q, scope) {
    const m = {
      id: nextId(), role: "ai", status: "streaming", stage: "search", question: q, scope,
      where: scope ? scope.filename : plural(state.docs.length, "document"),
      cand: Math.min(state.topK * 10, 50), md: "", shown: "", sources: [],
    };
    state.messages[key].push(m);
    if (state.current === key) { thread.classList.remove("is-empty"); thread.appendChild(aiEl(m)); stick(true); }
    const run = { key, m, controller: new AbortController(), t0: performance.now() };
    state.inFlight.set(key, run);
    refreshComposer();

    // Repaint at most once per frame; only when this chat is the one on screen.
    let queued = false;
    const repaint = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        const el = elFor(m.id);
        if (!el) return;
        const near = nearBottom();
        paintAi(el, m);
        stick(near);
        if (state.sel && state.sel.mid === m.id && panelShows("src")) renderInspector();
      });
    };

    try {
      const res = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: state.topK, chat_id: isTmp(run.key) ? null : run.key, filters: scope ? { source_id: scope.id } : null }),
        signal: run.controller.signal,
      });
      if (res.status === 429) throw Object.assign(new Error("Rate limited"), { kind: "rate" });
      if (!res.ok || !res.body) throw Object.assign(new Error(`HTTP ${res.status}`), { kind: "server" });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop(); // keep the last (possibly incomplete) frame
        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith("data:")) continue;
          const p = JSON.parse(line.slice(5).trim());
          if (p.type === "chat") {
            if (isTmp(run.key)) rekey(run.key, p.chat_id);
            loadChats();
          } else if (p.type === "status") {
            m.stage = { retrieving: "search", reranking: "rerank", generating: "write" }[p.stage] || m.stage;
            if (p.stage === "reranking") { m.reranked = true; if (p.candidates) m.cand = p.candidates; }
            if (p.stage === "generating" && !m.reranked) m.reranked = false;
          } else if (p.type === "sources") {
            m.sources = p.sources || [];
          } else if (p.type === "token") {
            m.stage = "write";
            m.md += p.content;
            m.shown = m.md;
          } else if (p.type === "answer") {
            m.stage = "write";
            m.md = m.shown = p.content;
          } else if (p.type === "done") {
            Object.assign(m, {
              status: "done", sources: p.sources || m.sources, cached: !!p.cached,
              provider: p.cached ? "" : p.llm_provider, model: p.llm_model, latency: p.latency_ms,
            });
            announce("Answer ready");
          } else if (p.type === "error") {
            Object.assign(m, { status: "error", error: "llm", errorText: p.message || "" });
          }
          repaint();
        }
      }
      if (m.status === "streaming") Object.assign(m, { status: "error", error: "incomplete" });
    } catch (err) {
      if (err.name === "AbortError") {
        m.status = "stopped";
        m.latency = Math.round(performance.now() - run.t0);
        announce("Stopped");
      } else {
        Object.assign(m, { status: "error", error: err.kind || "network" });
        announce("The question failed");
      }
    } finally {
      state.inFlight.delete(run.key);
      repaint();
      refreshComposer();
      if (!isTmp(run.key)) loadChats();
    }
  }
  function stopCurrent() {
    const run = state.current && state.inFlight.get(state.current);
    if (run) run.controller.abort();
    return !!run;
  }
  function retryMsg(mid) {
    const list = state.messages[state.current];
    const idx = list ? list.findIndex((x) => x.id === mid) : -1;
    if (idx < 0 || state.inFlight.has(state.current)) return;
    const m = list[idx];
    list.splice(idx, 1);
    elFor(mid)?.remove();
    stream(state.current, m.question, m.scope);
  }
  function editMsg(mid) {
    const list = state.messages[state.current];
    const idx = list ? list.findIndex((x) => x.id === mid) : -1;
    if (idx < 0) return;
    const q = list[idx].question;
    list.splice(Math.max(0, idx - 1), 2);
    renderThread();
    ta.value = q; autosize(); refreshComposer(); ta.focus();
  }

  // ---------------------------------------------------------------------------
  // Scrolling: follow only when the reader is already at the bottom
  // ---------------------------------------------------------------------------
  const nearBottom = () => scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
  function stick(near) {
    if (near) { scroller.scrollTop = scroller.scrollHeight; $("#jump").hidden = true; }
    else if (state.inFlight.has(state.current)) $("#jump").hidden = false;
  }
  scroller.addEventListener("scroll", () => { if (nearBottom()) $("#jump").hidden = true; });
  $("#jump").addEventListener("click", () => { scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" }); $("#jump").hidden = true; ta.focus(); });

  // ---------------------------------------------------------------------------
  // Composer
  // ---------------------------------------------------------------------------
  function autosize() {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
    ta.style.overflowY = ta.scrollHeight > 200 ? "auto" : "hidden";
  }
  function refreshComposer() {
    const busy = !!state.current && state.inFlight.has(state.current);
    sendBtn.classList.toggle("stop", busy);
    sendBtn.type = busy ? "button" : "submit";
    sendBtn.disabled = busy ? false : !ta.value.trim();
    sendBtn.setAttribute("aria-label", busy ? "Stop generating" : "Send question");
    $("#send-icon").setAttribute("href", busy ? "#i-stop" : "#i-up");
    const n = ta.value.length, c = $("#counter");
    c.hidden = n < 1800;
    c.textContent = `${fmt(n)} / 2,000`;
    c.classList.toggle("over", n >= 1950);
    if (!busy) $("#jump").hidden = true;
  }
  ta.addEventListener("input", () => { autosize(); refreshComposer(); });
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.current && state.inFlight.has(state.current)) return;
    const q = ta.value;
    if (!q.trim()) return;
    ta.value = ""; autosize(); refreshComposer();
    ask(q);
  });
  sendBtn.addEventListener("click", (e) => { if (stopCurrent()) { e.preventDefault(); ta.focus(); } });

  // ---------------------------------------------------------------------------
  // Popovers: scope, passages, status
  // ---------------------------------------------------------------------------
  let openPop = null;
  function closePops() {
    if (!openPop) return false;
    const { el, anchor } = openPop;
    el.remove();
    anchor.setAttribute("aria-expanded", "false");
    openPop = null;
    return true;
  }
  function showPop(anchor, html, { role = "dialog", label = "", above = true } = {}) {
    closePops();
    const el = document.createElement("div");
    el.className = "pop";
    el.setAttribute("role", role);
    if (label) el.setAttribute("aria-label", label);
    el.innerHTML = html;
    app.appendChild(el);
    const a = anchor.getBoundingClientRect(), r = app.getBoundingClientRect();
    el.style.left = Math.max(12, Math.min(a.left - r.left, r.width - el.offsetWidth - 12)) + "px";
    if (above) el.style.bottom = (r.bottom - a.top + 8) + "px";
    else el.style.top = (a.bottom - r.top + 8) + "px";
    el.style.maxHeight = (above ? a.top - r.top : r.bottom - a.bottom) - 20 + "px";
    el.style.overflowY = "auto";
    anchor.setAttribute("aria-expanded", "true");
    openPop = { el, anchor };
    return el;
  }
  document.addEventListener("pointerdown", (e) => {
    if (openPop && !openPop.el.contains(e.target) && !openPop.anchor.contains(e.target)) closePops();
  });

  function renderScope() {
    const s = state.scope;
    const chip = $("#scope-chip");
    chip.classList.toggle("on", !!s);
    chip.style.cssText = s ? docColor(s.filename) : "";
    $("#scope-text").textContent = s ? `Only ${s.filename}` : "All documents";
    $("#scope-icon").setAttribute("href", s ? "#i-file" : "#i-layers");
    $("#scope-clear")?.remove();
    if (s) {
      const b = document.createElement("button");
      b.type = "button"; b.id = "scope-clear"; b.className = "icon-btn chip-clear";
      b.setAttribute("aria-label", "Search all documents again");
      b.innerHTML = icon("x", "ic ic-sm");
      b.addEventListener("click", () => setScope(null));
      chip.after(b);
    }
    ta.placeholder = s ? `Ask about ${s.filename}…` : "Ask about your documents…";
  }
  function renderTitleSub() {
    $("#title-sub").textContent = state.scope
      ? `Answers only from ${state.scope.filename}`
      : state.docs.length
        ? `Answers grounded in ${plural(state.docs.length, "document")} · ${plural(totalChunks(), "passage")}`
        : "No documents yet. Add one to start asking questions.";
  }
  function setScope(doc) {
    state.scope = doc;
    renderScope();
    renderTitleSub();
    announce(doc ? `Questions now search only ${doc.filename}` : "Questions now search all documents");
  }
  $("#scope-chip").addEventListener("click", (e) => {
    if (openPop && openPop.anchor === e.currentTarget) return closePops();
    const cur = state.scope;
    const item = (id, label, n, on) => `<button type="button" class="menu-item" role="menuitemradio" aria-checked="${on}" data-scope="${esc(id)}"><span class="check">${on ? icon("check", "ic ic-sm") : ""}</span><span class="grow">${esc(label)}</span><span class="n">${n}</span></button>`;
    const pop = showPop(e.currentTarget, `<div class="menu-label">Search in</div>${item("", "All documents", state.docs.length, !cur)}${state.docs.length ? '<div class="menu-sep"></div>' : ""}${state.docs.map((d) => item(d.id, d.filename, fmt(d.chunks), !!cur && cur.id === d.id)).join("")}`, { role: "menu", label: "Search in" });
    pop.addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-scope]"); if (!b) return;
      setScope(b.dataset.scope ? state.docs.find((d) => d.id === b.dataset.scope) : null);
      closePops(); $("#scope-chip").focus();
    });
    pop.addEventListener("keydown", (ev) => {
      const items = [...pop.querySelectorAll(".menu-item")];
      const i = items.indexOf(document.activeElement);
      if (ev.key === "ArrowDown") { ev.preventDefault(); items[(i + 1) % items.length].focus(); }
      if (ev.key === "ArrowUp") { ev.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
    });
    (pop.querySelector('[aria-checked="true"]') || pop.querySelector(".menu-item")).focus();
  });
  function renderTopK() { $("#topk-text").textContent = plural(state.topK, "passage"); }
  $("#topk-chip").addEventListener("click", (e) => {
    if (openPop && openPop.anchor === e.currentTarget) return closePops();
    const pop = showPop(e.currentTarget, `<div class="pop-pad">
      <label for="topk-range">Passages to use <output id="topk-out" for="topk-range">${state.topK}</output></label>
      <input type="range" id="topk-range" min="1" max="20" value="${state.topK}">
      <p class="help">How many top-ranked passages the answer is written from. More gives broader context but slower answers.</p></div>`, { label: "Passages to use" });
    const r = pop.querySelector("#topk-range");
    r.addEventListener("input", () => {
      state.topK = +r.value;
      pop.querySelector("#topk-out").textContent = r.value;
      renderTopK();
      try { localStorage.setItem("rag-topk", r.value); } catch {}
    });
    r.focus();
  });

  // ---------------------------------------------------------------------------
  // Health
  // ---------------------------------------------------------------------------
  let bannerDismissed = false;
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      const h = await res.json();
      state.health = { api: true, qdrant: !!h.qdrant, redis: !!h.redis, version: h.version, at: Date.now() };
    } catch {
      state.health = { api: false, qdrant: false, redis: false, at: Date.now() };
    }
    renderHealth();
    return state.health;
  }
  function renderHealth() {
    const h = state.health;
    const problem = !h.api
      ? ["Can't reach the API", "Can't reach the API server. Questions and uploads won't work until it's back."]
      : !h.qdrant
        ? ["Qdrant offline", "Document search (Qdrant) isn't reachable. Questions can't be answered until it's back."]
        : !h.redis
          ? ["Redis offline", "Redis isn't reachable. New questions can't be sent or saved until it's back."]
          : null;
    $("#status-dot").className = "dot" + (problem ? " warn" : "");
    $("#status-text").textContent = problem ? problem[0] : "All systems normal";
    $("#banner-text").textContent = problem ? `${problem[1]} Checking again every 20 seconds.` : "";
    if (!problem) bannerDismissed = false;
    $("#banner").hidden = !problem || bannerDismissed;
  }
  $("#status-btn").addEventListener("click", (e) => {
    if (openPop && openPop.anchor === e.currentTarget) return closePops();
    const h = state.health || {};
    const row = (name, ok, what) => `<div class="health-row"><span class="dot ${ok ? "" : "warn"}"></span><span class="grow">${name}<br><span class="st">${what}</span></span><span class="st">${ok ? "Online" : "Offline"}</span></div>`;
    showPop(e.currentTarget, `<div class="menu-label">System status</div>
      ${row("API", h.api, "Answers questions")}
      ${row("Qdrant", h.qdrant, "Document search")}
      ${row("Redis", h.redis, "Chats, cache and rate limits")}
      <div class="menu-sep"></div><div class="health-row"><span class="st">${h.version ? `API v${esc(h.version)} · ` : ""}checked ${h.at ? ago(h.at) : "never"}</span></div>`, { label: "System status", above: true });
  });
  $("#banner-retry").addEventListener("click", async () => {
    const h = await checkHealth();
    toast(h.api && h.qdrant && h.redis ? "All systems normal" : "Still offline. Checking again in 20 seconds.", { duration: 3000 });
  });

  // ---------------------------------------------------------------------------
  // Source inspector
  // ---------------------------------------------------------------------------
  function markActive(sel) {
    document.querySelectorAll("#thread .is-active").forEach((el) => el.classList.remove("is-active"));
    if (sel) thread.querySelectorAll(`[data-mid="${sel.mid}"][data-n="${sel.n}"]`).forEach((el) => el.classList.add("is-active"));
  }
  function openSource(mid, n, trigger) {
    state.sel = { mid, n };
    markActive(state.sel);
    hideTip();
    openPanel("src", trigger);
  }
  function renderInspector() {
    const pane = $("#pane-src");
    const m = state.sel && findMsg(state.sel.mid);
    if (!m || !m.sources || !m.sources.length) {
      pane.innerHTML = `<div class="panel-empty"><p>Select a citation such as <span class="cite" style="cursor:default">1</span> in an answer to read the passage it came from.</p></div>`;
      return;
    }
    const total = m.sources.length;
    const n = Math.min(state.sel.n, total), s = m.sources[n - 1];
    const doc = docByName(s.filename);
    const max = Math.max(...m.sources.map((x) => x.score));
    const pct = max > 0 ? Math.max(0.04, Math.max(0, s.score) / max) : 0.04;
    const rank = [...m.sources].sort((a, b) => b.score - a.score).indexOf(s) + 1;
    pane.innerHTML = `
      <p class="insp-context">From the answer to <q>${esc(m.question)}</q></p>
      <div class="num-strip" role="group" aria-label="Sources in this answer">${m.sources.map((x, i) => `<button type="button" class="src-num ${i + 1 === n ? "is-active" : ""}" style="${docColor(x.filename)}" aria-pressed="${i + 1 === n}" data-insp="${i + 1}" aria-label="Source ${i + 1}">${i + 1}</button>`).join("")}</div>
      <section aria-labelledby="insp-title">
        <div class="insp-file"><span class="file-badge" style="${docColor(s.filename)}">${esc((doc ? doc.type : (s.filename.split(".").pop() || "doc")).toUpperCase())}</span><div><h3 id="insp-title">${esc(s.filename || "Unknown file")}</h3><div class="sub">Passage ${fmt(s.chunk_index + 1)}${doc ? " of " + fmt(doc.chunks) : ""}</div></div></div>
        <div class="meter">
          <div class="meter-top"><span>Reranker score</span><span class="mono">${Number(s.score).toFixed(2)}</span></div>
          <div class="track" role="img" aria-label="Relative relevance ${Math.round(pct * 100)} percent of the best match"><div class="fill" style="transform:scaleX(${pct})"></div></div>
          <div class="meter-note">${rank === 1 ? "Best match in this answer." : `Ranked ${rank} of ${total} in this answer.`} Higher is more relevant; compare within one answer.</div>
        </div>
        <blockquote class="passage">${esc(cleanPassage(s.content))}</blockquote>
        <div class="insp-nav">
          <button type="button" class="btn" data-step="-1" ${n === 1 ? "disabled" : ""}>${icon("chev-left", "ic ic-sm")}Previous</button>
          <span class="mono">${n} of ${total}</span>
          <button type="button" class="btn" data-step="1" ${n === total ? "disabled" : ""}>Next${icon("chev-right", "ic ic-sm")}</button>
        </div>
        ${doc ? `<button type="button" class="btn full" data-askdoc="${esc(doc.id)}">${icon("search", "ic ic-sm")}Ask only about this document</button>` : ""}
      </section>`;
  }
  $("#pane-src").addEventListener("click", (e) => {
    const b = e.target.closest("[data-insp], [data-step]");
    if (b && !b.disabled) {
      const n = b.dataset.insp ? +b.dataset.insp : state.sel.n + +b.dataset.step;
      state.sel.n = n;
      markActive(state.sel);
      renderInspector();
      const again = $(`#pane-src [data-insp="${n}"]`); if (again && b.dataset.insp) again.focus();
      return;
    }
    const ad = e.target.closest("[data-askdoc]");
    if (ad) scopeAndAsk(state.docs.find((d) => d.id === ad.dataset.askdoc));
  });
  function scopeAndAsk(doc) {
    if (!doc) return;
    setScope(doc);
    if (isOverlayPanel()) { panelReturn = null; closePanel(); }
    ta.focus();
    toast(`Questions now search only ${doc.filename}`, { duration: 3500 });
  }

  // Citation preview on hover/focus (mouse and keyboard; touch opens the inspector directly)
  let tipEl = null, tipTimer = null;
  function hideTip() { clearTimeout(tipTimer); if (tipEl) { tipEl.remove(); tipEl = null; } }
  function showTip(btn) {
    const m = findMsg(btn.dataset.mid); if (!m || !m.sources) return;
    const s = m.sources[+btn.dataset.n - 1]; if (!s) return;
    hideTip();
    tipEl = document.createElement("div");
    tipEl.className = "pop tip"; tipEl.setAttribute("role", "tooltip"); tipEl.id = "cite-tip";
    const text = cleanPassage(s.content);
    tipEl.innerHTML = `<div class="t-file" style="${docColor(s.filename)}"><span class="doc-sw" aria-hidden="true"></span><span>${esc(s.filename)}</span></div><p>${esc(text.slice(0, 170))}${text.length > 170 ? "…" : ""}</p><div class="t-foot">Passage ${fmt(s.chunk_index + 1)} · click to read the full passage</div>`;
    app.appendChild(tipEl);
    btn.setAttribute("aria-describedby", "cite-tip");
    const a = btn.getBoundingClientRect(), r = app.getBoundingClientRect();
    const top = a.top - r.top - tipEl.offsetHeight - 8;
    tipEl.style.left = Math.max(12, Math.min(a.left - r.left - 20, r.width - tipEl.offsetWidth - 12)) + "px";
    tipEl.style.top = (top < 8 ? a.bottom - r.top + 8 : top) + "px";
  }
  thread.addEventListener("pointerover", (e) => {
    const b = e.target.closest(".cite[data-mid], .src-num[data-mid]");
    if (!b || e.pointerType !== "mouse") return;
    clearTimeout(tipTimer); tipTimer = setTimeout(() => showTip(b), 180);
  });
  thread.addEventListener("pointerout", (e) => { if (e.target.closest(".cite, .src-num")) hideTip(); });
  thread.addEventListener("focusin", (e) => { const b = e.target.closest(".cite[data-mid], .src-num[data-mid]"); if (b && b.matches(":focus-visible")) showTip(b); });
  thread.addEventListener("focusout", (e) => { if (e.target.closest(".cite, .src-num")) hideTip(); });
  scroller.addEventListener("scroll", hideTip);

  // ---------------------------------------------------------------------------
  // Documents + uploads
  // ---------------------------------------------------------------------------
  async function loadDocs() {
    try {
      const res = await fetch("/documents");
      if (!res.ok) return;
      const data = await res.json();
      const known = new Set(state.docs.map((d) => d.id));
      const first = !state.docs.length && !state.docsLoaded;
      state.docs = (data.documents || []).map((d) => ({
        id: d.source_id, filename: d.filename, type: d.doc_type || "doc", at: d.ingested_at, chunks: d.chunk_count,
        isNew: !first && !known.has(d.source_id),
      }));
      state.docsLoaded = true;
      if (state.scope) state.scope = state.docs.find((d) => d.id === state.scope.id) || null;
      $("#docs-count").textContent = state.docs.length;
      $("#tab-docs-count").textContent = state.docs.length;
      renderTitleSub();
      renderScope();
      if (panelShows("docs")) renderDocs();
      if (panelShows("src")) renderInspector();
      // Document colours depend on this list, so repaint whatever is on screen
      // (a chat can finish loading before the documents do).
      if (!state.current || state.messages[state.current]) {
        const top = scroller.scrollTop;
        renderThread();
        scroller.scrollTop = top;
      }
    } catch {
      // Keep the last-known list; the status pill already reports connectivity.
    }
  }
  function renderDocs() {
    const pane = $("#pane-docs");
    const docs = [...state.docs].sort((a, b) => Date.parse(b.at) - Date.parse(a.at));
    pane.innerHTML = `
      <div class="kb-sum"><strong>${plural(docs.length, "document")}</strong><span class="mono">${plural(totalChunks(), "passage")}</span></div>
      <label class="dropzone" for="file-input">${icon("upload")}<span>Drop files here or <span class="u">browse</span></span><small>PDF, TXT or MD · up to 20 MB each · or drop anywhere in the window</small></label>
      ${state.uploads.length ? `<ul class="uploads" aria-label="Uploads">${state.uploads.map(upHtml).join("")}</ul>` : ""}
      <h2 class="doc-head">In your knowledge base</h2>
      ${docs.length ? `<ul class="docs">${docs.map((d) => `
        <li class="doc ${d.isNew ? "is-new" : ""}">
          <span class="file-badge" style="${docColor(d.filename)}">${esc(d.type.toUpperCase())}</span>
          <span class="doc-name">${esc(d.filename)}</span>
          <span class="doc-meta">${plural(d.chunks, "passage")} · added ${ago(Date.parse(d.at))}</span>
          <button type="button" class="doc-ask" data-askdoc="${esc(d.id)}">${icon("search", "ic ic-sm")}Ask about this</button>
        </li>`).join("")}</ul>` : '<p class="list-empty">No documents yet. Add a PDF, TXT or Markdown file to start asking questions.</p>'}`;
  }
  function upHtml(u) {
    const pct = u.status === "ready" ? 1 : u.status === "processing" && u.total ? Math.max(0.04, u.done / u.total) : u.status === "uploading" ? 0.02 : 0.04;
    const status = {
      uploading: "Uploading…",
      queued: "Waiting in queue",
      processing: u.total ? `Processing · ${fmt(u.done)} of ${plural(u.total, "passage")}` : "Reading the file…",
      ready: `Ready · ${plural(u.total, "passage")}`,
      failed: "Couldn't add",
    }[u.status];
    return `<li class="up ${u.status}">
      <div class="up-top">${u.status === "ready" ? icon("check", "ic ic-sm") : u.status === "failed" ? icon("alert", "ic ic-sm") : '<span class="spin" aria-hidden="true"></span>'}<span class="up-name" title="${esc(u.name)}">${esc(u.name)}</span><span class="up-status">${status}</span></div>
      ${u.status === "failed"
        ? `<p class="up-err">${esc(u.error)}</p>${u.file ? `<button type="button" class="btn" data-reup="${u.id}">${icon("retry", "ic ic-sm")}Try again</button>` : ""}`
        : `<div class="track" role="progressbar" aria-label="${esc(u.name)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(pct * 100)}"><div class="fill" style="transform:scaleX(${pct})"></div></div>`}
    </li>`;
  }
  $("#pane-docs").addEventListener("click", (e) => {
    const ad = e.target.closest("[data-askdoc]");
    if (ad) return scopeAndAsk(state.docs.find((d) => d.id === ad.dataset.askdoc));
    const re = e.target.closest("[data-reup]");
    if (re) {
      const u = state.uploads.find((x) => x.id === re.dataset.reup);
      if (u) { Object.assign(u, { status: "uploading", error: "", done: 0, total: 0 }); runUpload(u); }
    }
  });
  const paintUploads = () => { if (panelShows("docs")) renderDocs(); };
  function addFiles(files) {
    if (!files.length) return;
    openPanel("docs");
    for (const f of files) {
      const ext = (f.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
      const u = { id: "u" + (++uid), name: f.name, status: "uploading", done: 0, total: 0 };
      state.uploads.unshift(u);
      if (![".pdf", ".txt", ".md"].includes(ext)) { u.status = "failed"; u.error = `${ext || "This file type"} isn't supported. Use PDF, TXT or MD.`; continue; }
      if (f.size > 20 * 1024 * 1024) { u.status = "failed"; u.error = "The file is larger than 20 MB. Split it into smaller files and try again."; continue; }
      u.file = f;
      runUpload(u);
    }
    paintUploads();
  }
  async function runUpload(u) {
    paintUploads();
    try {
      const form = new FormData();
      form.append("file", u.file);
      const res = await fetch("/ingest", { method: "POST", body: form });
      if (!res.ok) {
        const detail = res.status === 429
          ? "Upload limit reached (10 per minute). Wait a moment, then try again."
          : (await res.json().catch(() => ({}))).detail || `The server returned HTTP ${res.status}.`;
        throw new Error(detail);
      }
      const { task_id } = await res.json();
      u.status = "queued"; paintUploads();
      while (true) {
        await sleep(1200);
        let data;
        try { data = await (await fetch(`/ingest/status/${encodeURIComponent(task_id)}`)).json(); } catch { continue; }
        if (data.status === "succeeded") { u.total = u.done = data.result?.chunk_count ?? u.total; break; }
        if (data.status === "failed") throw new Error(data.error || "Ingestion failed.");
        if (data.status === "started" || data.status === "retried") {
          u.status = "processing";
          if (data.progress?.total_chunks) { u.done = data.progress.chunks_processed; u.total = data.progress.total_chunks; }
        }
        paintUploads();
      }
      u.status = "ready";
      delete u.file;
      await loadDocs();
      paintUploads();
      toast(`Added ${u.name} · ${plural(u.total, "passage")}`, { duration: 4000 });
      announce(`${u.name} is ready to search`);
      setTimeout(() => { state.uploads = state.uploads.filter((x) => x !== u); paintUploads(); }, 6000);
    } catch (err) {
      u.status = "failed";
      u.error = String(err.message || err).slice(0, 240);
      paintUploads();
      announce(`Couldn't add ${u.name}`);
    }
  }
  $("#attach").addEventListener("click", () => $("#file-input").click());
  $("#file-input").addEventListener("change", (e) => { addFiles([...e.target.files]); e.target.value = ""; });

  // Drop anywhere in the window
  let dragDepth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  window.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); dragDepth++; $("#drop").hidden = false; });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("dragleave", () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $("#drop").hidden = true; });
  window.addEventListener("drop", (e) => { if (!hasFiles(e)) return; e.preventDefault(); dragDepth = 0; $("#drop").hidden = true; addFiles([...e.dataTransfer.files]); });

  // ---------------------------------------------------------------------------
  // Toasts
  // ---------------------------------------------------------------------------
  function toast(msg, { action, onAction, duration = 4000 } = {}) {
    const el = document.createElement("div");
    el.className = "toast";
    el.innerHTML = `<span class="msg">${esc(msg)}</span>${action ? `<button type="button">${esc(action)}</button>` : ""}<span class="timer" style="animation-duration:${duration}ms"></span>`;
    const timer = setTimeout(() => el.remove(), duration);
    if (action) el.querySelector("button").addEventListener("click", () => { clearTimeout(timer); el.remove(); onAction && onAction(); });
    $("#toasts").appendChild(el);
  }

  // ---------------------------------------------------------------------------
  // Theme
  // ---------------------------------------------------------------------------
  const THEMES = ["system", "light", "dark"];
  let theme = "system";
  try { theme = localStorage.getItem("rag-theme") || "system"; } catch {}
  function applyTheme() {
    if (theme === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.dataset.theme = theme;
    const next = THEMES[(THEMES.indexOf(theme) + 1) % 3];
    $("#theme-icon").setAttribute("href", { system: "#i-monitor", light: "#i-sun", dark: "#i-moon" }[theme]);
    $("#theme-btn").setAttribute("aria-label", `Theme: ${theme}. Switch to ${next}`);
    $("#theme-btn").title = `Theme: ${theme}`;
  }
  $("#theme-btn").addEventListener("click", () => {
    theme = THEMES[(THEMES.indexOf(theme) + 1) % 3];
    try { localStorage.setItem("rag-theme", theme); } catch {}
    applyTheme();
    announce(`Theme set to ${theme}`);
  });

  // ---------------------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------------------
  $("#new-chat").addEventListener("click", () => newChat());
  $("#top-new").addEventListener("click", () => newChat());
  $("#open-nav").addEventListener("click", () => {
    if (layout() === "narrow") openDrawer();
    else { app.classList.remove("sb-collapsed"); $("#sb-collapse").focus(); }
  });
  $("#sb-collapse").addEventListener("click", () => {
    if (layout() === "narrow") { closeDrawer(); $("#open-nav").focus(); }
    else { app.classList.add("sb-collapsed"); $("#open-nav").focus(); }
  });
  $("#docs-btn").addEventListener("click", (e) => {
    if (panelShows("docs")) closePanel();
    else openPanel("docs", e.currentTarget);
  });
  $("#panel-close").addEventListener("click", closePanel);
  $("#tab-src").addEventListener("click", () => setTab("src"));
  $("#tab-docs").addEventListener("click", () => setTab("docs"));
  $(".tabs").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const t = state.tab === "src" ? "docs" : "src";
    setTab(t); $(t === "src" ? "#tab-src" : "#tab-docs").focus();
  });
  $("#scrim").addEventListener("click", () => { closeDrawer(); if (app.classList.contains("panel-open")) closePanel(); });

  const isTyping = (t) => t.closest && t.closest("input, textarea, [contenteditable]");
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (closePops()) return;
      if (app.classList.contains("drawer-open")) { closeDrawer(); $("#open-nav").focus(); return; }
      if (app.classList.contains("panel-open") && isOverlayPanel()) { closePanel(); return; }
      if (stopCurrent()) return;
    }
    if (e.key === "/" && !isTyping(e.target)) { e.preventDefault(); ta.focus(); }
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); newChat(); }
  });

  // ---------------------------------------------------------------------------
  // Start
  // ---------------------------------------------------------------------------
  applyTheme();
  renderTopK();
  renderScope();
  renderChats();
  syncInert();
  route();
  loadDocs();
  loadChats();
  checkHealth();
  setInterval(checkHealth, 20000);
})();
