// RAG Chat UI — vanilla JS, no build step.
// Talks to the FastAPI backend served alongside this page.

const messagesEl = document.getElementById("messages");
const emptyStateEl = document.getElementById("empty-state");
const composerEl = document.getElementById("composer");
const queryInput = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const topKInput = document.getElementById("top-k-input");
const composerStatus = document.getElementById("composer-status");
const healthDot = document.getElementById("health-dot");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const uploadListEl = document.getElementById("upload-list");
const docListEl = document.getElementById("doc-list");
const docCountEl = document.getElementById("doc-count");
const docsToggle = document.getElementById("docs-toggle");
const newChatBtn = document.getElementById("new-chat-btn");
const chatListEl = document.getElementById("chat-list");

let messageIdCounter = 0;
let currentChatId = null;

// Which chat(s) currently have a request streaming, keyed by chat_id --
// or "__new__" for a not-yet-created chat, since it has no id until the
// server assigns one. Per-chat (not one global flag) so switching to a
// *different* chat while one streams doesn't block sending there too;
// switching back to a chat that's still streaming still shows it as busy.
const inFlightChats = new Set();

// The live (still in-DOM-memory, possibly detached) assistant message row
// for each in-flight chat, keyed the same as inFlightChats. Lets openChat()
// re-show real progress (typing dots or partial text) instead of a stale
// view when the user switches back into a chat that's still streaming.
const inFlightBubbles = new Map();

function isCurrentChatBusy() {
  return inFlightChats.has(currentChatId ?? "__new__");
}

function refreshComposerState() {
  sendBtn.disabled = isCurrentChatBusy();
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/**
 * Like escapeHtml, but also safe inside a quoted HTML attribute value.
 * The textContent round-trip escapeHtml() uses doesn't touch quote
 * characters (they're not special in text-node content), so a value
 * like `foo".onmouseover="..` would break out of a `title="${...}"`
 * attribute if escapeHtml() alone were used there.
 */
function escapeAttr(str) {
  return escapeHtml(str).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/**
 * Escape text, then apply a tiny bit of markup: "[Source N]" citations
 * (as a clickable button, wired up to jump to that source further down
 * the same message -- see the delegated click handler near the bottom
 * of this file), bold emphasis, and inline code (LLM answers commonly
 * use bare markdown for these, which reads as broken if left as literal
 * asterisks/backticks).
 * Safe to assign to innerHTML: escapeHtml() runs first, so the only
 * markup ever introduced is the literal tags this function adds itself --
 * no raw LLM/user content reaches the DOM unescaped. Citation numbers
 * come from a regex-matched \d+, never from unescaped LLM text, so they
 * can't carry markup either.
 */
function renderAnswerHtml(text, messageId) {
  // \s+ rather than a literal space: some models emit a narrow no-break
  // space (U+202F) inside "[Source N]" instead of a plain ASCII space.
  // Bracket class covers both ASCII [] and full-width 【】 -- the model
  // isn't consistent about which one it uses between responses.
  const withInline = escapeHtml(text)
    .replace(
      /[[【]Source\s+(\d+)(?::[^\]】]*)?[\]】]/g,
      `<button type="button" class="citation" data-msg="${messageId}" data-n="$1">[$1]</button>`
    )
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
  return renderMarkdownTables(withInline);
}

/**
 * Convert GitHub-style markdown tables (a "| a | b |" header row, a
 * "|---|---|" separator row, then more pipe rows) into real <table>
 * markup. Runs on already-escaped/inline-formatted HTML, splitting on
 * "\n" -- safe because none of the replacements above introduce a
 * literal newline. Non-table lines pass through untouched (the bubble's
 * white-space: pre-wrap already renders their newlines correctly).
 */
function renderMarkdownTables(html) {
  const lines = html.split("\n");
  const isRow = (l) => /^\s*\|.*\|\s*$/.test(l);
  const isSeparator = (l) => /^\s*\|?(\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$/.test(l);
  const splitCells = (l) => {
    let s = l.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|")) s = s.slice(0, -1);
    return s.split("|").map((c) => c.trim());
  };

  const out = [];
  for (let i = 0; i < lines.length; i++) {
    if (isRow(lines[i]) && isSeparator(lines[i + 1] ?? "")) {
      const header = splitCells(lines[i]);
      i += 2;
      const rows = [];
      while (i < lines.length && isRow(lines[i])) {
        rows.push(splitCells(lines[i]));
        i++;
      }
      i--; // outer loop's i++ accounts for the row just past the table
      out.push(
        `<table class="md-table"><thead><tr>${header
          .map((c) => `<th>${c}</th>`)
          .join("")}</tr></thead><tbody>${rows
          .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`)
          .join("")}</tbody></table>`
      );
    } else {
      out.push(lines[i]);
    }
  }
  return out.join("\n");
}

function relativeTime(isoString) {
  if (!isoString) return "";
  const diffMs = Date.now() - new Date(isoString).getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const ok = res.ok;
    healthDot.className = "brand-dot " + (ok ? "healthy" : "down");
    healthDot.title = ok ? "API reachable" : "API unreachable";
  } catch {
    healthDot.className = "brand-dot down";
    healthDot.title = "API unreachable";
  }
}
checkHealth();
setInterval(checkHealth, 20000);

// ---------------------------------------------------------------------------
// Document list
// ---------------------------------------------------------------------------

async function refreshDocuments() {
  try {
    const res = await fetch("/documents");
    if (!res.ok) return;
    const data = await res.json();
    renderDocuments(data.documents || []);
  } catch {
    // Leave the last-known list in place; the health dot already signals connectivity issues.
  }
}

function renderDocuments(docs) {
  docCountEl.textContent = docs.length || "";
  if (docs.length === 0) {
    docListEl.innerHTML = '<div class="empty-hint">No documents ingested yet.</div>';
    return;
  }
  docListEl.innerHTML = docs
    .map(
      (d) => `
      <div class="doc-row">
        <div class="doc-row-name" title="${escapeAttr(d.filename)}">${escapeHtml(d.filename)}</div>
        <div class="doc-row-meta">
          <span class="doc-row-type">${escapeHtml(d.doc_type)}</span>
          <span>${d.chunk_count} chunk${d.chunk_count === 1 ? "" : "s"}</span>
          <span>${relativeTime(d.ingested_at)}</span>
        </div>
      </div>`
    )
    .join("");
}

refreshDocuments();

docsToggle.addEventListener("click", () => {
  const nowHidden = !docListEl.hasAttribute("hidden");
  docListEl.toggleAttribute("hidden", nowHidden);
  docsToggle.classList.toggle("open", !nowHidden);
});

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFile(fileInput.files[0]);
  fileInput.value = "";
});

async function uploadFile(file) {
  const row = document.createElement("div");
  row.className = "upload-row";
  row.innerHTML = `<div class="spinner"></div><div class="upload-row-name">${escapeHtml(
    file.name
  )}</div><div class="upload-row-status">uploading</div>`;
  uploadListEl.prepend(row);

  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/ingest", { method: "POST", body: form });
    if (!res.ok) {
      const detail = (await res.json().catch(() => ({}))).detail || res.statusText;
      throw new Error(detail);
    }
    const { task_id } = await res.json();
    row.querySelector(".upload-row-status").textContent = "queued";
    pollIngestStatus(task_id, row);
  } catch (err) {
    row.classList.add("failed");
    row.querySelector(".spinner")?.remove();
    row.querySelector(".upload-row-status").textContent = String(err.message || err).slice(0, 60);
  }
}

async function pollIngestStatus(taskId, row) {
  const statusEl = row.querySelector(".upload-row-status");
  const poll = async () => {
    try {
      const res = await fetch(`/ingest/status/${taskId}`);
      const data = await res.json();
      if (data.status === "succeeded") {
        row.classList.add("succeeded");
        row.querySelector(".spinner")?.remove();
        statusEl.textContent = `${data.result?.chunk_count ?? "?"} chunks`;
        refreshDocuments();
        setTimeout(() => row.remove(), 6000);
        return;
      }
      if (data.status === "failed") {
        row.classList.add("failed");
        row.querySelector(".spinner")?.remove();
        statusEl.textContent = (data.error || "failed").slice(0, 60);
        return;
      }
      statusEl.textContent = data.status;
      setTimeout(poll, 1200);
    } catch {
      setTimeout(poll, 2000);
    }
  };
  poll();
}

// ---------------------------------------------------------------------------
// Chat sessions (sidebar)
// ---------------------------------------------------------------------------

async function loadChats() {
  try {
    const res = await fetch("/chats");
    if (!res.ok) return;
    const data = await res.json();
    renderChats(data.chats || []);
  } catch {
    // Leave the last-known list in place.
  }
}

function renderChats(chats) {
  if (chats.length === 0) {
    chatListEl.innerHTML = '<div class="empty-hint">No chats yet.</div>';
    return;
  }
  chatListEl.innerHTML = chats
    .map(
      (c) => `
      <button type="button" class="chat-row${c.id === currentChatId ? " active" : ""}"
        data-chat-id="${escapeAttr(c.id)}" title="${escapeAttr(c.title)}">${escapeHtml(c.title)}</button>`
    )
    .join("");
}

function highlightActiveChat(chatId) {
  chatListEl.querySelectorAll(".chat-row").forEach((row) => {
    row.classList.toggle("active", row.dataset.chatId === chatId);
  });
}

/** Remove message rows but keep the reusable #empty-state node in the DOM. */
function clearMessages() {
  messagesEl.querySelectorAll(".msg-row").forEach((el) => el.remove());
  emptyStateEl.style.display = "";
}

async function openChat(chatId) {
  // Switching chats is always allowed, even mid-stream elsewhere -- that
  // other stream keeps running in the background (see sendMessage) and
  // saves correctly regardless of what's currently on screen.
  if (chatId === currentChatId) return;
  try {
    const res = await fetch(`/chats/${chatId}`);
    if (!res.ok) return;
    const data = await res.json();
    currentChatId = data.id;
    clearMessages();
    for (const msg of data.messages || []) {
      if (msg.role === "user") {
        addUserMessage(msg.content);
      } else {
        renderCompleteAssistantMessage(msg.content, {
          sources: msg.sources,
          llm_provider: msg.llm_provider,
          llm_model: msg.llm_model,
        });
      }
    }
    // Stored history stops at the last completed message -- if this chat
    // is still streaming, its assistant reply isn't persisted yet, so
    // re-attach the live in-progress bubble instead of leaving the user's
    // question looking unanswered.
    const live = inFlightChats.has(chatId) ? inFlightBubbles.get(chatId) : null;
    if (live) {
      messagesEl.appendChild(live.row);
      live.renderLive(true);
    }
    highlightActiveChat(chatId);
    refreshComposerState();
    scrollToBottom();
  } catch (err) {
    // Leave the current view as-is on failure, but never swallow this
    // silently -- a bug here previously would have been invisible.
    console.error("openChat failed:", err);
  }
}

chatListEl.addEventListener("click", (e) => {
  const row = e.target.closest(".chat-row");
  if (!row) return;
  openChat(row.dataset.chatId);
});

newChatBtn.addEventListener("click", () => {
  currentChatId = null;
  clearMessages();
  highlightActiveChat(null);
  refreshComposerState();
});

loadChats();

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

function addUserMessage(text) {
  emptyStateEl.style.display = "none";
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `<div class="bubble user"></div>`;
  row.querySelector(".bubble").textContent = text;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function addAssistantPlaceholder() {
  const messageId = ++messageIdCounter;
  const col = document.createElement("div");
  col.className = "msg-row assistant";
  col.innerHTML = `
    <div class="msg-meta-col">
      <div class="bubble assistant"><span class="typing-dots"><span></span><span></span><span></span></span></div>
    </div>`;
  messagesEl.appendChild(col);
  scrollToBottom();
  const bubble = col.querySelector(".bubble");
  bubble.dataset.msgId = String(messageId);
  return bubble;
}

function finalizeAssistantMessage(bubbleEl, { sources, cached, llm_provider, llm_model, latency_ms }) {
  const col = bubbleEl.closest(".msg-meta-col");
  const messageId = bubbleEl.dataset.msgId;

  const tags = [];
  if (cached) tags.push('<span class="tag cached">cached</span>');
  if (llm_provider) tags.push(`<span class="tag">${escapeHtml(llm_provider)}</span>`);
  if (llm_model) tags.push(`<span class="tag">${escapeHtml(llm_model)}</span>`);
  if (typeof latency_ms === "number") tags.push(`<span class="tag">${Math.round(latency_ms)}ms</span>`);

  const meta = document.createElement("div");
  meta.className = "msg-meta";
  meta.innerHTML = tags.join("");
  col.appendChild(meta);

  if (sources && sources.length) {
    const wrap = document.createElement("div");
    wrap.className = "sources";
    wrap.innerHTML = `
      <button type="button" class="sources-toggle">
        <span class="chevron">&#9656;</span> ${sources.length} source${sources.length === 1 ? "" : "s"}
      </button>
      <div class="source-list">
        ${sources
          .map(
            (s, i) => `
          <div class="source-item" id="src-${messageId}-${i + 1}">
            <div class="source-item-head">
              <span>[${i + 1}] ${escapeHtml(s.filename || s.source_id || "unknown")}</span>
              <span class="source-item-score">${s.score}</span>
            </div>
            <div class="source-item-content">${escapeHtml((s.content || "").slice(0, 220))}${
              (s.content || "").length > 220 ? "…" : ""
            }</div>
          </div>`
          )
          .join("")}
      </div>`;
    const toggle = wrap.querySelector(".sources-toggle");
    const list = wrap.querySelector(".source-list");
    toggle.addEventListener("click", () => {
      toggle.classList.toggle("open");
      list.classList.toggle("open");
    });
    col.appendChild(wrap);
  }
  scrollToBottom();
}

/** Render a fully-known assistant message in one shot (no streaming) --
 * used to replay a stored chat's history when it's opened. */
function renderCompleteAssistantMessage(content, meta) {
  const bubble = addAssistantPlaceholder();
  bubble.innerHTML = renderAnswerHtml(content, bubble.dataset.msgId);
  finalizeAssistantMessage(bubble, meta);
}

async function sendMessage(query) {
  // The chat this specific request belongs to, fixed at send time -- the
  // user may switch to a different chat (or another new one) before this
  // resolves, so `currentChatId` itself can change out from under us.
  const targetChatId = currentChatId;
  const inFlightKey = targetChatId ?? "__new__";
  inFlightChats.add(inFlightKey);
  refreshComposerState();
  composerStatus.textContent = "";

  addUserMessage(query);
  const bubble = addAssistantPlaceholder();
  const row = bubble.closest(".msg-row");
  let started = false;
  let fullText = "";
  // Whether the chat this request is for is still the one on screen --
  // re-checked before every UI-visible update so a background response
  // (from a chat the user has since navigated away from) never repaints
  // or scrolls the conversation someone is currently reading.
  const isOnScreen = () => document.body.contains(bubble);

  // Redraws the bubble from the latest known text. `withCursor` shows a
  // blinking cursor while still streaming; called both from the token
  // loop and (via inFlightBubbles) when the user switches back into this
  // chat mid-stream, so what they see always reflects the true progress
  // instead of a stale "thinking" placeholder or blank message.
  function renderLive(withCursor) {
    if (!started) return; // still just the typing-dots placeholder
    bubble.innerHTML =
      renderAnswerHtml(fullText, bubble.dataset.msgId) + (withCursor ? '<span class="stream-cursor"></span>' : "");
  }

  // Lets openChat() re-show this exact in-progress message (dots or
  // partial text) if the user navigates away and back before it finishes --
  // otherwise a stored-history reload only has the user's question, since
  // the assistant message isn't persisted until the stream completes.
  inFlightBubbles.set(inFlightKey, { row, renderLive });

  try {
    const res = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: Number(topKInput.value) || 5, chat_id: targetChatId }),
    });

    if (res.status === 429) {
      throw new Error("Rate limit reached — please wait a moment before asking again.");
    }
    if (!res.ok || !res.body) {
      throw new Error(`Request failed (${res.status})`);
    }

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
        const payload = JSON.parse(line.slice(5).trim());

        if (payload.type === "token") {
          fullText += payload.content;
          started = true;
          if (isOnScreen()) {
            renderLive(true);
            scrollToBottom();
          }
        } else if (payload.type === "answer") {
          fullText = payload.content;
          started = true;
          if (isOnScreen()) renderLive(false);
        } else if (payload.type === "done") {
          renderLive(false); // drop the trailing cursor before finalizing
          if (isOnScreen()) finalizeAssistantMessage(bubble, payload);
          if (payload.chat_id) {
            if (isOnScreen()) {
              currentChatId = payload.chat_id;
              loadChats().then(() => highlightActiveChat(currentChatId));
            } else {
              loadChats(); // still refresh titles/order in the background
            }
          }
        } else if (payload.type === "error") {
          if (isOnScreen()) {
            bubble.classList.add("error");
            bubble.textContent = payload.message || "Something went wrong.";
          }
        }
      }
    }

    if (!started && !fullText && isOnScreen()) {
      bubble.textContent = "No response received.";
    }
  } catch (err) {
    if (isOnScreen()) {
      bubble.classList.add("error");
      bubble.textContent = String(err.message || err);
    }
  } finally {
    inFlightChats.delete(inFlightKey);
    inFlightBubbles.delete(inFlightKey);
    refreshComposerState();
    if (isOnScreen()) scrollToBottom();
  }
}

composerEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = queryInput.value.trim();
  if (!text || isCurrentChatBusy()) return;
  queryInput.value = "";
  queryInput.style.height = "auto";
  sendMessage(text);
});

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composerEl.requestSubmit();
  }
});

queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + "px";
});

// Delegated (not per-button) since citation buttons are created dynamically
// on every streamed token re-render.
messagesEl.addEventListener("click", (e) => {
  const btn = e.target.closest(".citation");
  if (!btn) return;
  const target = document.getElementById(`src-${btn.dataset.msg}-${btn.dataset.n}`);
  if (!target) return; // sources arrive with the "done" event; a click before that is a no-op

  const sourcesWrap = target.closest(".sources");
  const list = sourcesWrap?.querySelector(".source-list");
  const toggle = sourcesWrap?.querySelector(".sources-toggle");
  if (list && !list.classList.contains("open")) {
    list.classList.add("open");
    toggle?.classList.add("open");
  }

  target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  target.classList.add("flash");
  setTimeout(() => target.classList.remove("flash"), 900);
});
