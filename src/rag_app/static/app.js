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

let requestInFlight = false;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/**
 * Escape text, then apply a tiny bit of markup: "[Source N]" citations,
 * bold emphasis, and inline code (LLM answers commonly use bare markdown
 * for these, which reads as broken if left as literal asterisks/backticks).
 * Safe to assign to innerHTML: escapeHtml() runs first, so the only
 * markup ever introduced is the literal tags this function adds itself --
 * no raw LLM/user content reaches the DOM unescaped.
 */
function renderAnswerHtml(text) {
  // \s+ rather than a literal space: some models emit a narrow no-break
  // space (U+202F) inside "[Source N]" instead of a plain ASCII space.
  return escapeHtml(text)
    .replace(/\[Source\s+(\d+)(?::[^\]]*)?\]/g, "<cite>[$1]</cite>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
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
        <div class="doc-row-name" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</div>
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
  const col = document.createElement("div");
  col.className = "msg-row assistant";
  col.innerHTML = `
    <div class="msg-meta-col">
      <div class="bubble assistant"><span class="typing-dots"><span></span><span></span><span></span></span></div>
    </div>`;
  messagesEl.appendChild(col);
  scrollToBottom();
  return col.querySelector(".bubble");
}

function finalizeAssistantMessage(bubbleEl, { sources, cached, llm_provider, llm_model, latency_ms }) {
  const col = bubbleEl.closest(".msg-meta-col");

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
            (s) => `
          <div class="source-item">
            <div class="source-item-head">
              <span>${escapeHtml(s.filename || s.source_id || "unknown")}</span>
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

async function sendMessage(query) {
  requestInFlight = true;
  sendBtn.disabled = true;
  composerStatus.textContent = "";

  addUserMessage(query);
  const bubble = addAssistantPlaceholder();
  let started = false;
  let fullText = "";

  try {
    const res = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: Number(topKInput.value) || 5 }),
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
          if (!started) {
            bubble.innerHTML = "";
            started = true;
          }
          fullText += payload.content;
          bubble.innerHTML = renderAnswerHtml(fullText);
          scrollToBottom();
        } else if (payload.type === "answer") {
          fullText = payload.content;
          bubble.innerHTML = renderAnswerHtml(fullText);
        } else if (payload.type === "done") {
          finalizeAssistantMessage(bubble, payload);
        } else if (payload.type === "error") {
          bubble.classList.add("error");
          bubble.textContent = payload.message || "Something went wrong.";
        }
      }
    }

    if (!started && !fullText) {
      bubble.textContent = "No response received.";
    }
  } catch (err) {
    bubble.classList.add("error");
    bubble.textContent = String(err.message || err);
  } finally {
    requestInFlight = false;
    sendBtn.disabled = false;
    scrollToBottom();
  }
}

composerEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = queryInput.value.trim();
  if (!text || requestInFlight) return;
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
