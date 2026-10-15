// Escape-first markdown for LLM answers -- no dependencies.
// Every character of model text passes through escapeHtml() before any tag is
// added, so the only markup in the output is markup this file writes itself:
// safe to assign to innerHTML. Loaded as a plain <script> (globals) in the
// browser and via require() in tests/js/test_markdown.mjs.
(function (root) {
  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

  function escapeHtml(str) {
    return String(str ?? "").replace(/[&<>"']/g, (c) => ESC[c]);
  }
  const escapeAttr = escapeHtml; // quotes are escaped too, so attribute-safe

  // Models cite sources in several styles; normalise them all to "[n]":
  // [Source 2], [Source 2: title], 【Source 3】, *Source 1, Source 4*, (Sources 1 and 3).
  // \s+ rather than a space: some models emit U+202F inside "Source N".
  const CITE = /(?:[[【(]|\*{1,2})\s*Sources?\s+(\d+(?:\s*(?:,|and|&)\s*(?:Sources?\s+)?\d+)*)(?::[^\]】)*\n]*)?\s*(?:[\]】)]|\*{1,2})/g;
  function normalizeCitations(text) {
    return text.replace(CITE, (_, list) => list.match(/\d+/g).map((n) => `[${n}]`).join(""));
  }

  // PDF passages arrive hard-wrapped with end-of-line hyphenation; rejoin them for reading.
  function cleanPassage(s) {
    return String(s ?? "").replace(/-\n(?=[a-z])/g, "").replace(/\s*\n\s*/g, " ").trim();
  }

  function inline(text, ctx) {
    const keep = [];
    const stash = (html) => `\u0000${keep.push(html) - 1}\u0000`;
    let s = escapeHtml(text)
      .replace(/`([^`]+)`/g, (_, c) => stash(`<code>${c}</code>`))
      .replace(/\\\((.+?)\\\)|\\\[(.+?)\\\]/g, (_, a, b) => stash(`<code class="math">${a ?? b}</code>`))
      .replace(/\*\*(.+?)\*\*|__(.+?)__/g, (_, a, b) => `<strong>${a ?? b}</strong>`)
      .replace(/(^|[^*\w])\*(?!\s)([^*\n]+?)\*(?!\w)/g, "$1<em>$2</em>")
      .replace(/\[(\d+)\]/g, (m, n) => {
        n = Number(n);
        if (!ctx || n < 1 || n > ctx.count) return m; // only real sources become pills
        const style = ctx.colorFor ? ` style="${escapeAttr(ctx.colorFor(n))}"` : "";
        return `<button type="button" class="cite"${style} data-mid="${escapeAttr(ctx.msgId)}" data-n="${n}" aria-label="Source ${n}">${n}</button>`;
      });
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => keep[i]);
  }

  const ITEM = /^(\s*)([-*+•]|\d+[.)])\s+(.*)$/;
  const isRow = (l) => /^\s*\|.*\|\s*$/.test(l);
  const isSep = (l) => /^\s*\|?(\s*:?-{2,}:?\s*\|)+\s*(:?-{2,}:?\s*)?\|?\s*$/.test(l || "");
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const startsBlock = (l, next) => /^\s*```/.test(l) || /^#{1,6}\s/.test(l) || ITEM.test(l) || /^>\s?/.test(l) || (isRow(l) && isSep(next));

  /**
   * Render model markdown to HTML.
   * ctx: { msgId, count (number of sources), colorFor(n) -> inline style string }
   */
  function renderMarkdown(src, ctx) {
    const lines = normalizeCitations(String(src ?? "")).replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }

      // Fenced code (an unclosed fence while streaming renders as code up to the end)
      const fence = line.match(/^\s*```\s*([\w+-]*)/);
      if (fence) {
        const body = [];
        i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
        i++;
        const lang = fence[1] ? `<span class="code-lang">${escapeHtml(fence[1])}</span>` : "";
        out.push(`<div class="code-block"><div class="code-head">${lang}<button type="button" class="code-copy">Copy</button></div><pre><code>${escapeHtml(body.join("\n"))}</code></pre></div>`);
        continue;
      }

      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        const tag = h[1].length <= 2 ? "h3" : "h4"; // answers sit under the page's own h1/h2
        out.push(`<${tag}>${inline(h[2], ctx)}</${tag}>`);
        i++;
        continue;
      }

      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { out.push("<hr>"); i++; continue; }

      if (isRow(line) && isSep(lines[i + 1])) {
        const head = cells(line);
        i += 2;
        const rows = [];
        while (i < lines.length && isRow(lines[i])) rows.push(cells(lines[i++]));
        out.push(`<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th scope="col">${inline(c, ctx)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c, ctx)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
        continue;
      }

      if (/^>\s?/.test(line)) {
        const quote = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) quote.push(lines[i++].replace(/^>\s?/, ""));
        out.push(`<blockquote>${quote.map((q) => inline(q, ctx)).join("<br>")}</blockquote>`);
        continue;
      }

      const first = line.match(ITEM);
      if (first) {
        const ordered = /\d/.test(first[2]);
        const items = [];
        while (i < lines.length) {
          const m = lines[i].match(ITEM);
          if (m && /\d/.test(m[2]) === ordered) { items.push([m[3]]); i++; continue; }
          // Indented continuation line (e.g. "  *Source 1*" under a bullet)
          if (/^\s+\S/.test(lines[i]) && !m) { items[items.length - 1].push(lines[i].trim()); i++; continue; }
          // A blank line inside a list continues it if the next line is another item or indented text
          if (!lines[i].trim()) {
            let j = i;
            while (j < lines.length && !lines[j].trim()) j++;
            const n = lines[j] && lines[j].match(ITEM);
            if ((n && /\d/.test(n[2]) === ordered) || (lines[j] && /^\s+\S/.test(lines[j]) && !n)) { i = j; continue; }
          }
          break;
        }
        const tag = ordered ? "ol" : "ul";
        const start = ordered && parseInt(first[2], 10) !== 1 ? ` start="${parseInt(first[2], 10)}"` : "";
        out.push(`<${tag}${start}>${items.map((parts) => `<li>${parts.map((p) => inline(p, ctx)).join(" ")}</li>`).join("")}</${tag}>`);
        continue;
      }

      const para = [];
      while (i < lines.length && lines[i].trim() && !(para.length && startsBlock(lines[i], lines[i + 1]))) para.push(lines[i++]);
      out.push(`<p>${para.map((p) => inline(p, ctx)).join("<br>")}</p>`);
    }
    return out.join("");
  }

  const api = { escapeHtml, escapeAttr, normalizeCitations, cleanPassage, renderMarkdown };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof window !== "undefined" ? window : globalThis);
