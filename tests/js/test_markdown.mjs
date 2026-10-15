// Run: node tests/js/test_markdown.mjs
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { renderMarkdown, normalizeCitations, cleanPassage } = require("../../src/rag_app/static/markdown.js");
const ctx = { msgId: "m1", count: 5, colorFor: (n) => `--c:var(--doc-${n})` };
const pills = (html) => [...html.matchAll(/class="cite"[^>]*data-n="(\d+)"/g)].map((m) => Number(m[1]));

// A real stored answer: "*" bullets with indented "*Source N*" continuation lines and \( \) math.
const real = `Residual connections help.

* **Reformulating layers** – a block learns \\(F(x)=H(x)-x\\).
  *Source 1*

* **Improved gradient flow** – gradients propagate directly.
  *Source 1, Source 4*

In short, they work.`;
const html = renderMarkdown(real, ctx);
assert.equal((html.match(/<ul>/g) || []).length, 1, "one list, not two");
assert.equal((html.match(/<li>/g) || []).length, 2);
assert.deepEqual(pills(html), [1, 1, 4]);
assert.ok(html.includes('<code class="math">F(x)=H(x)-x</code>'));
assert.ok(!html.replace(/<[^>]+>/g, "").includes("Source"), "citation text is replaced by pills");
assert.ok(html.includes('style="--c:var(--doc-4)"'));

// Other citation styles models use
assert.equal(normalizeCitations("a [Source 2: Attention paper] b 【Source 3】 (Sources 1 and 3)"), "a [2] b [3] [1][3]");

// Only real sources become pills
assert.deepEqual(pills(renderMarkdown("see [9] and [2]", ctx)), [2]);

// Tables, headings, ordered lists, code fences
assert.ok(renderMarkdown("| a | b |\n|---|---|\n| 1 | 2 |", ctx).includes("<th scope=\"col\">a</th>"));
assert.ok(renderMarkdown("## Title", ctx).startsWith("<h3>Title</h3>"));
assert.ok(renderMarkdown("1. a\n2. b", ctx).startsWith("<ol><li>a</li><li>b</li></ol>"));
const code = renderMarkdown("```python\nif a < b and x[1]:\n```", ctx);
assert.ok(code.includes("<pre><code>if a &lt; b and x[1]:</code></pre>"), "code is escaped and never gets pills");

// Underscored filenames are not italicised; HTML in model output is escaped
assert.ok(renderMarkdown("Deep_residual_learning_rp.pdf", ctx).includes("Deep_residual_learning_rp.pdf"));
const xss = renderMarkdown('<img src=x onerror="alert(1)"> **<script>**', ctx);
assert.ok(!/<img|<script/.test(xss) && xss.includes("&lt;script&gt;"));

assert.equal(cleanPassage("learn-\ning residual\nfunctions"), "learning residual functions");
console.log("markdown.js: all checks passed");
