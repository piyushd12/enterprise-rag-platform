"""
Patterns and text normalization shared by every guardrail layer.

Hand-rolled rather than a library: the PII/secret types required here are
all pattern-shaped (Presidio would pull in spaCy plus a ~500MB model for
the same regexes plus NER we don't need), and the injection checks are
phrase patterns run over a few normalized variants of the text. Everything
is pure and synchronous -- a query is scored in well under a millisecond.

Injection scoring is phrase-level by design. Enterprise documents and
legitimate questions use words like "ignore", "override" and "system
prompt" all the time, so no single word scores; a pattern only fires on a
directive-shaped phrase ("ignore all previous instructions", "reveal your
system prompt"). Each pattern carries a weight, a text's score is the sum
over distinct patterns, and the caller compares it to a threshold:
mentioning such a phrase in a question is a flag, issuing it is a block.

Every mask pattern is length-bounded (MAX_MATCH_LEN): the streaming output
filter holds back that many characters so a match can never straddle text
already sent. The one exception, a full PEM private-key block, is never
streamed -- the stream blocks on its header first.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Latin lookalikes used to dodge keyword matching ("іgnоrе" with Cyrillic
# і/о/е). Not a full confusables table -- just the letters that actually
# occur in the phrases below. Applied after casefold, so lowercase only.
_CONFUSABLES = str.maketrans(
    {
        "а": "a", "в": "b", "е": "e", "ѕ": "s", "і": "i", "ј": "j", "к": "k",
        "м": "m", "н": "h", "о": "o", "р": "p", "с": "c", "т": "t", "у": "y",
        "х": "x", "ԁ": "d", "һ": "h", "ӏ": "l", "ɡ": "g",
        "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p",
        "τ": "t", "υ": "u", "χ": "x",
    }
)
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")


def _visible(text: str) -> str:
    """NFKC, Unicode tag characters decoded to the ASCII they smuggle, other
    invisible format characters (zero-width spaces/joiners, bidi marks,
    soft hyphens) dropped, casefolded, lookalikes mapped to Latin."""
    out = []
    for ch in unicodedata.normalize("NFKC", text):
        cp = ord(ch)
        if 0xE0020 <= cp <= 0xE007E:
            out.append(chr(cp - 0xE0000))
        elif unicodedata.category(ch) != "Cf":
            out.append(ch)
    return "".join(out).casefold().translate(_CONFUSABLES)


def _decoded_base64(text: str) -> list[str]:
    """Printable-text decodings of base64 runs ("aWdub3Jl..." hiding a
    payload). Runs that don't decode to mostly-printable text are skipped,
    so ordinary long tokens never produce garbage variants."""
    decoded = []
    for run in _BASE64_RUN.findall(text):
        try:
            raw = base64.b64decode(run + "=" * (-len(run) % 4), validate=True)
            s = raw.decode("utf-8")
        except (binascii.Error, ValueError):
            continue
        if s and sum(c.isprintable() or c.isspace() for c in s) / len(s) > 0.9:
            decoded.append(s)
    return decoded


# Four or more single characters separated by spaces/punctuation:
# "i g n o r e", "i.g.n.o.r.e", "i-g-n-o-r-e".
_SPACED_RUN = re.compile(r"(?<!\w)(?:\w[\s.\-*_|/]{1,3}){3,}\w(?!\w)")


def _squash(text: str) -> str:
    """Letters and digits only: "i g n o r e  p r e v i o u s" and
    "i.g.n.o.r.e" both become "ignoreprevious"."""
    return re.sub(r"[^0-9a-z]", "", text)


def _despaced(text: str) -> tuple[str, list[str]]:
    """The text with each letter-spaced run collapsed in place (so a
    partly-spaced phrase reads normally), plus the squashed runs on their
    own (a fully-spaced phrase collapses into one word-boundary-free run,
    matched by the squashed patterns). Only spaced runs are squashed --
    squashing ordinary text would erase word boundaries and let "you are
    nowhere" read as "you are now"."""
    runs = [_squash(r) for r in _SPACED_RUN.findall(text)]
    return _SPACED_RUN.sub(lambda m: _squash(m.group(0)), text), runs


# ---------------------------------------------------------------------------
# Injection / jailbreak patterns
# ---------------------------------------------------------------------------

_OVERRIDE_VERB = r"(?:ignore|disregard|forget|override|bypass|skip)"
_PRIOR = (
    r"(?:previous|prior|above|earlier|preceding|foregoing|all(?:\s+(?:the|of\s+the|your))?"
    r"|your|system|original|initial|any)"
)
_RULES = (
    r"(?:instructions?|rules|prompts?|directions|guidelines|directives|programming"
    r"|guardrails|constraints|restrictions)"
)
_OVERRIDE = rf"\b{_OVERRIDE_VERB}\b[^.\n]{{0,30}}?\b{_PRIOR}\b[^.\n]{{0,20}}?\b{_RULES}\b"

# (name, weight, pattern). Weights: 1.0 blocks on its own at the default
# threshold; 0.5 only flags unless something else also fires.
_INJECTION_PATTERNS: list[tuple[str, float, str]] = [
    # Instruction override. Mentioning the phrase (a question about prompt
    # injection) flags; issuing it -- at the start of the text or of a
    # sentence/clause -- adds enough to block.
    ("override_mention", 0.5, _OVERRIDE),
    (
        "override_directive",
        1.0,
        rf"(?:^|[.!?:;\n]|\b(?:please|now|and|then|also|just)\b)\s*{_OVERRIDE}",
    ),
    # Aimed at the assistant's own rules ("forget your instructions") --
    # a question *about* prompt injection talks about "the model's" or
    # "its" instructions, not "your".
    (
        "override_yours",
        1.0,
        rf"\b{_OVERRIDE_VERB}\b[^.\n]{{0,15}}?\byour\b[^.\n]{{0,20}}?\b{_RULES}\b",
    ),
    ("new_instructions", 1.0, r"\b(?:new|updated|real|actual)\s+(?:instructions|rules|task|orders)\s*:"),
    ("your_new_role", 0.5, r"\byour\s+(?:new|real|true)\s+(?:instructions|rules|role|task|purpose|persona)\b"),
    # Role reassignment / personas.
    ("you_are_now", 0.5, r"\byou\s+are\s+(?:now|no\s+longer)\b"),
    ("from_now_on", 0.5, r"\bfrom\s+now\s+on\b[^.\n]{0,20}\byou\b"),
    ("pretend", 0.5, r"\b(?:pretend|imagine)\s+(?:to\s+be|you\s+are|that\s+you|you're)\b"),
    (
        "unrestricted_persona",
        1.0,
        (r"\b(?:act|behave|respond|roleplay|role-play|answer)\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?"
        r"(?:an?\s+)?(?:unrestricted|unfiltered|uncensored|jailbroken|evil|amoral|rogue)\b"),
    ),
    ("do_anything_now", 1.0, r"\bdo\s+anything\s+now\b"),
    (
        "special_mode",
        1.0,
        (r"\b(?:enable|enter|activate|switch\s+(?:to|into)|turn\s+on)\s+(?:the\s+)?"
        r"(?:developer|dev|dan|god|jailbreak|unrestricted|debug|admin)\s+mode\b"),
    ),
    (
        "no_restrictions",
        0.5,
        (r"\b(?:without|with\s+no|free\s+of|ignoring)\s+(?:any\s+)?"
        r"(?:restrictions|limitations|filters|censorship|guidelines|safety)\b"),
    ),
    # System-prompt extraction. Asking about "the system prompt" in general
    # is a flag (a legitimate question about the documents); asking for
    # *your* prompt is a block.
    (
        "extract_your_prompt",
        1.0,
        (r"\b(?:reveal|show|print|display|output|repeat|recite|share|tell\s+me|give\s+me|leak|dump"
        r"|expose|write\s+out|list|what\s+(?:is|are|was|were))\b[^.\n]{0,30}?\byour\s+"
        r"(?:system\s+|initial\s+|original\s+|hidden\s+|secret\s+|internal\s+|full\s+|exact\s+)*"
        r"(?:prompt|instructions|rules|guidelines|configuration)\b"),
    ),
    (
        "extract_the_prompt",
        0.5,
        (r"\b(?:reveal|show|print|display|output|repeat|recite|leak|dump|what\s+(?:is|are|was|were))\b"
        r"[^.\n]{0,30}?\bthe\s+(?:system|initial|original|hidden|secret|internal)\s+"
        r"(?:prompt|instructions|message)\b"),
    ),
    (
        "repeat_above",
        1.0,
        (r"\b(?:repeat|print|output|copy|recite)\b[^.\n]{0,25}?\b(?:everything|all|the\s+text"
        r"|the\s+words|what\s+is\s+written)\b[^.\n]{0,25}?\b(?:above|before\s+this|preceding)\b"),
    ),
    ("what_were_you_told", 1.0, r"\bwhat\s+(?:were|are|have)\s+you\s+(?:been\s+)?(?:told|instructed|programmed)\b"),
    # Spoofed roles and delimiters.
    (
        "delimiter_tag",
        1.0,
        r"<\s*/?\s*(?:system|assistant|documents?|conversation_history|instructions?)\s*>",
    ),
    ("chat_template_token", 1.0, r"\[/?(?:inst|sys)\]|<<\s*/?sys\s*>>|<\|(?:im_start|im_end|system|assistant|user|endoftext)\|>"),
    ("role_label", 0.5, r"(?:^|\n)\s*(?:system|assistant|developer)\s*:"),
    # Exfiltration via a rendered link/image carrying data out.
    ("markdown_exfil", 0.5, r"!\[[^\]]{0,40}\]\(\s*https?://[^)\s]{1,200}\?[^)\s]{1,200}\)"),
]

_COMPILED = [(name, weight, re.compile(p, re.IGNORECASE)) for name, weight, p in _INJECTION_PATTERNS]
# The same patterns with every whitespace/boundary token removed, matched
# against squashed letter-spaced runs (see _despaced). Anchors and
# punctuation don't survive squashing, so the position- and symbol-based
# patterns are left out.
_SQUASHED = [
    (name, weight, re.compile(re.sub(r"\\s[+*]?|\\b|\s", "", p), re.IGNORECASE))
    for name, weight, p in _INJECTION_PATTERNS
    if not name.endswith("_directive") and name not in ("role_label", "delimiter_tag", "chat_template_token", "markdown_exfil")
]

# Added once when a pattern only matches after de-obfuscation (leetspeak,
# base64, letter-spacing, invisible characters). Nobody writes a genuine
# question about prompt injection in base64.
OBFUSCATION_WEIGHT = 0.5


def injection_score(text: str) -> tuple[float, list[str]]:
    """Score `text` for prompt-injection/jailbreak phrasing.

    Returns (score, matched pattern names). A name suffixed "~obfuscated"
    means it only matched in a de-obfuscated variant.
    """
    plain = _visible(text)
    leet = plain.translate(_LEET)
    despaced, spaced_runs = _despaced(leet)
    obfuscated_variants = [leet, despaced] + [_visible(s) for s in _decoded_base64(text)]
    # The raw (pre-normalization) text, so invisible-character tricks count
    # as obfuscation too: if a pattern matches the normalized text but not
    # the raw one, something was hiding it.
    raw = text.casefold()

    score = 0.0
    hits: list[str] = []
    obfuscated = False
    for name, weight, rx in _COMPILED:
        if rx.search(plain):
            score += weight
            hits.append(name)
            obfuscated |= not rx.search(raw)
        elif any(rx.search(v) for v in obfuscated_variants):
            score += weight
            hits.append(f"{name}~obfuscated")
            obfuscated = True

    for name, weight, rx in _SQUASHED:
        if name in hits or f"{name}~obfuscated" in hits:
            continue
        if any(rx.search(run) for run in spaced_runs):
            score += weight
            hits.append(f"{name}~obfuscated")
            obfuscated = True

    if obfuscated:
        score += OBFUSCATION_WEIGHT
        hits.append("obfuscation")
    return round(score, 3), hits


# ---------------------------------------------------------------------------
# PII and secrets
# ---------------------------------------------------------------------------


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def _verhoeff_ok(digits: str) -> bool:
    """Aadhaar's check digit: without it, any 12-digit number would mask."""
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


# (label, pattern, validator). Order matters: earlier entries mask first,
# so a card number is never half-eaten by the phone pattern.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "PRIVATE KEY",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----[\s\S]*?(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|\Z)"
        ),
    ),
    ("SECRET", re.compile(r"\bgsk_[A-Za-z0-9]{20,120}")),
    ("SECRET", re.compile(r"\bsk-or-v1-[A-Za-z0-9]{20,120}")),
    ("SECRET", re.compile(r"\bsk-(?:proj-|ant-[a-z0-9]{2,8}-)?[A-Za-z0-9_-]{20,150}")),
    ("SECRET", re.compile(r"\blsv2_[A-Za-z0-9_]{20,120}")),
    ("SECRET", re.compile(r"\bpylf_v\d+_[A-Za-z0-9_]{20,120}")),
    ("SECRET", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("SECRET", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,120}")),
    ("SECRET", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,150}")),
    ("SECRET", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,120}")),
    ("SECRET", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    ("SECRET", re.compile(r"\beyJ[A-Za-z0-9_-]{8,70}\.eyJ[A-Za-z0-9_-]{8,70}\.[A-Za-z0-9_-]{8,70}")),
]
# `password = hunter2x9`-style assignments: only the value is masked, and
# only when it looks like a credential (has both a letter and a digit), so
# prose like "token = embedding" is left alone.
_ASSIGNED_SECRET = re.compile(
    r"\b(?P<key>api[_-]?key|secret(?:[_-]?key)?|access[_-]?key|client[_-]?secret|auth[_-]?token"
    r"|access[_-]?token|token|password|passwd|pwd)\b(?P<sep>\s*[:=]\s*[\"']?)"
    r"(?P<value>(?=[^\s\"']{0,150}\d)(?=[^\s\"']{0,150}[A-Za-z])[A-Za-z0-9_\-./+=]{8,150})",
    re.IGNORECASE,
)
PRIVATE_KEY_HEADER = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")

_PII_PATTERNS: list[tuple[str, re.Pattern[str], object]] = [
    # Contiguous, 4-4-4-x or Amex 4-6-5 -- with one consistent separator,
    # so a run like "1850-1900 2016-2025" in an answer is never a "card".
    (
        "CARD",
        re.compile(
            r"(?<![\d-])(?:\d{13,19}|\d{4}([ -])\d{4}\1\d{4}\1\d{1,7}|\d{4}([ -])\d{6}\2\d{5})(?![\d-])"
        ),
        lambda m: _luhn_ok(_digits(m)),
    ),
    (
        "AADHAAR",
        re.compile(r"(?<![\d-])[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?![\d-])"),
        lambda m: _verhoeff_ok(_digits(m)),
    ),
    ("SSN", re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])"), None),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,24}\b"), None),
    (
        "PHONE",
        re.compile(
            # +<country> <groups>, or (NNN) NNN-NNNN / NNN-NNN-NNNN / NNN.NNN.NNNN,
            # or an Indian mobile. Separators are required for the US forms
            # so bare figures in a financial answer never mask.
            r"(?<![\w+])\+\d{1,3}[ .-]?\(?\d{1,4}\)?(?:[ .-]?\d{2,5}){2,4}(?![\w-])"
            r"|(?<![\w(])\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\w-])"
            r"|(?<![\w-])[6-9]\d{4}[ -]?\d{5}(?![\w-])"
        ),
        None,
    ),
    ("PAN", re.compile(r"\b[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]\b"), None),
    (
        "NINO",
        re.compile(
            r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]"
            r" ?\d{2} ?\d{2} ?\d{2} ?[A-D]\b"
        ),
        None,
    ),
]

# Longest text any bounded mask pattern can match; the stream filter holds
# back at least this much. (The PEM block is unbounded but blocks the
# stream at its header instead -- see module docstring.)
MAX_MATCH_LEN = 250


def _sub(rx: re.Pattern[str], label: str, text: str, validator, found: list[str]) -> str:
    def repl(m: re.Match[str]) -> str:
        if validator is not None and not validator(m.group(0)):
            return m.group(0)
        found.append(label)
        return f"[{label}]"

    return rx.sub(repl, text)


def mask_secrets(text: str, found: list[str] | None = None) -> str:
    """Replace API keys, tokens and private keys with placeholders."""
    found = [] if found is None else found
    for label, rx in _SECRET_PATTERNS:
        text = _sub(rx, label, text, None, found)

    def assigned(m: re.Match[str]) -> str:
        found.append("SECRET")
        return f"{m.group('key')}{m.group('sep')}[SECRET]"

    return _ASSIGNED_SECRET.sub(assigned, text)


def mask_pii(text: str, found: list[str] | None = None) -> str:
    """Replace emails, phone numbers, card numbers and national IDs."""
    found = [] if found is None else found
    for label, rx, validator in _PII_PATTERNS:
        text = _sub(rx, label, text, validator, found)
    return text


def redact(text: str) -> str:
    """Secrets and PII both masked -- for logs and traces."""
    return mask_pii(mask_secrets(text))


def find_pii(text: str) -> list[str]:
    """Distinct PII labels present in text, in pattern order."""
    found: list[str] = []
    mask_pii(text, found)
    return list(dict.fromkeys(found))
