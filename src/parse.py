"""Answer extraction, answer equivalence and correctness for each task.

extract(dataset, text) -> canonical answer string, or None if unparseable.
equiv(dataset, a, b)   -> whether two canonical answers count as the same answer (used for agreement).
correct(dataset, a, gold) -> bool.
"""
import re
import string
import unicodedata

# ---------------------------------------------------------------- generic text normalisation (SQuAD style)

def norm_text(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # strip accents: fabergé -> faberge
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _sing(s):
    """Crude singularisation so 'eggs' == 'egg'."""
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in s.split())


def _contains(a, b):
    """Token-boundary containment of b in a (both normalised, non-empty)."""
    return bool(a) and bool(b) and f" {b} " in f" {a} "

# ---------------------------------------------------------------- numbers / LaTeX

def _num(s):
    s = s.replace(",", "").replace("$", "").strip()
    s = re.sub(r"\\%|%", "", s)
    try:
        v = float(s)
        return v
    except ValueError:
        return None


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        i = text.rfind("\\fbox")
        if i < 0:
            return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def norm_latex(s):
    """Light normalisation of a MATH answer (after Hendrycks et al. / lm-eval 'is_equiv')."""
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(mbox|mathrm|textbf)\{([^}]*)\}", r"\2", s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\$", ""),
                 ("\\%", ""), ("%", ""), ("\\displaystyle", "")]:
        s = s.replace(a, b)
    s = re.sub(r"(degrees|degree|units|unit|cm|inches|inch|meters|dollars|cents)\b", "", s)
    s = s.replace(" ", "")
    s = re.sub(r"^[a-zA-Z]\s*=", "", s)  # "x=5" -> "5"
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)
    if s.endswith("."):
        s = s[:-1]
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    if s.startswith("0.") is False and re.fullmatch(r"-?\.\d+", s):
        s = s.replace(".", "0.", 1)
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m:  # canonical simple fraction a/b
        s = f"{m.group(1)}{m.group(2)}/{m.group(3)}"
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m and int(m.group(2)) != 0:
        v = int(m.group(1)) / int(m.group(2))
        s = str(int(v)) if v == int(v) else f"{m.group(1)}/{m.group(2)}"
    s = s.replace(",\\!", "").replace("{,}", "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    return s

# ---------------------------------------------------------------- extraction

def extract(dataset, text):
    t = text.strip()
    if dataset == "mmlu":
        m = re.search(r"\b([ABCD])\b", t) or re.search(r"([ABCD])[\.\):]", t)
        return m.group(1) if m else None
    if dataset == "triviaqa":
        t = t.split("\n")[0].strip()
        t = re.sub(r"^(answer|the answer is)\s*[:\-]?\s*", "", t, flags=re.I)
        return norm_text(t) or None
    if dataset == "gsm8k":
        m = re.findall(r"answer is[:\s]*\**\s*\$?\s*(-?[\d,]*\.?\d+)", t, flags=re.I)
        if not m:
            b = last_boxed(t)
            m = re.findall(r"-?[\d,]*\.?\d+", b) if b else re.findall(r"-?[\d,]*\.?\d+", t)
        if not m:
            return None
        v = _num(m[-1])
        if v is None:
            return None
        return str(int(v)) if v == int(v) else str(round(v, 6))
    if dataset == "math500":
        b = last_boxed(t)
        if b is None:  # fallback: "The (final) answer is: X"
            m = re.findall(r"answer is[:\s]*\$?([^\n$]+?)\$?\s*\.?\s*$", t, flags=re.I | re.M)
            b = m[-1] if m else None
        return norm_latex(b) if b is not None else None
    if dataset == "ethics_cs":
        w = t.lower()
        hits = [(w.find(k), v) for k, v in [("wrong", "wrong"), ("acceptable", "acceptable"),
                                            ("not wrong", "acceptable")] if k in w]
        if not hits:
            return None
        if "not wrong" in w and w.find("not wrong") <= min(h[0] for h in hits) + 0:
            return "acceptable"
        return min(hits)[1]
    if dataset == "ethics_justice":
        w = t.lower()
        if "unreasonable" in w or "not reasonable" in w:
            return "unreasonable"
        if "reasonable" in w:
            return "reasonable"
        return None
    if dataset == "mmlu_tf":
        w = t.lower()
        hits = [(w.find(k), k) for k in ("true", "false", "incorrect", "correct") if k in w]
        if not hits:
            return None
        k = min(hits)[1]
        return {"true": "true", "correct": "true", "false": "false", "incorrect": "false"}[k]
    raise ValueError(dataset)

# ---------------------------------------------------------------- equivalence & correctness

def equiv(dataset, a, b):
    if a is None or b is None:
        return False
    if dataset == "triviaqa":
        # Exact match after normalisation, or containment when the shorter answer is specific (>= 2 tokens).
        # (Containment of 1-token answers is not used: "eggs" would otherwise match every "... eggs" answer and
        # make the relation highly non-transitive.)
        a, b = _sing(a), _sing(b)
        if a == b:
            return True
        short, long_ = (a, b) if len(a.split()) <= len(b.split()) else (b, a)
        return len(short.split()) >= 2 and _contains(long_, short)
    if dataset in ("gsm8k", "math500"):
        if a == b:
            return True
        x, y = _num(a), _num(b)
        return x is not None and y is not None and abs(x - y) < 1e-6
    return a == b


def correct(dataset, a, gold):
    if a is None:
        return False
    if dataset == "triviaqa":
        golds = {_sing(norm_text(g)) for g in gold}
        a = _sing(a)
        return any(a == g or _contains(a, g) for g in golds if g)
    if dataset == "math500":
        return equiv(dataset, a, norm_latex(gold))
    if dataset == "gsm8k":
        return equiv(dataset, a, gold)
    return a == gold
