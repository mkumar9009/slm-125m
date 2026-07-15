"""Phase 7a: grounded (RAFT-style) SFT data — prompts, parsing, deterministic filters.

Pure functions only, so they can be unit-tested locally without Modal or a Gemini key.

The teacher and the judge are the same model (gemini-2.5-flash), which means the judge
is biased toward accepting its own output. Two mechanisms compensate, neither of which
trusts the judge's verdict:

  1. the judge must return a VERBATIM span from the passage as evidence, and
  2. `is_grounded` checks that span really occurs in the passage.

A judge that rubber-stamps a hallucinated answer still has to invent a quote, and the
string check catches the invented quote.
"""

from __future__ import annotations

import json
import re

# --------------------------------------------------------------------------- #
# Chat format. thesreedath/slm-125m-base ships <|user|>/<|assistant|>/<|system|>
# in its vocab but NO chat_template, so we define one and train the model into it.
# --------------------------------------------------------------------------- #

SYSTEM = (
    "You are a careful legal and financial assistant. Answer only from the provided "
    "context. If the context does not contain the answer, say exactly: "
    "Not stated in the context."
)
REFUSAL = "Not stated in the context."

CHAT_TEMPLATE = (
    # Must open with <|bos|>: every training example did, and without it the model sees
    # an out-of-distribution prompt and false-refuses answerable questions.
    "<|bos|>"
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}<|system|>{{ m['content'] }}<|eos|>"
    "{% elif m['role'] == 'user' %}<|user|>{{ m['content'] }}<|eos|>"
    "{% elif m['role'] == 'assistant' %}<|assistant|>{{ m['content'] }}<|eos|>"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def render_prompt(context: str, question: str) -> str:
    """The user turn: context then question. Must match what inference sends."""
    return f"<context>\n{context}\n</context>\n\nQuestion: {question}"


def to_record(context: str, question: str, answer: str, answerable: bool,
              source: str, task: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": render_prompt(context, question)},
            {"role": "assistant", "content": answer},
        ],
        "answerable": answerable,
        "source": source,
        "task": task,
    }


# --------------------------------------------------------------------------- #
# Passage chunking. Corpus docs are far too long for a 1024-token context
# (case-law averages 11.5k chars, SEC 95k), so cut them at paragraph boundaries.
# --------------------------------------------------------------------------- #

MIN_CHARS, MAX_CHARS = 900, 2_200


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split on blank lines, greedily packing paragraphs up to max_chars."""
    chunks: list[str] = []
    buf = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 2 > max_chars:
            if len(buf) >= MIN_CHARS:
                chunks.append(buf)
            buf = para[:max_chars]
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if len(buf) >= MIN_CHARS:
        chunks.append(buf)
    return chunks


# --------------------------------------------------------------------------- #
# Task mix. Weights follow the Group B recipes: grounded QA carries the load,
# refusals teach the model to decline rather than confabulate.
# --------------------------------------------------------------------------- #

TASK_MIX: dict[str, float] = {
    "qa": 0.70,          # 3 answerable + 1 refusal per passage
    "summarize": 0.12,
    "extract": 0.10,
    "rewrite": 0.08,
}

_JSON_CONTRACT = (
    'Return ONLY a JSON array. Each element must be exactly:\n'
    '{"question": str, "answer": str, "answerable": bool}\n'
    "No markdown, no prose, no code fences."
)

_QA = """You are building a supervised fine-tuning set from a legal/financial passage.

Write EXACTLY 4 items:
  - 3 questions ANSWERABLE from the passage, with "answerable": true. Vary the difficulty:
      1 easy lookup (a fact stated directly),
      1 multi-step (requires combining two or more statements),
      1 inference (requires reasoning about what the passage implies).
  - 1 question that is PLAUSIBLE for this passage but NOT answerable from it, with
    "answerable": false and the answer EXACTLY: "Not stated in the context."

Hard rules:
  - Answer ONLY from the passage. Invent nothing. No outside knowledge.
  - Every answerable answer must be directly supported by specific words in the passage.
  - Do NOT repeat the same question twice, or reword one question into another.
  - Vary the grammar: do not begin every question with "What". Use How / Why / Under what
    circumstances / Who / On what basis / Did ... , etc.
  - A question must be self-contained: never write "in this passage" or "according to the text".
  - Answers: 1-3 sentences, complete, no truncation.

""" + _JSON_CONTRACT

_SUMMARIZE = """Summarize the legal/financial passage below.

Write EXACTLY 2 items, each with "answerable": true:
  - one asking for a one-sentence summary,
  - one asking for the key points as a short bullet list.

Hard rules:
  - Compress only what is present. Invent no facts, figures, dates or holdings.
  - Phrase the two "question" fields as natural instructions, worded differently.

""" + _JSON_CONTRACT

_EXTRACT = """Convert the legal/financial passage below into structured data.

Write EXACTLY 2 items, each with "answerable": true:
  - one instructing extraction of the key entities/figures as a JSON object,
  - one instructing classification (e.g. document type, or the disposition/outcome),
    with a short justification.

Hard rules:
  - Extract only values that literally appear in the passage. Never guess a missing field;
    omit it instead.
  - The "answer" for the extraction item must itself be valid JSON, as a string.

""" + _JSON_CONTRACT

_REWRITE = """Rewrite the legal/financial passage below.

Write EXACTLY 2 items, each with "answerable": true:
  - one asking for a plain-English rewrite for a non-specialist,
  - one asking for a rewrite in a different register (e.g. a formal client memo).

Hard rules:
  - Preserve meaning exactly. Add no facts and drop no material qualifiers.
  - Keep each rewritten answer under 150 words.

""" + _JSON_CONTRACT

TASK_PROMPT = {
    "qa": _QA,
    "summarize": _SUMMARIZE,
    "extract": _EXTRACT,
    "rewrite": _REWRITE,
}

PAIRS_PER_TASK = {"qa": 4, "summarize": 2, "extract": 2, "rewrite": 2}


def gen_prompt(task: str, passage: str) -> str:
    return f"{TASK_PROMPT[task]}\n\nPASSAGE:\n\"\"\"\n{passage}\n\"\"\"\n"


# --------------------------------------------------------------------------- #
# Judge. Demanding a verbatim span is the whole point: a verdict alone is cheap
# for a self-preferring judge to hand out, a real quote is not.
# --------------------------------------------------------------------------- #

JUDGE_PROMPT = """You are a strict grader. For each item, decide whether the ANSWER is
fully supported by the PASSAGE.

For each item return:
  {"verdict": "PASS" or "FAIL",
   "evidence": "<a VERBATIM span copied character-for-character from the PASSAGE that
                supports the answer, or \\"\\" if the item is a refusal>",
   "reason": "<one short sentence>"}

Grading rules:
  - PASS an answerable item ONLY if every claim in it is supported by the passage, and you
    can copy an exact supporting span into "evidence". If you cannot find such a span, FAIL.
  - Do NOT invent, paraphrase, normalize or repair the evidence span. Copy it exactly.
  - An answer that is correct in the real world but NOT stated in the passage is a FAIL.
  - For a refusal item (answer is "Not stated in the context."): PASS only if the passage
    genuinely does not answer the question. Set "evidence" to "".
  - Also FAIL: truncated answers, empty answers, or answers that just restate the question.

Return ONLY a JSON array with one object per item, in order. No markdown, no code fences.

PASSAGE:
\"\"\"
__PASSAGE__
\"\"\"

ITEMS:
__ITEMS__
"""


def judge_prompt(passage: str, pairs: list[dict]) -> str:
    # str.replace, not .format: the prompt is full of literal JSON braces.
    items = json.dumps(
        [{"i": i, "question": p["question"], "answer": p["answer"]}
         for i, p in enumerate(pairs)],
        indent=1,
    )
    return (JUDGE_PROMPT
            .replace("__PASSAGE__", passage)
            .replace("__ITEMS__", items))


# --------------------------------------------------------------------------- #
# Parsing + deterministic filters (no LLM in the loop)
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def parse_json_array(raw: str) -> list[dict]:
    """Tolerate stray code fences / prose around the JSON the model was told not to add."""
    txt = _FENCE.sub("", raw.strip())
    try:
        out = json.loads(txt)
    except json.JSONDecodeError:
        start, end = txt.find("["), txt.rfind("]")
        if start == -1 or end <= start:
            return []
        try:
            out = json.loads(txt[start : end + 1])
        except json.JSONDecodeError:
            return []
    return [x for x in out if isinstance(x, dict)] if isinstance(out, list) else []


def parse_pairs(raw: str) -> list[dict]:
    """Q&A items with the three required fields, coerced to the right types."""
    pairs = []
    for x in parse_json_array(raw):
        q, a = str(x.get("question", "")).strip(), str(x.get("answer", "")).strip()
        if q and a:
            pairs.append({"question": q, "answer": a,
                          "answerable": bool(x.get("answerable", True))})
    return pairs


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip()


def format_ok(question: str, answer: str, answerable: bool, task: str = "qa") -> bool:
    """Structural gate: length, truncation, echoing, refusal wording.

    Task-aware on purpose. A QA item must be a self-contained question. A summarize /
    extract / rewrite item is an *instruction* -- it has no "?", and it legitimately
    refers to "the passage". Applying the QA rules to those tasks rejects all of them.
    """
    if not (10 <= len(question) <= 400) or not (2 <= len(answer) <= 4_000):
        return False
    if answer.rstrip().endswith((",", ";", "-", "—")):   # truncated mid-clause
        return False
    if _norm(answer) == _norm(question):
        return False
    if not answerable and _norm(answer) != _norm(REFUSAL):
        return False
    if answerable and _norm(answer) == _norm(REFUSAL):
        return False

    if task == "qa":
        if not question.endswith("?"):
            return False
        # A QA question must stand alone: "according to the passage" is a training-time
        # artifact that will never appear at inference.
        if re.search(r"\b(this|the) (passage|text|context|document|excerpt)\b",
                     question, re.I):
            return False
    return True


def is_grounded(evidence: str, passage: str, min_span: int = 24) -> bool:
    """Does the judge's quoted span actually occur in the passage?

    This is the anti-self-preference check for extractive QA. The judge is the same model
    that wrote the answer, so its PASS verdict is not trustworthy on its own -- but a
    fabricated quote will not be found here. Whitespace is normalized because models
    silently reflow it; nothing else is forgiven.
    """
    ev = _norm(evidence)
    if len(ev) < min_span:          # a 3-word "quote" supports nothing
        return False
    return ev in _norm(passage)


# --------------------------------------------------------------------------- #
# Evaluation judge (Phase 7c). Scores OUR model's answer, not the judge's own -- so
# unlike the generation judge this is not a self-preference problem. It IS still the
# model that wrote the gold answers, hence the explicit "equivalence, not style" rule:
# otherwise it rewards its own phrasing and penalises a correct answer worded differently.
# --------------------------------------------------------------------------- #

EVAL_JUDGE_PROMPT = """You are grading a small model's answer to a question about a passage.

Return ONLY this JSON object, no markdown:
{"verdict": "CORRECT" | "PARTIAL" | "WRONG", "grounded": true | false, "reason": "<one short sentence>"}

verdict -- compare the MODEL ANSWER to the GOLD ANSWER on FACTS ONLY:
  CORRECT : states the same facts as the gold answer. Different wording, extra harmless
            detail, or a shorter form are all still CORRECT. Judge substance, NOT style,
            length, or phrasing.
  PARTIAL : some correct facts, but incomplete or with one minor factual slip.
  WRONG   : contradicts the gold answer, names the wrong party/date/amount, or is
            irrelevant. Confusing WHO did WHAT (e.g. plaintiff vs defendant) is WRONG,
            not PARTIAL.

grounded -- is every claim in the MODEL ANSWER supported by the PASSAGE?
  false if it asserts anything the passage does not say, even if it happens to be true.

PASSAGE:
\"\"\"
__PASSAGE__
\"\"\"

QUESTION: __QUESTION__
GOLD ANSWER: __GOLD__
MODEL ANSWER: __PRED__
"""


def eval_judge_prompt(passage: str, question: str, gold: str, pred: str) -> str:
    return (EVAL_JUDGE_PROMPT
            .replace("__PASSAGE__", passage)
            .replace("__QUESTION__", question)
            .replace("__GOLD__", gold)
            .replace("__PRED__", pred))


def is_refusal(text: str) -> bool:
    """The model refused. Substring, not equality: it sometimes pads the phrase."""
    return _norm(REFUSAL).rstrip(".") in _norm(text)


_FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _figures(text: str) -> set[str]:
    """Numbers worth checking. Single digits are skipped: they are list markers
    ("1.", "2.") far more often than they are facts."""
    out = set()
    for m in _FIGURE.findall(text):
        norm = m.replace(",", "").rstrip(".")
        if len(norm.replace(".", "")) >= 2:
            out.add(norm)
    return out


def no_invented_figures(answer: str, passage: str) -> bool:
    """Every number in the answer must appear in the passage.

    Summaries, extractions and rewrites transform the whole passage, so no single span
    can support them and `is_grounded` does not apply. That leaves the judge unchecked --
    and the judge wrote the answer. This is the deterministic backstop: fabricated dates
    and dollar figures are the costliest hallucination in legal/financial text, and they
    are trivially detectable.
    """
    return _figures(answer) <= _figures(passage)
