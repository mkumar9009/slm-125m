"""HTML for the SLM-125M Legal SFT playground. Grounded QA: context + question -> answer."""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLM 125M · Legal Grounded QA</title>
<style>
  :root {
    --ink:#18181b; --muted:#71717a; --faint:#a1a1aa;
    --line:#e4e4e7; --bg:#fafaf9; --panel:#fff;
    --accent:#1e3a5f; --warn-bg:#fffbeb; --warn-line:#fde68a; --warn-ink:#92400e;
    --answer-bg:#f0f9ff; --refuse-bg:#f4f4f5;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--bg); color:var(--ink);
    font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Inter,sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  .wrap { max-width:820px; margin:0 auto; padding:56px 24px 80px; }

  header { border-bottom:1px solid var(--line); padding-bottom:28px; margin-bottom:24px; }
  h1 { margin:0; font-size:32px; font-weight:640; letter-spacing:-.02em; }
  h1 .dim { color:var(--faint); font-weight:400; }
  .sub { margin:8px 0 0; color:var(--muted); font-size:15px; }

  .specs {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(104px,1fr));
    gap:1px; background:var(--line); border:1px solid var(--line);
    border-radius:8px; overflow:hidden; margin-bottom:22px;
  }
  .spec { background:var(--panel); padding:11px 13px; }
  .spec dt { color:var(--muted); font-size:11px; text-transform:uppercase;
             letter-spacing:.06em; margin:0 0 3px; }
  .spec dd { margin:0; font-size:15px; font-weight:600; font-variant-numeric:tabular-nums; }

  .note {
    background:var(--warn-bg); border:1px solid var(--warn-line); color:var(--warn-ink);
    border-radius:8px; padding:11px 14px; font-size:13.5px; margin-bottom:26px;
  }
  .note b { font-weight:650; }

  label { display:block; font-size:13px; font-weight:600; margin:0 0 8px; }
  label .hint { font-weight:400; color:var(--muted); }

  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
  .chip {
    border:1px solid var(--line); background:var(--panel); color:var(--muted);
    border-radius:999px; padding:5px 11px; font-size:12.5px; cursor:pointer;
    font-family:inherit; transition:.12s;
  }
  .chip:hover { border-color:var(--accent); color:var(--accent); }

  textarea, input.q {
    width:100%; padding:13px; border:1px solid var(--line); border-radius:8px;
    background:var(--panel); color:var(--ink);
    font:15px/1.6 ui-monospace,"SF Mono",Menlo,monospace;
  }
  textarea { min-height:150px; resize:vertical; }
  input.q { font-family:inherit; margin-bottom:0; }
  textarea:focus, input.q:focus { outline:none; border-color:var(--accent);
                                  box-shadow:0 0 0 3px rgba(30,58,95,.08); }
  .field { margin-bottom:16px; }

  .row { display:flex; align-items:center; gap:14px; margin-top:4px; }
  button.go {
    margin-left:auto; background:var(--accent); color:#fff; border:0;
    border-radius:8px; padding:11px 24px; font-size:14px; font-weight:600;
    font-family:inherit; cursor:pointer; transition:.12s;
  }
  button.go:hover:not(:disabled) { background:#16304e; }
  button.go:disabled { opacity:.5; cursor:not-allowed; }

  .out { margin-top:26px; }
  .answerbox {
    border:1px solid var(--line); border-radius:8px; padding:16px 18px; min-height:60px;
    font:15px/1.7 ui-sans-serif,sans-serif; white-space:pre-wrap; word-break:break-word;
    background:var(--panel);
  }
  .answerbox.answered { background:var(--answer-bg); border-color:#bae6fd; }
  .answerbox.refused  { background:var(--refuse-bg); color:var(--muted); }
  .answerbox .idle { color:var(--faint); }
  .tag { display:inline-block; font-size:11px; font-weight:600; text-transform:uppercase;
         letter-spacing:.05em; padding:2px 8px; border-radius:4px; margin-bottom:8px; }
  .tag.answered { background:#0369a1; color:#fff; }
  .tag.refused  { background:#a1a1aa; color:#fff; }
  .meta { margin-top:8px; font-size:12px; color:var(--faint); font-variant-numeric:tabular-nums; }

  @keyframes pulse { 0%,100%{opacity:.35} 50%{opacity:1} }
  .loading { animation:pulse 1.1s ease-in-out infinite; }

  footer { margin-top:40px; padding-top:20px; border-top:1px solid var(--line);
           color:var(--faint); font-size:12.5px; }
  footer code { font-size:12px; color:var(--muted); }
  footer a { color:var(--muted); }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <h1>SLM 125M <span class="dim">· Legal Grounded QA</span></h1>
    <p class="sub">Give it a passage and a question. It answers <b>from the passage</b>,
       or refuses when the passage doesn't contain the answer.</p>
  </header>

  <dl class="specs">
    <div class="spec"><dt>Params</dt><dd>125.8M</dd></div>
    <div class="spec"><dt>Grounded refusal</dt><dd>83%</dd></div>
    <div class="spec"><dt>Answer acc.</dt><dd>23%</dd></div>
    <div class="spec"><dt>Base</dt><dd>thesreedath</dd></div>
    <div class="spec"><dt>Tuning</dt><dd>full SFT</dd></div>
  </dl>

  <div class="note">
    <b>Research demo — do not trust the answers.</b> When this model answers, it is
    factually correct only about <b>23%</b> of the time, and it states fabricated dates,
    figures and citations confidently. It is good at <i>refusing</i> when the passage
    lacks the answer; it is not good at being right. Not for real legal or financial use.
  </div>

  <div class="field">
    <label>Context passage <span class="hint">— the text the answer must come from</span></label>
    <div class="chips" id="chips"></div>
    <textarea id="ctx" spellcheck="false"></textarea>
  </div>

  <div class="field">
    <label>Question</label>
    <input class="q" id="q" spellcheck="false" placeholder="Ask something answerable from the passage — or not, to test refusal.">
    <div class="row">
      <button class="go" id="go">Answer</button>
    </div>
  </div>

  <div class="out">
    <label>Model output</label>
    <div class="answerbox" id="o"><span class="idle">Pick an example or paste your own passage, add a question, and hit Answer.</span></div>
    <div class="meta" id="m"></div>
  </div>

  <footer>
    Fine-tuned from <code>thesreedath/slm-125m-base</code> ·
    <a href="https://huggingface.co/mkr79456/slm-125m-legal-sft">model on HuggingFace</a><br>
    API: <code>POST /answer {"context", "question"}</code>
  </footer>

</div>

<script>
const EXAMPLES = [
  {
    label: "SEC filing — answerable",
    ctx: "Main Place Funding Corporation (MPFC) was incorporated on June 24, 1994 in the State of Delaware. MPFC is a wholly owned subsidiary of the Bank and was organized for the purpose of financing mortgage assets.",
    q: "Where was Main Place Funding Corporation incorporated?"
  },
  {
    label: "Case law — answerable",
    ctx: "The plaintiff filed suit on March 3, 1998. The district court granted summary judgment for the defendant, holding that the two-year statute of limitations had run. On appeal, the Ninth Circuit reversed, finding that equitable tolling applied because the defendant had concealed the injury.",
    q: "Why did the Ninth Circuit reverse the district court?"
  },
  {
    label: "Refusal — answer absent",
    ctx: "The Company's net revenues for the fiscal year ended December 31, 2003 were $412.6 million, an increase of 8% over the prior year, driven primarily by growth in the Consumer Products segment.",
    q: "Who is the Chief Executive Officer of the Company?"
  },
];

const $ = id => document.getElementById(id);
const box = $("o"), meta = $("m"), btn = $("go");

EXAMPLES.forEach(ex => {
  const c = document.createElement("button");
  c.className = "chip"; c.type = "button"; c.textContent = ex.label;
  c.onclick = () => { $("ctx").value = ex.ctx; $("q").value = ex.q; $("q").focus(); };
  $("chips").appendChild(c);
});

async function answer() {
  const context = $("ctx").value.trim(), question = $("q").value.trim();
  if (!context) { $("ctx").focus(); return; }
  if (!question) { $("q").focus(); return; }

  btn.disabled = true; btn.textContent = "Thinking…";
  box.className = "answerbox";
  box.innerHTML = '<span class="idle loading">The container may be cold — first request takes ~20s.</span>';
  meta.textContent = "";
  const t0 = performance.now();

  try {
    const r = await fetch("/answer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context, question }),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    if (d.error) throw new Error(d.error);

    box.className = "answerbox " + (d.refused ? "refused" : "answered");
    const tag = document.createElement("span");
    tag.className = "tag " + (d.refused ? "refused" : "answered");
    tag.textContent = d.refused ? "refused" : "answered";
    const body = document.createElement("div");
    body.textContent = d.answer;
    box.innerHTML = ""; box.append(tag, body);

    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    meta.textContent = (d.refused
      ? "Refused — the model judged the passage doesn't answer this."
      : "Answered — remember: ~23% accurate, may be confidently wrong.") + ` · ${secs}s`;
  } catch (e) {
    box.className = "answerbox";
    box.innerHTML = "";
    const s = document.createElement("span"); s.className = "idle";
    s.textContent = "Request failed: " + e.message;
    box.appendChild(s);
  } finally {
    btn.disabled = false; btn.textContent = "Answer";
  }
}

btn.onclick = answer;
$("q").addEventListener("keydown", e => { if (e.key === "Enter") answer(); });
</script>
</body>
</html>
"""
