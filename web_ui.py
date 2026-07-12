"""HTML for the SLM-125M playground. Kept out of modal_app.py to stay readable."""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLM 125M · Legal + Financial</title>
<style>
  :root {
    --ink:#18181b; --muted:#71717a; --faint:#a1a1aa;
    --line:#e4e4e7; --bg:#fafaf9; --panel:#fff;
    --accent:#1e3a5f; --warn-bg:#fffbeb; --warn-line:#fde68a; --warn-ink:#92400e;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--bg); color:var(--ink);
    font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Inter,sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  .wrap { max-width:820px; margin:0 auto; padding:56px 24px 80px; }

  header { border-bottom:1px solid var(--line); padding-bottom:28px; margin-bottom:28px; }
  h1 { margin:0; font-size:34px; font-weight:640; letter-spacing:-.02em; }
  h1 .dim { color:var(--faint); font-weight:400; }
  .sub { margin:8px 0 0; color:var(--muted); font-size:15px; }

  .specs {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(112px,1fr));
    gap:1px; background:var(--line); border:1px solid var(--line);
    border-radius:8px; overflow:hidden; margin-bottom:24px;
  }
  .spec { background:var(--panel); padding:12px 14px; }
  .spec dt { color:var(--muted); font-size:11px; text-transform:uppercase;
             letter-spacing:.06em; margin:0 0 3px; }
  .spec dd { margin:0; font-size:16px; font-weight:600;
             font-variant-numeric:tabular-nums; }

  .note {
    background:var(--warn-bg); border:1px solid var(--warn-line); color:var(--warn-ink);
    border-radius:8px; padding:11px 14px; font-size:13.5px; margin-bottom:28px;
  }
  .note b { font-weight:650; }

  label { display:block; font-size:13px; font-weight:600; margin-bottom:8px; }
  label .hint { font-weight:400; color:var(--muted); }

  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
  .chip {
    border:1px solid var(--line); background:var(--panel); color:var(--muted);
    border-radius:999px; padding:5px 11px; font-size:12.5px; cursor:pointer;
    font-family:inherit; transition:.12s;
  }
  .chip:hover { border-color:var(--accent); color:var(--accent); }

  textarea {
    width:100%; min-height:96px; resize:vertical; padding:14px;
    border:1px solid var(--line); border-radius:8px; background:var(--panel);
    color:var(--ink); font:15px/1.6 ui-monospace,"SF Mono",Menlo,monospace;
  }
  textarea:focus { outline:none; border-color:var(--accent);
                   box-shadow:0 0 0 3px rgba(30,58,95,.08); }

  .controls { display:flex; align-items:center; gap:20px; flex-wrap:wrap; margin:16px 0 0; }
  .ctl { display:flex; align-items:center; gap:9px; font-size:13px; color:var(--muted); }
  .ctl input[type=range] { width:104px; accent-color:var(--accent); }
  .ctl b { color:var(--ink); font-variant-numeric:tabular-nums; min-width:26px; }

  button.go {
    margin-left:auto; background:var(--accent); color:#fff; border:0;
    border-radius:8px; padding:11px 22px; font-size:14px; font-weight:600;
    font-family:inherit; cursor:pointer; transition:.12s;
  }
  button.go:hover:not(:disabled) { background:#16304e; }
  button.go:disabled { opacity:.5; cursor:not-allowed; }

  .out { margin-top:28px; }
  .outbox {
    border:1px solid var(--line); border-radius:8px; background:var(--panel);
    padding:18px; min-height:104px;
    font:15px/1.75 ui-monospace,"SF Mono",Menlo,monospace;
    white-space:pre-wrap; word-break:break-word;
  }
  .outbox .prompt { color:var(--faint); }
  .outbox .gen { color:var(--ink); background:#f0f9ff;
                 box-shadow:0 0 0 2px #f0f9ff; border-radius:2px; }
  .outbox .idle { color:var(--faint); font-family:ui-sans-serif,sans-serif; }
  .meta { margin-top:9px; font-size:12px; color:var(--faint);
          font-variant-numeric:tabular-nums; min-height:16px; }

  @keyframes pulse { 0%,100%{opacity:.35} 50%{opacity:1} }
  .loading { animation:pulse 1.1s ease-in-out infinite; }

  footer { margin-top:44px; padding-top:20px; border-top:1px solid var(--line);
           color:var(--faint); font-size:12.5px; }
  footer code { font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <h1>SLM 125M <span class="dim">· Legal + Financial</span></h1>
    <p class="sub">A base language model pretrained from scratch on 2.04B tokens.
       It <b>continues</b> text — it does not answer questions.</p>
  </header>

  <dl class="specs">
    <div class="spec"><dt>Params</dt><dd>125.8M</dd></div>
    <div class="spec"><dt>Layers</dt><dd>12</dd></div>
    <div class="spec"><dt>Vocab</dt><dd>16,384</dd></div>
    <div class="spec"><dt>Tokens</dt><dd>2.04B</dd></div>
    <div class="spec"><dt>Val PPL</dt><dd>10.93</dd></div>
    <div class="spec"><dt>Cost</dt><dd>~$18</dd></div>
  </dl>

  <div class="note">
    <b>Fabricated output.</b> This model learned the <i>shape</i> of legal and financial
    language, not the law. Case citations, dollar figures and dates it produces are
    invented and look convincing. Not a factual source.
  </div>

  <label for="p">Prompt <span class="hint">— an opening the model will continue</span></label>
  <div class="chips" id="chips"></div>
  <textarea id="p" spellcheck="false"></textarea>

  <div class="controls">
    <div class="ctl">
      <span>Temperature</span>
      <input type="range" id="t" min="0.1" max="1.5" step="0.1" value="0.8">
      <b id="tv">0.8</b>
    </div>
    <div class="ctl">
      <span>Max tokens</span>
      <input type="range" id="n" min="20" max="200" step="10" value="80">
      <b id="nv">80</b>
    </div>
    <button class="go" id="go">Generate</button>
  </div>

  <div class="out">
    <label>Completion <span class="hint">— prompt in grey, generated text highlighted</span></label>
    <div class="outbox" id="o"><span class="idle">Pick an example or write a prompt, then hit Generate.</span></div>
    <div class="meta" id="m"></div>
  </div>

  <footer>
    Base model · 40% case-law, 40% SEC filings, 20% fineweb-edu · 1 epoch on 4×H100.<br>
    API: <code>POST /complete {"prompt", "max_new_tokens", "temperature"}</code>
  </footer>

</div>

<script>
const EXAMPLES = [
  "The plaintiff alleges that the defendant",
  "The defendant moved to suppress the evidence on the grounds that",
  "In determining whether the search was reasonable, the court",
  "Pursuant to the terms of this Agreement,",
  "Item 7. Management's Discussion and Analysis of Financial Condition. Revenues",
  "The Company's net revenues for the fiscal year",
];

const $ = id => document.getElementById(id);
const box = $("o"), meta = $("m"), btn = $("go"), ta = $("p");

EXAMPLES.forEach(text => {
  const c = document.createElement("button");
  c.className = "chip";
  c.type = "button";
  c.textContent = text.length > 46 ? text.slice(0, 46) + "…" : text;
  c.title = text;
  c.onclick = () => { ta.value = text; ta.focus(); };
  $("chips").appendChild(c);
});

$("t").oninput = e => $("tv").textContent = e.target.value;
$("n").oninput = e => $("nv").textContent = e.target.value;

async function generate() {
  const prompt = ta.value.trim();
  if (!prompt) { ta.focus(); return; }

  btn.disabled = true;
  btn.textContent = "Generating…";
  box.innerHTML = '<span class="idle loading">The container may be cold — first request takes ~20s.</span>';
  meta.textContent = "";
  const t0 = performance.now();

  try {
    const r = await fetch("/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_new_tokens: +$("n").value,
        temperature: +$("t").value,
      }),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    if (d.error) throw new Error(d.error);

    box.innerHTML = "";
    const a = document.createElement("span");
    a.className = "prompt";
    a.textContent = d.prompt;
    const b = document.createElement("span");
    b.className = "gen";
    b.textContent = d.completion;
    box.append(a, b);

    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    meta.textContent = `${d.completion.trim().split(/\\s+/).length} words · ${secs}s`;
  } catch (e) {
    box.innerHTML = "";
    const s = document.createElement("span");
    s.className = "idle";
    s.textContent = "Request failed: " + e.message;
    box.appendChild(s);
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate";
  }
}

btn.onclick = generate;
ta.onkeydown = e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") generate();
};
</script>
</body>
</html>
"""
