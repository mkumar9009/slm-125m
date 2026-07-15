---
license: apache-2.0
language:
- en
base_model: thesreedath/slm-125m-base
pipeline_tag: text-generation
tags:
- legal
- grounded-qa
- rag
- instruction-tuned
---

# SLM-125M Legal SFT (grounded Q&A)

A **125M-parameter** small language model fine-tuned for **grounded question answering**
over legal and financial passages. Given a context passage and a question, it answers
**from the passage** or replies *"Not stated in the context."* when the passage does not
support an answer.

Fine-tuned from [`thesreedath/slm-125m-base`](https://huggingface.co/thesreedath/slm-125m-base).

> ⚠️ **Read the limitations before using this.** This is a research artifact. When it
> chooses to answer, it is **factually correct only ~23% of the time** and will state
> fabricated dates, figures, and case citations with full confidence. **Do not use it for
> any real legal, financial, or factual purpose.**

## What it does well vs. badly (measured, not claimed)

Evaluated on 746 held-out examples, judged by an LLM, with a base-model baseline and an
adversarial swapped-context probe.

| Metric | This model | Base model | Meaning |
|---|---:|---:|---|
| **Grounded refusal** (swapped context) | **83%** | – | Refuses when the passage does not support the question |
| Refusal recall | 79% | 0% | Refuses genuinely unanswerable questions |
| Hallucination rate | 21% | 100% | Answers something unanswerable *(lower is better)* |
| **Answer accuracy** | **23%** | 0% | Of questions it attempts, judged correct |
| False-refusal rate | 14% | 0% | Refuses an answerable question *(lower is better)* |
| Invented-figure rate | 1% | 14% | Invents a number absent from the passage *(lower is better)* |

**The honest one-liner:** it reliably knows *what it doesn't know* (83% grounded refusal),
but it is only right about **1 in 4** of the questions it does attempt. The first number is
solid; the second is capped by what a 125M model can extract and reason over.

## Intended use

- Research and teaching: a small, fully-reproducible grounded-QA / RAG pipeline.
- Studying refusal behaviour and context-grounding in tiny models.

**Not** for production, legal advice, financial analysis, or anything where a confidently
wrong answer causes harm.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("mkr79456/slm-125m-legal-sft")
model = AutoModelForCausalLM.from_pretrained("mkr79456/slm-125m-legal-sft")

context = "Main Place Funding Corporation was incorporated on June 24, 1994 in Delaware."
question = "On what date was Main Place Funding Corporation incorporated?"

messages = [
    {"role": "system", "content": "You are a careful legal and financial assistant. "
     "Answer only from the provided context. If the context does not contain the answer, "
     "say exactly: Not stated in the context."},
    {"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tok(prompt, return_tensors="pt")
# This tokenizer emits token_type_ids, which generate() rejects -- pass only these two.
ids = {k: enc[k] for k in ("input_ids", "attention_mask")}
out = model.generate(**ids, max_new_tokens=80, do_sample=False)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
# -> "Main Place Funding Corporation was incorporated on June 24, 1994."
```

The chat template is baked into the tokenizer. Special tokens: `<|bos|> <|eos|> <|pad|>
<|unk|> <|system|> <|user|> <|assistant|>`. Greedy decoding is recommended (this is
extraction, not creative generation).

## Training

- **Data:** 16,589 examples of grounded Q&A over legal/financial passages, synthesized by
  `gemini-3.1-flash-lite` from a deduplicated, decontaminated corpus (US case law + SEC
  filings + a little general web text). Teacher output was filtered by an LLM judge plus a
  deterministic check that each answer's supporting quote actually occurs in the passage.
- **Refusal balance:** ~27% of examples are refusals, including **swapped-context
  negatives** (an answerable question paired with an unrelated passage → refuse). These
  negatives are what lifted grounded refusal from 14% to 83%; without them the model
  answered from parametric memory instead of reading the passage.
- **Method:** full fine-tuning (all 125M params), 3 epochs, lr 2e-5 cosine, 1×H100, ~5 min.
  Loss masked on the prompt — only the assistant turn is supervised.

## Limitations

- **~23% answer accuracy.** It excels at verbatim lookups ("incorporated on June 24, 1994")
  and fails at anything requiring reasoning or careful reading.
- **Confabulates when it answers.** 74% of its wrong answers assert something the passage
  does not say. It has invented case citations, dates, and dollar amounts.
- **Attribution errors:** it confuses *who did what* (e.g. plaintiff vs. defendant).
- **English only. 1024-token context.** Long passages are truncated.
- Its knowledge and blind spots mirror its teacher (`gemini-3.1-flash-lite`), since the
  training data was synthetic.

## License

Apache-2.0.
