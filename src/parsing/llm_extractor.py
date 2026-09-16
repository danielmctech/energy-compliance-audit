"""
llm legal-structure extraction via ollama (qwen3.8:27b / deepseek-r1:32b).
same prompt for all models -> fair comparison. cached by (model, text-hash) so
evaluation reruns are free.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import common as c
    from common import extract_json, ollama_chat, stable_hash
except ImportError:  # allow use as top-level script too
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import common as c
    from common import extract_json, ollama_chat, stable_hash

STRUCTURE_PROMPT = """You are an expert EU energy-law structure extractor.
Given the document text below, return ONLY a valid JSON object (no prose, no markdown) with exactly these keys:
{
  "document_title": string|null,
  "instrument": string|null,              // e.g. "Regulation (EU) 2016/679"
  "celex": string|null,
  "publication_date": "YYYY-MM-DD"|null,
  "entry_into_force_date": "YYYY-MM-DD"|null,
  "recitals_count": int,
  "articles": [ {"number": "e.g. 28 or 28a", "title": string|null, "paragraphs_count": int} ],
  "chapters": [ {"number": string, "title": string|null} ],
  "defs": [ {"term": string, "definition_head": string} ],        // up to 10
  "refs": [ {"target": string, "context": string} ],              // up to 15 cited instruments/articles
  "tables_count": int,
  "obligations_count": int,
  "derogations_count": int
}
Rules:
1. Report ONLY what is literally in the text; use null/0/[] when absent.
2. Never invent article numbers or titles.
3. "obligations_count" = number of sentences containing "shall" in the excerpt.
4. Return valid JSON only.
"""


def make_prompt(text_excerpt: str) -> str:
    return STRUCTURE_PROMPT + "\n===== TEXT =====\n" + text_excerpt


def extract_structure(
    text: str,
    model: str = None,
    max_chars: int = 12000,
    use_cache: bool = True,
    num_predict: int = None,
) -> dict:
    """
    run the LLM extraction prompt and return the parsed dict.
    raises json.JSONDecodeError / ValueError if the output is unparseable.
    """
    model = model or c.env_models()["reasoner_default"]
    excerpt = text[:max_chars]
    prompt = make_prompt(excerpt)
    ck = stable_hash(model, prompt)

    if num_predict is None:
        # give R1 headroom for its CoT; qwen needs less
        num_predict = 24576 if "deepseek" in model else 8192

    raw = ollama_chat(
        model,
        prompt,
        temperature=0.0,
        num_ctx=32768,
        num_predict=num_predict,
        max_retries=3,
        use_cache=use_cache,
        cache_key=ck,
    )
    return extract_json(raw)


if __name__ == "__main__":
    import json
    import sys as _s
    md = _s.argv[1]
    model = _s.argv[2] if len(_s.argv) > 2 else c.env_models()["reasoner_default"]
    result = extract_structure(Path(md).read_text(), model=model)
    print(json.dumps(result, indent=2, ensure_ascii=False))
