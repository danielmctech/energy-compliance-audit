# EU Energy Regulatory Corpus — Sparse–Dense–Graph RAG Pipeline

A ten-stage pipeline that ingests a 25-document EU energy-regulatory corpus
(Regulation/Directive electricity rules, REMIT II/ACER, NIS2, ENTSO-E, GDPR,
ePrivacy, and national TSO/DSO rules), builds a lineage-traceable chunk corpus
and a regulatory knowledge graph, exposes a hybrid retrieval layer, and evaluates
it with a frozen LLM-as-judge pipeline against a 60-query relational benchmark
and a 160-query held-out generalization set.

---

## Architecture

```
  PDF corpus (25 docs)
         │
  01 corpus audit ────► document inventory, section/article statistics
         │
  02 parser eval ─────► layout fidelity of PDF→markdown vs ground truth
         │               (LLM structure extraction: qwen3.8:27b)
  03a structure maps ─► articles, definitions, obligations
         │
  03b AST chunks ─────► ~3,640 chunks with lineage IDs (doc:article:N / doc:sentence:span)
         │
  04 graph ────────────► 2,076 nodes / 5,178 edges
         │                (CROSS_REFERENCES, AMENDS, DEFINED_IN, APPLIES_TO)
  05 retrieval layer ──► retrieve(query, k, mode)
         │                mode ∈ sparse | dense | graph | hybrid | hybrid_graph | …
  06 GraphRAG API ───── Neo4j-backed GraphRAG ingestion + query API
         │
  07 graph exploration ─► scenario queries (CROSS_REFERENCES / AMENDS paths)
         │
  08 evaluation ────────► 60-query benchmark + 160-query held-out generalization
                          12 retrieval conditions × 10 metrics (Recall, Precision,
                          MRR, nDCG, Hit) + LLM-as-judge answer scoring
```

The retrieval layer (`src/retrieval/`) combines three base methods:

| method | engine |
|---|---|
| `sparse` | Okapi BM25 over a regex-normalised token stream (`rank_bm25`) |
| `dense` | `all-MiniLM-L6-v2` embeddings + FAISS cosine (L2-normalised vectors, `IndexFlatIP`) |
| `graph` | 1–2 hop expansion of seed articles over `CROSS_REFERENCES` / `AMENDS` edges |

and a fusion layer (`hybrid`, `hybrid_graph`, `hybrid_gr_1hop`, `hybrid_gr_expand`,
`hybrid_gr_relations`, `hybrid_graph_rerank`, plus two-hop GCG variants
`gcg_1hop_50`, `gcg_2hop_50`, `gcg_2hop_50_untruncated`,
`gcg_2hop_split119_50`, `gcg_2hop_split119_60`, `gcg_2hop_split129_21`,
`gcg_2hop_split129_21_gc` — see `src/retrieval/two_hop.py`).

Every `RetrievedChunk` carries `lineage_id`, `source_methods`, graph-edge provenance,
and (in fused modes) per-method RRF contribution, so any result can be traced
back to a specific chunk, article, and the retrieval methods that surfaced it.

---

## Regulatory coverage

| Source | Scope |
|---|---|
| EU Electricity Regulation 2019/943 | Market access, capacity allocation |
| EU Electricity Directive 2019/944 | Prosumer rights, non-discrimination |
| REMIT / REMIT II (1227/2011, 2024) | Market manipulation, insider trading |
| ACER operational guidance | REMIT transaction reporting fields |
| GDPR Article 25 | Data protection by design, metering data |
| ePrivacy Directive | Smart meter data handling |
| NIS2 Directive 2022/2555 | Cybersecurity for critical energy infrastructure |
| ENTSO-E network codes | Settlement periods, balancing methodology |
| National TSO/DSO rules | Jurisdiction-specific transpositions (ES, DE, FR…) |

---

## Evaluation framework (notebook 08)

**Benchmark** — 60 relational queries across 8 semantic families
(article → cross-reference, definition → usage, obligation → actor, etc.),
constructed deterministically (no LLM) in three phases: definition →
validation → leakage audit. Each query carries gold-target chunks with
lineage IDs.

**Held-out generalization set** — 160 truly unseen queries (same 8 families)
built to test out-of-distribution performance. No overlap with the benchmark.

**Retrieval metrics** (table A): Recall@K, Precision@K, MRR, nDCG@K, Hit@K
for K ∈ {3, 5, 10, 30} across all 12 retrieval conditions.

**Answer-quality metrics** (table B): a two-model LLM-as-judge pipeline
(primary: `gpt-oss`; escalator: `nemotron-3-nano` when confidence < 0.85)
scores free-text answers on 4 axes + 5 boolean checks.

**Significance tests**: paired Wilcoxon signed-rank (bootstrap 95 % CI)
and Cohen's d effect size across the three headline conditions
(B0, G1, G2-no-GC).

**Key result**: graph-condition retrieval (G2 no graph-context) outperforms
dense-only (B0) on relational queries; the difference is statistically
significant (p < 0.05, d > 0.5). Fine-tuning is documented and reported
separately (§A.4) but excluded from the headline table because 49/60 of its
training pairs leaked into the benchmark.

---

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 24 GB VRAM | 32 GB VRAM (RTX 5090) |
| RAM | 32 GB | 64 GB |
| Storage | 60 GB free | 100 GB SSD |
| CUDA | 12.x | 13.x |
| OS | Ubuntu 22.04 | Ubuntu 24.04 / WSL2 |

---

## Prerequisites

- [Ollama](https://ollama.com) >= 0.23.0 installed and running
- [pyenv](https://github.com/pyenv/pyenv) installed (see setup below)
- Python 3.11.9 via pyenv
- Git with SSH key configured for GitHub
- Models pulled in Ollama (see step 4 below)
- [Neo4j](https://neo4j.com) (optional, for notebook 06 GraphRAG API)

---

## Quick start

### 1. Clone the repository

```bash
git clone git@github.com:YOUR_USERNAME/energy-audit.git
cd energy-audit
```

### 2. Install Python via pyenv

```bash
pyenv install 3.11.9
pyenv local 3.11.9       # creates .python-version automatically
```

### 3. Create and activate the virtual environment

```bash
pyenv virtualenv 3.11.9 energy-audit
pyenv local energy-audit
python --version          # should return Python 3.11.9
```

### 4. Pull the Ollama models used by the pipeline

Notebooks 02/03a call Ollama for LLM structure extraction (cached, so each
document is only extracted once). The evaluation harness (notebook 08) uses
`qwen3.8:27b` as generator and `gpt-oss` as judge:

```bash
ollama pull qwen3.8:27b          # reasoner + generator
ollama pull gpt-oss              # judge (primary)
ollama pull nemotron-3-nano      # judge (escalator)
ollama ps                        # verify models load on GPU
```

No vector DB or embeddings model needs to be pulled manually: `05` builds its
dense index locally — `sentence-transformers` downloads `all-MiniLM-L6-v2` on
first run, `faiss-cpu` is a pure CPU library, and `rank_bm25` is pure Python.

### 5. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Register the Jupyter kernel

```bash
python -m ipykernel install --user \
  --name energy-audit \
  --display-name "Energy Audit (Python 3.11)"
```

### 7. Configure environment variables

```bash
cp .env.example .env
# Edit .env - point OLLAMA_BASE_URL / ENERGY_AUDIT_REASONER at your setup
```

### 8. Run the pipeline notebooks in order

| Notebook | Purpose |
|---|---|
| `notebooks/01_corpus_normalisation_and_Metadata_Audit.ipynb` | Corpus audit — document inventory, section/article stats |
| `notebooks/02_Parser_Evaluation.ipynb` | Parser evaluation — layout fidelity against ground truth |
| `notebooks/03a_Structure_Extraction.ipynb` | Regulatory structure maps (articles, definitions, obligations) |
| `notebooks/03b_AST_Construction.ipynb` | AST chunk corpus with lineage IDs |
| `notebooks/04_Graph_Construction.ipynb` | Knowledge graph: 2,076 nodes, 5,178 edges |
| `notebooks/05_Retrieval_Layer.ipynb` | Hybrid Sparse-Dense-Graph RAG + 12-mode evaluation |
| `notebooks/06_GraphRAG_Query_API.ipynb` | Neo4j-backed GraphRAG ingestion + query API |
| `notebooks/07_Graph_Exploration.ipynb` | Scenario queries over the regulatory graph |
| `notebooks/08_Evaluation_Framework.ipynb` | Benchmark, held-out set, LLM-as-judge, significance tests |

### Per-notebook guides

Each notebook has a short guide in [`docs/notebooks/`](docs/notebooks/README.md) —
what it does, its inputs/outputs, where the logic lives in `src/`, and its
importance to the thesis. Read the [README](docs/notebooks/README.md) for the
importance table and the `lineage_id` contract that ties the pipeline together.

---

## Retrieval layer

Retrieval entry point (`src/retrieval/`):

```python
from retrieval import retrieve
results = retrieve("Who must publish inside information under REMIT?", k=10, mode="hybrid_graph")
```

Fusion modes: `hybrid` (RRF sparse+dense), `hybrid_graph` (RRF + graph neighbours),
`hybrid_gr_1hop` / `hybrid_gr_expand` / `hybrid_gr_relations` (graph-rerank
variants), plus two-hop GCG variants with configurable distance-1 / distance-2
slot allocation (`split_N_M` in `src/retrieval/two_hop.py`).

---

## Generated data artifacts

**All of `notebooks/data/` is excluded from git.** Every file under that
directory (AST chunks, graph nodes/edges, benchmark definitions, retrieval
run outputs, answer/judge JSONLs, aggregate tables, figures, LLM response
caches, backup directories, diagnostic reports) is produced by the notebook
pipeline or an evaluation script. All are regenerable from scratch.

To rebuild the full data tree after cloning:

```bash
# 1. LLM structure extraction (notebooks 02/03a — populates llm_cache,
#    structure_maps, ast, graph)
#    Run notebooks 01-04 in Jupyter.

# 2. 60-query benchmark + 160-query held-out set
PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
    python scripts/build_heldout_benchmark_artifacts.py

# 3. Retrieval over all 12 systems + tables A/B/C/D
PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
    python scripts/rerun_retrieval_and_tables.py

# 4. Held-out retrieval (B0 / G1 / G2 / G2-no-GC)
PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
    python scripts/run_heldout_systems.py

# 5. Held-out analysis (per-query comparison, failure attribution, statistics)
PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
    python scripts/analyze_heldout_run.py

# 6. GCG diagnostic reports (each ablation has its own script)
python scripts/make_gcg_2hop_report.py
python scripts/make_gcg_2hop_split119_report.py
python scripts/make_gcg_rank_attribution_report.py

# 7. GraphRAG (notebook 06) and graph exploration (notebook 07)
#    Run notebooks 06-07 in Jupyter.
```

---

## Project structure

```
energy-audit/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── .python-version                # pyenv (3.11)
├── data/raw|processed/            # 01-03b intermediates (gitignored)
├── outputs/                       # 01 artifacts (gitignored)
├── notebooks/
│   ├── 01_corpus_normalisation_and_Metadata_Audit.ipynb
│   ├── 02_Parser_Evaluation.ipynb
│   ├── 03a_Structure_Extraction.ipynb
│   ├── 03b_AST_Construction.ipynb
│   ├── 04_Graph_Construction.ipynb
│   ├── 05_Retrieval_Layer.ipynb
│   ├── 06_GraphRAG_Query_API.ipynb
│   ├── 07_Graph_Exploration.ipynb
│   ├── 08_Evaluation_Framework.ipynb
│   └── data/                      # ALL gitignored — rebuilt by the pipeline
├── scripts/
│   ├── build_heldout_benchmark_artifacts.py
│   ├── run_heldout_systems.py
│   ├── audit_benchmark.py
│   ├── analyze_heldout_run.py
│   ├── rerun_retrieval_and_tables.py
│   └── ...
└── src/
    ├── common.py                  # config + path resolution
    ├── neo4j_config.py            # Neo4j + GraphRAG client factory (env-driven)
    ├── bootstrap_manifest.py      # rebuild outputs/corpus_metadata.json
    ├── parsing/                   # 02 PDF parsing + ground-truth layer
    │   ├── parsers.py             # PDF→markdown backends
    │   ├── llm_extractor.py       # LLM structure extraction via Ollama (cached)
    │   └── ground_truth.py        # GT extraction (heuristic + LLM review)
    ├── structure/                 # 03a → 04 deterministic corpus → AST → graph
    │   ├── structure_maps.py      # 03a articles / definitions / obligations
    │   ├── ast_builder.py         # 03b chunking + lineage IDs
    │   └── graph_builder.py       # 04 CROSS_REFERENCES / AMENDS / DEFINED_IN edges
    ├── retrieval/                 # hybrid sparse-dense-graph RAG (05)
    │   ├── __init__.py            # Retriever + retrieve(query, k, mode)
    │   ├── _corpus.py             # chunk/graph loading, tokenizer
    │   ├── sparse.py              # BM25
    │   ├── dense.py               # MiniLM + FAISS cosine
    │   ├── graph.py               # 1–2 hop neighbour expansion
    │   ├── fusion.py              # RRF k=60 + graph boost
    │   ├── two_hop.py             # GCG split_N_M admission policy
    │   ├── pair_builder.py        # LoRA training pairs
    │   └── fine_tune.py           # 2-stage LoRA (historical, excluded from thesis)
    ├── reranking/                 # graph-context reranker
    │   └── graph_context.py       # ctx-grafo: graph-neighbourhood description
    ├── graphrag_n4j/              # Neo4j-backed GraphRAG (06)
    │   ├── ingestion.py           # ingest corpus into Neo4j
    │   ├── retriever.py           # graph-first retrieval
    │   ├── context.py             # context assembly
    │   ├── rag.py                 # end-to-end RAG
    │   └── schema.py              # Neo4j schema
    └── evaluation/                # 08 thesis evaluation framework
        ├── __init__.py
        ├── benchmark.py           # 60-query relational benchmark builder
        ├── config.py              # experiment configs, 12 retrieval conditions
        ├── metrics.py             # recall/precision/MRR/nDCG/hit
        ├── generation.py          # frozen LLM generator (qwen3.8:27b, temp=0)
        ├── answer_metrics.py      # LLM-as-judge rubric + escalator
        ├── retrieval_run.py       # per-query retrieval runner
        ├── stats.py               # bootstrap + Wilcoxon signed-rank
        ├── tables.py              # table_A/B/D assembly
        ├── report.py              # reproducibility report
        ├── figures.py             # matplotlib figure generation
        ├── contexts.py            # context-window assembly
        ├── error_analysis.py      # per-query error attribution
        ├── overlap.py             # retrieval-overlap analysis
        ├── baselines.py           # baseline system construction
        ├── scenarios.py           # scenario query construction
        └── semantic.py            # semantic similarity utilities
```

---

## Environment variables

Copy `.env.example` to `.env` and edit as needed:

```
# Paths (optional - defaults to the repo root)
# ENERGY_AUDIT_ROOT=/path/to/energy-audit

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
ENERGY_AUDIT_REASONER=qwen3.8:27b          # generator + structure extraction

# RAG / dense
RETRIEVAL_EXTRA_DENSE=                      # optional: JSON list of extra dense variants

# Neo4j (notebook 06)
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=...
```

---

## Tests

```bash
pytest tests/ -v
```

Tests cover: retrieval core (sparse/dense/graph/hybrid), graph builder,
structure maps, ground truth, Neo4j ingestion, evaluation metrics, stats,
benchmark overlap, answer judge, and CLI scripts.

---

## License

MIT License.
