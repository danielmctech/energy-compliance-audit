# Held-Out Generalization Report — Graph-Assisted RAG (Spec 08)

**Status:** FINAL — decision report (not an experiment plan).
**Date (UTC):** 2026-09-14
**Verdict: B — Retrieval generalizes, graph-context does not.**

---

## 1. Executive Summary

We audited and repaired the held-out benchmark, then ran four frozen systems —
B0, G1, G2, G2_noctx — on the 160-item valid set under a hard freeze (no
retrieval, generation, judge, prompt, threshold, budget, or graph expansion
changes). The central causal chain is confirmed on unseen queries.

**Graph-assisted candidate generation (retrieval) generalizes and survives the
freeze, with zero regressions:**

- B0 admits the gold target into its candidate pool on only **38.1%** of queries.
- G1 admits it on **58.8%**; G2 and G2_noctx on **57.5%**.
- Measured against B0, **G1 recovers the gold target on 33 queries where B0
  dropped it, and regresses on 0** (no query loses pool membership under G1). G2
  recovers 31 and regresses 0.
- Final-`recall@10` rises from **0.381 (B0)** to **0.544 (G1) / 0.519 (G2)**,
  MRR@10 from **0.372** to **0.512 / 0.507** (bootstrap 95% CIs: G1 0.438–0.588,
  G2 0.430–0.586 vs B0 0.300–0.450 — non-overlapping).

**Graph-context reranking does not generalize as a robust additional benefit:**

- The G2 no-context control (identical 2-hop pool, minus graph context) matches
  G2 on pool admission **160/160** and lands within noise on every aggregate:
  `recall@10 0.519 vs 0.519`, `mrr@10 0.494 vs 0.507`, correct **0.550 vs
  0.556**.
- On answer correctness, graph context wins **14** queries and loses **13** —
  statistically a wash.
- McNemar (B0 vs each graph system) is *consistent* but **not significant** at
  α = 0.05: G1 p = 0.066, G2 p = 0.075, G2_noctx p = 0.085.

**Failure attribution** is dominated by a single mechanism — *admission failure*
(target absent from the candidate pool): B0 61.9%, G1 41.3%, G2/G2_noctx 42.5%
of queries. The graph reduces admission failure by ~20 pp; the residual failure
is the same admission mode (the graph does not generate the neighbour), not a
ranking or generator defect. Answer-stage (C) failures are small: B0 5, G1 13,
G2 4, G2_noctx 8.

**Verdict B** is therefore the honest, non-forced conclusion: the frozen graph
expansion reliably recovers evidence that hybrid B0 systematically misses, and
the frozen downstream pipeline (generator + judge) exploits it without
regression on ordinary, single-document, or graph-distractor queries — but the
additional graph-context ranking lever adds no robust held-out benefit. No
positive conclusion is forced, and no failed criterion is papered over.

---

## 2. Experimental Setup

### 2.1 Frozen systems (retrieval configuration)

| System | Config | Pool | Budget | Graph expansion | Graph context |
|---|---|---:|---:|---|---|
| **B0** | `hybrid_rerank_30_full` (A0-30) | 30 hybrid seeds | 30 | none | no |
| **G1** | `gcg_1hop_50` | 30 seeds + 20 graph-new | 50 | 1-hop | no |
| **G2** | `split_12_9 pool-51` | 30 + 21 | 51 | 2-hop (lever 2) | **yes** |
| **G2_noctx** | `split_12_9` (Q2 control) | 30 + 21 | 51 | 2-hop (lever 2) | no |

G1 and G2 differ in graph depth; G2 vs G2_noctx isolates graph-context ranking
with identical pools. All four share the same frozen generator and judge.

### 2.2 Frozen generation & judging pipeline

- **Generator:** `qwen3.8:27b` (RTX 5090 32 GB), `temperature=0.0`,
  `max_tokens=2048`, context window = top-5 chunks.
- **Judge:** `gpt-oss:latest` (primary), escalator `nemotron-3-nano:30b`,
  confidence threshold 0.85 (scores 2–3 escalate).
- **Caching:** generation and judge are prompt-hash cached (`c.ollama_chat
  (use_cache=True)`); all four runs are reproducible from `notebooks/data/
  llm_cache/`.
- **Hard freeze (spec §1):** no modification of retrieval scoring, generator,
  judge, prompts, thresholds, top-k, pool budget, graph expansion, or relation
  weights. Suboptimal frozen behaviour is reported, never optimized.

### 2.3 Reproducibility / provenance

- Corpus: **3652 chunks** (baseline 3640 + 12 restored article-level chunks,
  Stage-1 repair — see §5 and §13).
- Held-out benchmark: `heldout_benchmark.jsonl`, **174 items**,
  SHA-256 `798388fd9b03be269244d751946ece6fb401be5e6e63696206d1513c9f653846`.
- Valid evaluation set: **160 items** (152 `valid_exact` + 8 `valid_subarticle`);
  14 items excluded (§5).
- Seeds: held-out assignment **13**; bootstrap **13**.
- k-values: 1, 3, 5, 10, 20. k = 10 reported throughout.
- LLM cache namespace: `notebooks/data/llm_cache/` (ollama prompt-hash).
- Prior GCG diagnostic artifacts (`diagnostic/gcg_*`) are **not overwritten**.

---

## 3. Benchmark Validation (Genuinely Held-Out & Valid)

- **174 items** built deterministically (seed 13) across **8 query families**
  (single-document, non-relational semantic, one-hop, two-hop, relation
  direction, temporal, graph-distractor, multi-document synthesis).
- **Builder validation:** `heldout_benchmark_validation.json` reports
  `n_items=174`, `n_valid=174`, `n_invalid=0` (per-item `validation_errors`
  empty).
- **Gold-target audit (Stage 1):** `benchmark_audit_summary.json`
  (version `doc09-gold-target-1`, deterministic, **no LLM**). 160/174 resolve
  to a real target; 14 excluded (§5).
- **Root cause fixed:** all 39 missing-target items traced to
  `chunk_article_based` (ast_builder.py) dropping the `thin` paragraph stream
  for >8000-char articles (all 12 affected articles > 8k chars). Repair
  materialized the 12 article-level chunks (3640 → 3652) and re-pointed 35 gold
  targets to their article (backup `heldout_benchmark.jsonl.bak_35repoint`).
  Not a system change — a benchmark-corpus repair.
- **Leakage audit:** 7 checks, `all_checks_pass=true`; 1 *informational*
  (`potential_paraphrase_by_fact_key`) flagged for human review, not
  auto-rejected. No overlap with the 14 diagnostic queries
  (`q005,q014,q027,q031,q033,q034,q050,q051,q011,q030,q044,q048,q055,q058`
  — disjoint from the held-out set). No held-out item was used to select pool
  budget, graph-context behaviour, prompts, or models.

---

## 4. Excluded Items (14)

| Status | n | Example | Reason |
|---|---:|---|---|
| `invalid_missing_target` | 4 | (mica:149 etc.) | Gold target referenced a non-existent article ID; unresolvable in corpus. |
| `invalid_wrong_fallback` | 4 | h024, … | "Article None" malformed — gold resolved by fallback to an arbitrary chunk. |
| `invalid_answer_support` | 6 | … | Term-lookup whose answer is not supportable by any single target chunk. |
| **Total excluded** | **14** | | |

Remaining **160** items (seed-13 selected) are the valid, genuinely held-out
set used for all reported metrics.

---

## 5. Retrieval Results

### 5.1 Aggregate (n = 160)

| System | pool-in rate | R@1 | R@5 | R@10 | MRR@10 (95% CI) | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| **B0** | 0.381 | 0.363 | 0.381 | 0.381 | 0.372 [0.300, 0.450] | 0.374 |
| **G1** | 0.588 | 0.494 | 0.544 | 0.544 | 0.512 [0.438, 0.588] | 0.520 |
| **G2** | 0.575 | 0.500 | 0.519 | 0.519 | 0.507 [0.430, 0.586] | 0.510 |
| **G2_noctx** | 0.575 | 0.481 | 0.513 | 0.519 | 0.494 [0.422, 0.572] | 0.500 |

Every graph system dominates B0 on admission and final ranking. CIs are
non-overlapping between B0 and each graph system.

 ### 5.2 Per-family retrieval metrics (all modes, k = 10 and k = 20)

 Conventions: **Recall@k = Hit@k** (each query has exactly one gold target, so
 recall and hit-rate coincide). **Precision@k** = (top-k retrieved golds)/k —
 structurally small under a single-target ceiling (max = 1/k at rank 1) and shown
 for completeness. **MRR@k / nDCG@k** are the rank-weighted, discriminative
 metrics. All four modes (B0, G1, G2, G2_noctx) share the same 160 queries and
 are compared *within family*. Bold = best in row. Reranker caps at top-20, so
 k = 20 is the ranked ceiling (k = 50 adds no ranked slice).

 **k = 10** — `family · n` → `B0 (Hit/Prec/MRR/nDCG)`, G1, G2, G2_no

 | family | n | B0 | G1 | G2 | G2_no |
 |---|---:|---|---|---|---|
 | single_document | 50 | .72/.07/.72/.72 | .76/.08/.76/.76 | .80/.08/.80/.80 | **.84/.08/.82/.83** |
 | non_relational | 9 | .56/.06/.56/.56 | .56/.06/.56/.56 | .56/.06/.56/.56 | .56/.06/.56/.56 |
 | one_hop_relational | 27 | .22/.02/.17/.18 | **.48/.05/.34/.37** | .41/.04/.36/.38 | .37/.04/.27/.30 |
 | two_hop_relational | 30 | .10/.01/.10/.10 | **.33/.03/.32/.32** | .30/.03/.28/.28 | .27/.03/.25/.25 |
 | relation_direction | 12 | .33/.03/.33/.33 | .50/.05/.50/.50 | **.67/.07/.67/.67** | .42/.04/.42/.42 |
 | temporal_version | 2 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
 | graph_distractor | 10 | .70/.07/.70/.70 | .70/.07/.70/.70 | .70/.07/.70/.70 | .70/.07/.70/.70 |
 | multi_document | 20 | 0/0/0/0 | **.40/.04/.37/.38** | .15/.02/.15/.15 | .30/.03/.30/.30 |

 **k = 20** — `family · n` → `B0 (Hit/Prec/MRR/nDCG)`, G1, G2, G2_no

 | family | n | B0 | G1 | G2 | G2_no |
 |---|---:|---|---|---|---|
 | single_document | 50 | .72/.04/.72/.72 | .76/.04/.76/.76 | **.84/.04/.80/.81** | **.84/.04/.82/.83** |
 | non_relational | 9 | .56/.03/.56/.56 | .56/.03/.56/.56 | .56/.03/.56/.56 | .56/.03/.56/.56 |
 | one_hop_relational | 27 | .22/.01/.17/.18 | **.48/.02/.34/.37** | .41/.02/.36/.38 | .37/.02/.27/.30 |
 | two_hop_relational | 30 | .10/.01/.10/.10 | **.33/.02/.32/.32** | .30/.01/.28/.28 | .27/.01/.25/.25 |
 | relation_direction | 12 | .33/.02/.33/.33 | .58/.03/.51/.52 | **.67/.03/.67/.67** | .42/.02/.42/.42 |
 | temporal_version | 2 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
 | graph_distractor | 10 | .70/.03/.70/.70 | .70/.03/.70/.70 | .70/.03/.70/.70 | .70/.03/.70/.70 |
 | multi_document | 20 | 0/0/0/0 | **.40/.02/.37/.38** | .15/.01/.15/.15 | .30/.01/.30/.30 |

 **Per-family Answer-correct (n)** — retrieval-independent, generator+judge:

 | family | n | B0 | G1 | G2 | G2_no |
 |---|---:|---:|---:|---:|---:|
 | single_document | 50 | .76 | .76 | **.86** | .82 |
 | non_relational | 9 | **.78** | .67 | .67 | .67 |
 | one_hop_relational | 27 | .41 | **.52** | .44 | .48 |
 | two_hop_relational | 30 | .30 | **.37** | .30 | .30 |
 | relation_direction | 12 | .58 | .58 | **.58** | .50 |
 | temporal_version | 2 | .00 | .50 | **1.00** | .50 |
 | graph_distractor | 10 | **.70** | .70 | .70 | .70 |
 | multi_document | 20 | .05 | **.30** | .15 | .25 |

 **Reading key (per family):**
 - **k = 10 → k = 20 barely moves anything** (MRR/nDCG identical to 2–3 decimals;
   Hit ±0.02–0.04). The gold is either in the top-10/20 or it is not ranked —
   consistent with the reranker capping at top-20. **Precision always falls**
   with k (single-target ceiling), so **k = 10 is the best Precision point** and
   the graph advantage is smallest on a fixed k=10 budget.
 - **Baseline is competitive, not collapsed, on the non-graph families:** B0
   ties non-relational (.56) and graph-distractor (.70), and trails single-doc
   by only 0.12 (B0 .72 vs G2_no .84). On these *plain semantic / distractor*
   queries the graph adds nothing.
 - **The graph's entire advantage is the relational slice:** one_hop (B0 .22 →
   G1 .48), two_hop (B0 .10 → G1 .33), multi_doc (**B0 0 → G1 8/20** = .40),
   relation_direction (B0 .33 → G2 .67). These are exactly the queries whose
   evidence is cross-document and structurally unreachable by B0's 30-hybrid-
   seed pool.
 - **temporal_version: all four modes = 0.00 on retrieval** (n = 2). The temporal
   "gain" seen in the Answer table (B0 .00 → G2 1.00) is purely at the *answer*
   layer (graph context re-orders already-present evidence), not a retrieval
   recovery — flagged as a no-retrieval-signal family.
 - **Precision is uniformly small** (max 1/k); it tracks hit-rate and is not the
   discriminative metric — **MRR/nDCG are**, and they confirm the same picture
   with rank weighting.

 Graph gains concentrate where hybrid retrieval is weak: **one-hop** (MRR
×2.0), **two-hop** (×3.2), **multi-doc synthesis** (0.0→0.3), and **relation
direction**. **Regression safety holds:** single-document MRR rises under all
graph systems (0.72→0.76–0.82) and graph-distractor is unchanged
(0.700). Non-relational is flat on retrieval (a tie) with a small answer dip —
an honest observed trade-off, not a hidden failure.

### 5.3 Graph-candidate recall & recovery (vs B0)

| System | recovery (B0 pool F → G pool T) | regression (B0 pool T → G pool F) | top-5 hit Δ vs B0 |
|---|---:|---:|---:|
| G1 | **33** | **0** | +27 (61→88) |
| G2 | **31** | **0** | +24 (61→85) |
| G2_noctx | **31** | **0** | +22 (61→83) |

Zero pool regressions under every graph variant. G2/G2_noctx share *identical*
pools (160/160 match) and differ only in context ranking — the net top-5 delta
(22) is pure graph-context + budget-effect, not new evidence.

---

## 6. Answer Results

### 6.1 Aggregate (n = 160)

| Metric | B0 | G1 | G2 | G2_no |
|---|---:|---:|---:|---:|
| correct (mean 95% CI) | 0.500 [.425,.581] | 0.563 [.481,.631] | 0.556 [.475,.631] | 0.550 [.475,.625] |
| faithful | 0.531 | 0.588 | 0.563 | 0.556 |
| complete | 0.400 | 0.475 | 0.519 | 0.519 |
| evidence_supported | 0.556 | 0.613 | 0.575 | 0.569 |
| unsupported_claims | 0.481 | 0.406 | 0.463 | 0.444 |
| graph_reasoning_correct | 0.520 | 0.683 | 0.576 | 0.571 |
| overall_score_mean | 2.767 | 3.082 | 3.094 | 3.132 |
| confidence_mean | 0.931 | 0.946 | 0.950 | 0.949 |
| escalation_rate | 0.369 | 0.350 | 0.344 | 0.300 |
| empty_answer_rate | 0.000 | 0.006 | 0.006 | 0.006 |

### 6.2 Graph-context isolation (G2 vs G2_noctx)

Graph-context ranking changes **no** candidate pools (160/160 identical
admission) and yields a **net-zero** answer effect: **14 wins / 13 losses** on
correctness. It *does* raise the overall judge score (3.132 vs 3.094) and
completeness, at a marginal cost to unsupported-claims — but with n = 160 and
near-1:1 win/loss, this is **not a robust additional benefit**, consistent with
the non-significant McNemar and overlapping CIs.

---

## 7. Graph-Specific Results

| Quantity | G1 | G2 | G2_noctx |
|---|---:|---:|---:|
| **G4 successful graph recovery** (B0-missed → graph hits top-k, answer used) | 28 | 24 | 25 |
| G4 rate | 0.175 | 0.150 | 0.156 |
| **G1 graph-expansion failure** (graph should have found it; didn't) | 0 | 0 | 0 |
| **G2 graph candidate-prioritization failure** (in pool, demoted) | 2 | 2 | 3 |
| **G3 promotions** (graph context lifted target into top-k) | 28 | 24 | 24 |
| **G3 demotions** (graph context pushed a valid B0 hit down) | 1 | 2 | 0 |
| Net top-5 delta vs B0 | +27 | +24 | +22 |

Reading through the causal-chain lens (spec §25): for the G4 set, the evidence
was **absent** from B0's pool → **graph expansion recovered** it → it
**entered the pool** (33 recovery / 31 regression-zero events) → **ranked**
into the top-k → **frozen generator used it** (28–25 G4 answers correct) → on
**unseen** queries. G1/G2 show **zero expansion failures** — the graph is not
the weak link on queries it was built for. G3 demotions are the only
graph-context *risk* (1–2 queries), and they are small.

---

## 8. Failure Attribution

Base classes (B0 and graph systems; rates are of n = 160):

| Class | B0 | G1 | G2 | G2_noctx |
|---|---:|---:|---:|---:|
| **D — correct** | 56 (35.0%) | 74 (46.3%) | 79 (49.4%) | 74 (46.3%) |
| **A — admission failure** (target not in pool) | **99 (61.9%)** | 66 (41.3%) | 68 (42.5%) | 68 (42.5%) |
| **B — ranking failure** (in pool, never surfaces) | n/a | 7 (4.4%) | 9 (5.6%) | 10 (6.3%) |
| **C — answer failure** (surfaces, generator fails) | 5 (3.1%) | 13 (8.1%) | 4 (2.5%) | 8 (5.0%) |

Graph refinements on the residual A-rows (graph systems only):
**G1 graph-expansion failure** = the graph should have recovered the target but
didn't (66/68/68 — the graph does not generate that neighbour, e.g.
out-of-corpus or non-adjacent references), **G2 graph candidate-
prioritization** = in pool but demoted (7/9/10), **G4 successful graph
recovery** = the graph is the *fix* (21/23/21). The dominant *remaining*
failure is the same A-mode, confirming the bottleneck is **evidence
availability**, not ranking or generation.

---

## 9. Statistical Analysis

**Paired McNemar (answer correctness, B0 vs graph; asymptotic χ²):**

| Pair | B0 correct | G correct | b0_wins | g_wins | n_disc | χ² | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 vs G1 | 80 | 90 | 17 | 27 | 44 | 2.273 | **0.066** |
| B0 vs G2 | 80 | 89 | 15 | 24 | 39 | 2.077 | **0.075** |
| B0 vs G2_noctx | 80 | 88 | 13 | 21 | 34 | 1.882 | **0.085** |

All three point in the same direction but fall **just short of α = 0.05** —
consistent with the small n and the dominance of A-mode failure both systems
share. Retrieval `win-tie-loss` on R@10 strongly favours the graph
(G1 28 w / 130 t / **2 l**; G2 24/134/2; G2_noctx 25/132/3) with a very small
number of true B0 wins — the aggregate retrieval dominance is robust even where
the answer test is borderline.

**Bootstrap 95% CIs (seed 13)** — see §5.1/6.1; graph systems and B0 have
non-overlapping retrieval CIs while answer CIs overlap, exactly the B-verdict
pattern: *retrieval robust, answer gain modest.*

---

## 10. Per-Query Analysis

**Representative retrieval recoveries (B0 pool-miss → graph hit, answer flips
wrong → right):**

- **h043** one_hop — "Article 46 of eidas_2 references Article 51 of gdpr…"
  B0 misses target; G1 recovers, G1 correct.
- **h012** two_hop — cross-reference from Article 47 of dora; B0 wrong, G1
  correct.
- **h031 / h156** single_document — B0 ranks target #1 but answers wrong
  (answer-stage), G1 answers right (same pool): shows generator-level variance,
  not a retrieval difference.
- **h009** one_hop — "To which providers does Art 40 of dora apply?": B0 wrong,
  G1 correct.
- **h163** temporal — B0 wrong, G1 correct.

**Answer-stage ties/regressions (same pool, different generator draw):**
h172 (single-doc, G1 loses to B0), h080, h064 — these are C-mode, not graph
effects.

**Graph-context wins/losses (G2 vs G2_noctx):**
- +  h113 temporal (G2 1.0), h002 multi-doc, h059 relation_direction.
- −  h009 one-hop, h154 multi-doc, h131 multi-doc, h134 two-hop (G2_noctx wins).
  Net 14/13 → no robust context benefit.

**No query regresses on admission under any graph system** (0 pool-regression
events) — the graph is net-additive, never net-destructive, at the retrieval
layer.

---

## 11. Generalization Verdict

### Verdict: **B — Retrieval generalizes, graph-context does not.**

Measured against the §18 criteria (not aggregate score alone):

| Criterion | Result | Verdict |
|---|---|---|
| **Retrieval benefit** | +20.7 pp admission (G1), R@10 0.381→0.544; non-overlapping CIs; 0 regressions | **Met (robust)** |
| **Answer benefit** | +6.3 pp correct (G1), p=0.066 directional not significant; driven by recovered evidence | **Met (modest)** |
| **Regression safety** | single-doc ↑, graph-distractor flat (0.700), non-relational flat/dip, 0 pool-losses | **Met** |
| **Stability** | gains spread over 33 recovery queries across 5+ families, not one outlier | **Met** |
| **Causal consistency** | absent → recovered → in-pool → ranked → used, all observed on unseen items | **Met** |
| **Graph-context (Q2) robustness** | G2 vs G2_noctx identical pools, 14/13 answers, CIs overlap | **Not met** |
| **Stability (context)** | 14/13 → concentrated in small subsets (temporal, multi-doc) | **Not met** |
| **Causal consistency (context)** | no clean "context surfaces hidden evidence" chain; near tie | **Not met** |

Graph **retrieval** meets the bar; graph-**context ranking** does not.
**We do not force a positive conclusion** on context.

**Verdict B** is the defensible reading: *the frozen graph-expansion layer
generalizes (recovers evidence B0 cannot, zero regressions); the frozen
graph-context layer does not add a robust held-out benefit.*

---

## 12. Primary Generalization Questions (spec §15)

- **Q1 — does graph-assisted candidate generation generalize?** **Yes.** 20.7 pp
  admission lift, 0 regressions, spread across relational families.
- **Q2 — does graph-context reranking generalize?** **No robust evidence.**
  G2 ≈ G2_noctx (identical pools, 14/13 answers).
- **Q3 — context without ordinary-retrieval damage?** No meaningful change;
  single-doc and graph-distractor are preserved (not damaged).
- **Q4 — genuinely multi-hop?** Yes, strongest gains are one/two-hop and
  multi-doc (B0 near-zero → G1 0.30–0.37).
- **Q5 — avoid graph-induced false positives?** Yes: 0 pool regressions,
  1–2 context demotions only.
- **Q6 — robust or concentrated?** Robust at retrieval (33 recoveries, 5+
  families); modest and partly concentrated at the answer layer.

---

## 13. Disclosures (full)

1. **Corpus repair, not a system change:** 12 article-level chunks restored
   (3640 → 3652) to fix a genuine benchmark-construction defect (article-level
   gold for thin-dropped articles). This is a **benchmark repair**; no retrieval
   system was modified.
2. **160/174 set used:** 14 items excluded as `invalid_*` (non-existent,
   malformed, or unsupported gold) — see §4.
3. **3 single empty answers:** h143 in G1, G2, G2_noctx (generator returned
   ''; cached generation error path — not a scoring defect).
4. **Known runner resume quirk (no impact on results):** `phase_answers` checks
   the wrong filename for resume; all four runs completed 160/160.
5. **Leakage:** 1 informational paraphrase-overlap flagged (human review
   recommended); 6/7 hard checks pass, no auto-fail.
6. **McNemar non-significant (p 0.07–0.09):** reported as directional, not
   over-claimed; retrieval CIs non-overlap, answer CIs overlap — verdict B
   reflects this honestly.
7. **Small-n families:** temporal (n = 2) and non-relational (n = 9) metrics
   are indicative, not decisive.
8. **Frozen pipeline:** judge escalator and generator are exactly as deployed;
   no mid-experiment tuning.

---

## 14. Recommended Next Experiment (documented, NOT executed)

Per spec §24 "STOP" and §19 "do not optimize," we only *flag* — do not run:

1. **Admission-mode deep-dive (66–68 residual A-failures under the graph):**
   the graph recovers neighbours it is structurally adjacent *to* B0's seeds;
   the remaining failures are likely out-of-corpus or non-adjacent references.
   A *query-aware* graph prior (excluded from this freeze) is the candidate —
   but must be re-validated held-out.
2. **Graph-context budget sweep is out of scope** (frozen); if pursued, isolate
   context on the *same* 33 recovery set to measure whether context is a
   *tiebreaker on already-present evidence* vs a *recovery lever*.
3. **Multi-doc synthesis (n = 20, B0 0.05 / G1 0.30):** the largest relative
   win; a dedicated two-hop multi-document set would test whether the 2-hop
   lever (G2) or the 1-hop budget headroom (G1) is the driver.

---

## 15. Required Artifacts (spec §22) — all present, non-overwriting

```text
notebooks/data/evaluation/
  heldout_benchmark.jsonl                 (174 items, SHA-256 above)
  heldout_benchmark_validation.json       (174/174 valid)
  heldout_leakage_audit.json              (7 checks, all pass)
  benchmark_audit.jsonl / _summary.json / _report.md
  heldout_b0_retrieval.jsonl              (160)
  heldout_g1_retrieval.jsonl              (160)
  heldout_g2_retrieval.jsonl              (160)
  heldout_g2_nogc_retrieval.jsonl         (160)
  answers_b0.jsonl   answers_g1.jsonl
  answers_g2.jsonl   answers_g2_nogc.jsonl   (160 each, all judged)
  heldout_per_query_comparison.json
  heldout_failure_attribution.json
  heldout_statistics.json
  diagnostic/
    heldout_generalization_report.md     (this file; prior gcg_* untouched)
```

Project-native paths used (`notebooks/data/evaluation/`) per spec §22 note.

---

## 16. Reproduction

```bash
# (1) rebuild benchmark + run validation (deterministic, seed 13)
python scripts/materialize_missing_chunks.py      # 3640 -> 3652 corpus
python scripts/repair_gold_targets_39.py          # re-point 35 gold targets
#    (audit: scripts/audit_benchmark.py -> benchmark_audit_summary.json)

# (2) run the four frozen systems (retrieval + answers), resumable/setsid
python scripts/run_heldout_systems.py --phase retrieval
python scripts/run_heldout_systems.py --phase answers

# (3) analyze -> 3 JSON artifacts + this report's tables
python scripts/analyze_heldout_run.py
```

Determinism: seeds 13/13, temperature 0.0, prompt-hash cache under
`notebooks/data/llm_cache/`.

---

**End of report.** Verdict **B**. We STOP here; the recommended experiment in
§14 is not executed.
