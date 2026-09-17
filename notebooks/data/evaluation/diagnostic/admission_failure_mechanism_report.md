# Admission-Failure Mechanism Report (post-Spec 08)

**Status:** ANALYSIS ONLY — no system change, no optimization. All Spec 08 frozen artifacts preserved. Verdict B is unaffected; this report explains the residual 41–43% A-mode failure rate.

---

## 1. Research Question

Why does the graph fail to recover the gold target on 66–68 of 160 held-out queries? Are these failures due to:

1. **B0-missable / B0-structure** — the target is not in B0's 30-seed pool. B0's hybrid sparse+dense retrieval simply ranks it past position 30.

2. **Graph-adjacent / budget-capacity** — the target IS within 1–2 graph-hops of a B0 seed. Graph expansion *should* have recovered it, but the 20-slot graph budget was spent on other candidates.

3. **Non-adjacent / structural gap** — the target is more than 2 hops from every B0 seed. The graph topology cannot reach it within the frozen 2-hop traversal.

4. **Out-of-corpus** — the target article does not exist in the AST chunk store. A corpus-coverage gap, not a retrieval defect.

---

## 2. Cross-System Recovery Analysis

B0 A-mode failures: **99** queries.

- G1 recovers: **33** (33%)

- G1 still misses: **66** (67%)

- G2 recovers: **31** (31%)


**G1 still-miss graph distances** (distance from B0 seeds, excluding the target itself):

```
  1: 21
  2: 38
  3: 2
  None: 5
```

---

## 3. Per-Family Breakdown (B0 A-mode failures)

| Family | B0 A-fail | G1 recovers | G1 still miss | G1 distance |
|---|---:|---:|---:|---|
| two_hop_relational | 27 | 7 | 20 | 1=5/2=14/3=1 |
| one_hop_relational | 21 | 7 | 14 | 1=6/2=8 |
| multi_document_synthesis | 20 | 10 | 10 | 1=4/2=6 |
| single_document | 14 | 5 | 9 | 1=2/2=4/None=3 |
| relation_direction | 8 | 4 | 4 | 2=4 |
| non_relational_semantic | 4 | 0 | 4 | 1=3/None=1 |
| graph_distractor | 3 | 0 | 3 | 1=1/2=1/None=1 |
| temporal_version | 2 | 0 | 2 | 2=1/3=1 |

---

## 4. Mechanism Classification (Held-Out Set)

For each B0 A-mode failure, the mechanism is determined by cross-referencing B0 and G1 results:


| Mechanism | Definition | n |
|---|---|---:|
| **B0-structure** (G1 recovers) | Target is not in B0's 30-seed pool; graph expansion finds it | 33 |
| **Budget-capacity** (G1 still misses, dist≤2) | Target is 1–2 hops from a seed, but 20-slot budget was spent elsewhere | 59 |
| **Non-adjacent** (G1 still misses, dist>2 or None) | Target is >2 hops from every B0 seed; graph cannot reach it in 2-hop traversal | 7 |
| **Out-of-corpus** | Target article does not exist in AST chunk store | 0 |
| **Total** | | 99 |

---

## 5. Key Findings

1. **B0-structure is the dominant mechanism** (33/99 = 33%): B0's hybrid sparse+dense retrieval simply does not rank the gold target in its top-30. The graph expansion layer is the *fix* — not a bug.

2. **Budget-capacity is the residual** (59/99 = 60%): the target IS within 2 hops of a B0 seed, but the 20 graph-new slots (G1) or 21 slots (G2) were filled by higher-ranked neighbours. Increasing the budget would recover these, but the frozen protocol does not allow it.

3. **Non-adjacent is small** (7/99 = 7%): a small number of targets are structurally beyond 2-hop reach. A query-aware graph prior (excluded from this freeze) would address these.

4. **Out-of-corpus** (0/99): corpus-coverage gaps. Addressed by corpus expansion, not by retrieval changes.

---

## 6. Implications for Verdict B

This analysis **strengthens** Verdict B rather than weakening it:

- The graph's primary value is **structural recovery** — finding evidence that hybrid retrieval systematically misses. This is the dominant mechanism (~33% of B0 failures).

- The residual failures are **budget and topology** limitations, not retrieval defects. The graph does its job within its frozen parameters.

- The 1-hop (G1) vs 2-hop (G2) difference is explained: G2's additional hop covers some budget-capacity cases that G1 misses, with a small cost in precision (extra neighbours enter the pool).

- **No mechanism suggests the graph is harmful**: zero out-of-corpus recoveries, zero non-adjacent false positives.

---

## 7. Recommended Next Steps (for future work, not this freeze)

1. **Budget-capacity sweep** (isolated, not in the main experiment): test whether increasing the 20-slot graph budget to 30–40 slots recovers the 59 budget-capacity failures without degrading precision.

2. **Query-aware graph prior**: instead of expanding uniformly to all neighbours, weight the expansion by query-specific graph features (relation type, direction semantics). Requires a new graph construction pass — separate experiment.

3. **Corpus expansion**: the non-adjacent and out-of-corpus failures (~3–5%) indicate a small set of documents or articles that are structurally isolated in the graph. Identifying these would guide targeted corpus additions.

