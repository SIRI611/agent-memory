# Step 5 — Full-Paper Deep Dive

Timestamp: 2026-09-23 20:10 (UTC-6)

All 7 candidates were fetched with `scripts/fetch_paper.sh` from arXiv PDFs into `papers/` and extracted with `pdftotext -layout` (all returned `ok:`). Reading was a targeted skim (intro, method, experiments, limitations) per the budget rule — not end-to-end. Additionally fetched and skimmed intros of LoCoMo-Conv (2609.03467) and GOAL-MEM (2605.12213). Model-recall papers (RankGPT, Generative Agents, MemoryOS, PrefEval, BRIGHT) were NOT fetched; their records are from memory and abstract-level only.

---

## 1. Towards Root Memories (RootMem / IMLogic) — arXiv:2606.23283
- **Problem framing (verified):** Given query q and memory bank M (~15,000 entries over 20 users, from HaluMem), a "logical memory" m_l "shares negligible semantic similarity with q yet is indispensable"; queries are built to overlap with a distractor "semantic memory" m_s. 2,216 MCQ/open-ended QA; metric = MCQ accuracy + error-type distribution + latency. Baselines include BM25, MiniLM, text-embedding-3-small, Qwen3-Embedding, graph memories, rerank, query-reconstruction+rerank.
- **Core mechanism (verified):** Offline, an LLM extractor periodically distils raw memories into structured *root memory units* U_i = (execution rule R_i, personalized logical evidence E_i); "a maximum unit budget" — "We set the maximum number of root memory units to 10". Online, a GPT-4o-mini *Root Memory Router* Φ_R(q, R_i) ∈ {0,1} decides for each unit whether it is necessary; activated units are concatenated with standard semantic top-k memories (Eq. 4).
- **Key insight (verified):** failures come from "over-reliance on semantic similarity"; the fix is a *representation* (distilled decision logic) that makes logical relevance checkable. Cost is controlled by compressing, not by a cheap scorer: "Instead of scanning all raw memories, the router evaluates the execution rules R_i of each root memory unit".
- **Application domain (verified):** personalized LLM memory; long conversations; GPT-4o-mini, Qwen3-30B, GLM-4.6 backbones; plugged into Mem0/EverMemOS etc.
- **Venue (verified):** arXiv preprint (v1, 22 Jun 2026).
- **Assumptions & scope:** bounded (≤10) abstracted units per user; semantic retrieval still required for the raw facts; router is a generative LLM emitting booleans (not calibrated probabilities / ranking); not evaluated on InMind.
- **Closest-passage evidence:** §3.2 "Instead of scanning all raw memories, the router evaluates the execution rules Ri of each root memory unit to identify the units relevant to the current query… ΦR acts as a boolean decision-maker."
- **Refined overlap:** Framing = match; Domain = match; Mechanism = partial (exhaustive necessity judgment without a similarity pre-filter, but over ≤10 LLM-distilled units, not every raw fact, and with a generative LLM, not a cheap calibrated scorer); Insight = partial (same diagnosis, opposite cost strategy). **Downgraded from 3 → 2 matching axes.**

## 2. InMind — arXiv:2607.24368
- **Problem framing (verified):** Implicit association = (1) necessity "required to answer q safely, correctly, or appropriately", (2) semantic distance, (3) knowledge bridge. 125 expert-verified tasks, 10 life domains; paired naive (direct recall) vs indirect queries; in-context control 84.0%; 6 systems (A-RAG, xMemory, Mem0, A-Mem, HippoRAG 2, MemoryOS) + Naive RAG; answer-blind target recall. Dataset filter: "BM25 and MiniLM score every memory–query pair, and we discard the 700 candidates whose target reads as an obvious lexical or dense match."
- **Core mechanism (verified):** Benchmark; diagnostic "always-in-state" probe: one ≤200-line markdown profile rewritten by GPT-5-mini each session and prepended whole → 68.8% indirect. Explicitly "We propose no system."
- **Key insight (verified):** "Hypothesis 1 is what fails; the retriever works as designed." "Query-conditioned retrieval asks a similarity function for a judgment that requires a world model, then consults the world model afterwards. Until a relevance function conditioned on knowledge exists…". On exhaustive scoring (§2.1): "trivially satisfiable by letting Retrieve run the full model over every stored memory, but that forfeits the sublinear cost that motivates retrieval and merely relocates the world model into the scoring function. We therefore read the hypothesis as deployed systems do, with θ ranging over efficient similarity computations." Also §5.3 frames routing (write-time: what earns persistent state; read-time: when to search).
- **Application domain (verified):** personal-assistant memory; health/wellness/safety-heavy.
- **Venue (verified):** arXiv preprint (2607.24368).
- **Assumptions & scope:** n=125 (±4–5 pt CI); GPT-5-mini is answerer and judge; no negative controls for over-warning (Limitations §7).
- **Closest-passage evidence:** §2.1 quote above — the proposed mechanism is named and set aside, not tested.
- **Refined overlap:** Framing = match (the proposal adopts InMind's task and necessity wording verbatim); Domain = match; Insight = partial (the "bottleneck is the similarity/query-conditioned stage" half is InMind's own conclusion; the "cheap evaluator makes exhaustive scoring affordable" half is new); Mechanism = differ (not implemented; dismissed on cost). **2 matching axes.**

## 3. Jev-Mem — arXiv:2609.23986
- **Problem framing (verified):** Long-term conversational memory QA on LoCoMo; LLM-as-a-Judge (0.777 vs 0.700 MAGMA), memory construction time (158 s) and query latency (0.93 s), gpt-4o-mini backbone. No implicit-association evaluation, no InMind.
- **Core mechanism (verified):** System-One controller J(S,Q) (Jev) produces probabilities for typing, relation judgment, query routing over graph views, budget allocation, candidate scoring (query relevance a_v, relation usefulness, novelty, support), sufficiency/stopping. Read path: "Vector and keyword rankings are fused through reciprocal-rank fusion… The highest-ranked nodes initialize the visited set and search frontier", then graph expansion; candidate score s(v) = weighted mix incl. embedding similarity z_v and Jev a_v. Write path: "Deterministic retrieval first combines vector similarity, lexical overlap, shared entities, and temporal proximity to identify at most K_w candidates… The System-One controller then evaluates only these candidate pairs." No store/discard gate: "The current design preserves observations rather than making an irreversible learned store-or-discard decision at ingestion time."
- **Key insight (verified):** memory-control decisions are "semantic but not generative", so a non-autoregressive probabilistic evaluator is cheaper than an LLM; explicitly avoids exhaustive comparison: "Comparing it against every existing node would make controller cost grow directly with memory size. Jev-Mem therefore separates candidate discovery from relation judgment."
- **Application domain (verified):** long-horizon agent memory (personal agents, coding, etc.); evaluated on LoCoMo only.
- **Venue (verified):** arXiv preprint (v1, 21 Sep 2026).
- **Assumptions & scope:** similarity anchors are assumed to reach relevant regions of the graph; bounded controller calls.
- **Closest-passage evidence:** §3.2/§3.3 quotes above.
- **Refined overlap:** Domain = match; Mechanism = partial (same scorer and same query-relevance probability, but explicitly anchored on vector+keyword candidates — the opposite design choice on exactly the axis the proposal claims); Insight = partial (cheap System-One evaluator for memory decisions is Jev-Mem's thesis; the "candidate generation is the bottleneck / go exhaustive" half is absent and contradicted); Framing = differ (LoCoMo explicit QA). **1 matching axis.**

## 4. TAG — Beyond Similarity: Task-Aligned Retrieval — arXiv:2605.27951
- **Problem framing (verified):** rule-governed generation (Wikipedia NPOV rewriting, HumanEval+PEP 8, RuleArena NBA); retrieve applicable rules, not similar chunks.
- **Core mechanism (verified):** offline 5-phase LLM rule extraction into condition–action tuples; online "Stage 2 evaluates each rule independently through pairwise applicability judgment: match(x, r_i) = f_judge(x, r_i) … returns YES or NO. The matched set … replaces similarity-based top-k retrieval." (~127–195 rules per corpus.)
- **Key insight (verified):** "retrieval objective mismatch": similarity ≠ applicability; applicability must be judged by a model.
- **Application domain (verified):** policy/style/regulatory documents — not user memory.
- **Venue (verified):** arXiv preprint (cs.IR, 27 May 2026).
- **Assumptions & scope:** Limitations: "pairwise applicability matching also introduces additional inference cost compared with dense retrieval"; assumes documents convertible to explicit rules.
- **Closest-passage evidence:** §3.2 Eq. 4 quote.
- **Refined overlap:** Mechanism = match (exhaustive, candidate-free, pointwise model judgment over every item replacing similarity top-k; differs only in scorer — generative LLM YES/NO vs cheap calibrated probability + top-k); Insight = partial (similarity is the wrong relevance function; but no "cheap evaluator makes it affordable" claim — they flag cost as a limitation); Framing = differ; Domain = differ. **1 matching axis (mechanism) — but the most dangerous kind.**

## 5. PACE / PACEMaker — arXiv:2609.03293
- **Problem framing (verified):** persona-grounded user requests + egocentric KB facts; decide whether a request is conflicting; metrics Recall@K, Hit@K, Gold@K, MRR, PASS rate.
- **Core mechanism (verified):** two-step query reformulation → dense + BM25 seed retrieval, "top-K fused documents are then passed through a pre-hop filter agent" → multi-hop kNN document-graph traversal → post-hop filter agent.
- **Key insight (verified):** implicit conflict factors are not directly associated with the request; reformulation + traversal + LLM filtering recovers them.
- **Application domain (verified):** personalized assistants (restaurant booking, scheduling, companions' constraints).
- **Venue (verified):** arXiv preprint (3 Sep 2026).
- **Assumptions & scope:** still similarity-seeded; LLM filtering only over retrieved/traversed candidates.
- **Closest-passage evidence:** Fig. 2 / §4 "Seed Retrieval… Top-K fused documents … pre-hop filter agent".
- **Refined overlap:** Framing = match (implicit, safety/appropriateness-relevant user facts); Domain = match; Mechanism = differ (similarity-seeded, the Jev-Mem pattern); Insight = partial. **2 matching axes.**

## 6. LoCoMo-Plus — arXiv:2602.10715
- **Problem framing (verified):** "cognitive memory under cue–trigger semantic disconnect"; LLM-generated cue dialogues inserted into LoCoMo trajectories; semantic filtering removes high-overlap cue–query pairs; constraint-consistency evaluation.
- **Core mechanism (verified):** benchmark + evaluation framework; RAG baselines "retrieve a fixed set of the top-5".
- **Key insight (verified):** surface-level factual-recall benchmarks miss latent constraints.
- **Application domain (verified):** conversational agents.
- **Venue (verified):** ACL 2026 per Semantic Scholar metadata; PDF is arXiv version.
- **Assumptions & scope:** LLM-authored associations (InMind critiques this as model-prior sampling).
- **Refined overlap:** Framing = match (partial-strong); Domain = match; Mechanism = differ; Insight = partial. **2 matching axes.**

## 7. MemReranker — arXiv:2605.06132
- **Problem framing (verified):** reranking memories inside "retrieve-then-rerank" where BGE-M3 does candidate recall; MAP on a memory retrieval benchmark + finance/healthcare.
- **Core mechanism (verified):** Qwen3-Reranker 0.6B/4B distilled from multi-teacher pairwise LLM judgments; BCE pointwise distillation for calibrated scores; InfoNCE hard negatives.
- **Key insight (verified):** generic rerankers are similarity-bound and miscalibrated; a small reasoning-distilled calibrated scorer fixes the reranker.
- **Application domain (verified):** agent memory (MemOS).
- **Venue (verified):** arXiv preprint / tech report (v2, 14 May 2026).
- **Refined overlap:** Domain = match; Mechanism = partial (cheap calibrated pointwise memory scorer, but only on dense-recalled candidates); Insight = differ (locates the problem in the reranker — the opposite of the proposal); Framing = differ. **1 matching axis.**

## Notes from related-work skims
- InMind's related work cites ImplicitMemBench and LoCoMo-Plus as nearest benchmarks and General Agentic Memory (GAM) as the "search harder" alternative; no exhaustive-scoring system on InMind is cited.
- No paper found (search + citation lookup; S2 lists 0 citations of InMind) that evaluates any method on InMind, or that uses a cheap calibrated evaluator exhaustively over raw user facts.
