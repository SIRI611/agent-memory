# Step 6 — Compare Against the Proposed Novelty

Timestamp: 2026-09-23 20:15 (UTC-6)

Counting rule: only an axis judged **match** in Step 5 counts; **partial** is reported but not counted. Level = 5 − (axes matching).

- **Proposed work**
  - Title: —
  - Date: —
  - Source: —
  - Problem framing: Query-time retrieval of stored personal facts that are implicitly necessary (no lexical/embedding cue) for a later request; top-5 facts to context; target recall@5 (and indirect accuracy) on InMind (125 tasks) vs BM25, MiniLM, Jev-Mem-style hybrid-anchor + Jev rerank.
  - Core mechanism: Cheap calibrated System-One evaluator (TypeSafe Jev, probabilities, ~$0.042/1M input tokens) scores EVERY (request, fact) pair with "is this fact required to answer this request safely, correctly, or appropriately?"; no candidate-generation stage; top-5 by probability. Optional write-time Jev gate (persistent user-specific fact, P≥0.5) shrinks the pool before query-time scoring.
  - Key insight: The bottleneck is the similarity-based candidate-generation stage, not the reranker; a cheap calibrated evaluator makes exhaustive knowledge-conditioned relevance scoring affordable, i.e., satisfies InMind's retrieval hypothesis that InMind dismissed on cost.
  - Application domain: Personal-assistant long-term user memory (health/safety/preference-sensitive implicit personalization); evaluated on InMind.

- **Prior work A — Towards Root Memories (RootMem / IMLogic)**
  - Title: Towards Root Memories: Benchmarking and Enhancing Implicit Logical Memory Retrieval for Personalized LLMs
  - Date: 2026-06
  - Source: arXiv:2606.23283 (arXiv preprint)
  - Problem framing: Retrieve a semantically distant but logically essential user memory against semantic distractors; IMLogic (20 users, >15k memories, 2,216 QA); MCQ accuracy. — **match**
  - Core mechanism: LLM distils raw memories into ≤10 root-memory units (rule + evidence); GPT-4o-mini router makes a boolean necessity decision for each unit; activated units concatenated with semantic top-k. — **partial** (exhaustive necessity check without similarity pre-filter, but over ≤10 distilled units with a generative LLM, not every raw fact with a cheap calibrated scorer)
  - Key insight: Over-reliance on semantic similarity; fix by a decision-logic representation; cost controlled by compression ("Instead of scanning all raw memories…"). — **partial**
  - Application domain: Personalized LLM memory. — **match**
  - Axes matching: 2 → **Level 3 — Medium Overlap** (*two axes differ*)

- **Prior work B — InMind**
  - Title: Keep It InMind: Benchmarking the Implicit-Association Blind Spot in Agent Memory
  - Date: 2026-07
  - Source: arXiv:2607.24368 (arXiv preprint)
  - Problem framing: Identical task and necessity definition; 125 tasks, paired direct/indirect controls, answer-blind target recall. — **match**
  - Core mechanism: Benchmark + always-in-state profile probe (68.8%); "We propose no system"; exhaustive full-model scoring named and dismissed on cost. — **differ**
  - Key insight: Failure lies in the query-conditioned similarity interface; a "relevance function conditioned on knowledge" is missing; routing is the open problem. — **partial** (diagnosis shared; affordability-via-cheap-evaluator is new)
  - Application domain: Personal-assistant memory. — **match**
  - Axes matching: 2 → **Level 3 — Medium Overlap** (*two axes differ*)

- **Prior work C — PACE / PACEMaker**
  - Title: PACE: Towards Surfacing Hidden Conflicts in User Requests
  - Date: 2026-09
  - Source: arXiv:2609.03293 (arXiv preprint)
  - Problem framing: Surface latent egocentric KB facts that make a reasonable-looking request inappropriate; Recall@K/MRR + PASS. — **match**
  - Core mechanism: Query reformulation → dense+BM25 seeds → LLM pre-hop filter → multi-hop graph traversal → LLM post-hop filter. — **differ** (similarity-seeded)
  - Key insight: Implicit factors need reformulation/traversal/LLM filtering to be found. — **partial**
  - Application domain: Personalized assistants. — **match**
  - Axes matching: 2 → **Level 3 — Medium Overlap** (*two axes differ*)

- **Prior work D — LoCoMo-Plus**
  - Title: Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents
  - Date: 2026-02
  - Source: arXiv:2602.10715 (ACL 2026 per S2)
  - Problem framing: Latent constraints under cue–trigger semantic disconnect in long dialogue. — **match**
  - Core mechanism: Benchmark/evaluation protocol; top-5 RAG baselines. — **differ**
  - Key insight: Surface-recall benchmarks miss cognitive memory. — **partial**
  - Application domain: Conversational agents. — **match**
  - Axes matching: 2 → **Level 3 — Medium Overlap** (*two axes differ*)

- **Prior work E — Jev-Mem**
  - Title: Jev-Mem: System-One-Controlled Agentic Memory for Efficient AI Agents
  - Date: 2026-09
  - Source: arXiv:2609.23986 (arXiv preprint)
  - Problem framing: Explicit long-term conversational QA on LoCoMo; accuracy + construction time + latency. — **differ**
  - Core mechanism: Jev controller for typing, relations, routing, budget, candidate scoring, stopping; read path seeded by vector+keyword RRF anchors then graph expansion; write path Jev-judges only ≤K_w similarity candidates; no store/discard gate. — **partial** (same scorer, same query-relevance probability, but explicitly candidate-anchored)
  - Key insight: Memory control is "semantic but not generative", so a System-One evaluator is cheaper; exhaustive comparison rejected as cost-growing. — **partial**
  - Application domain: Long-horizon agent memory. — **match**
  - Axes matching: 1 → **Level 4 — Low Overlap** (*three axes differ*)

- **Prior work F — TAG (Beyond Similarity)**
  - Title: Beyond Similarity: Task-Aligned Retrieval for Language Models
  - Date: 2026-05
  - Source: arXiv:2605.27951 (arXiv preprint)
  - Problem framing: Rule/constraint-governed generation (NPOV, PEP 8, NBA). — **differ**
  - Core mechanism: Pairwise LLM applicability judgment for every extracted rule, replacing similarity top-k. — **match** (exhaustive candidate-free pointwise judgment; scorer differs)
  - Key insight: Similarity ≠ applicability ("retrieval objective mismatch"); cost is listed as a limitation. — **partial**
  - Application domain: Policy/style/regulatory documents. — **differ**
  - Axes matching: 1 → **Level 4 — Low Overlap** (*three axes differ*)

- **Prior work G — MemReranker**
  - Title: MemReranker: Reasoning-Aware Reranking for Agent Memory Retrieval
  - Date: 2026-05
  - Source: arXiv:2605.06132 (arXiv preprint / tech report)
  - Problem framing: Rerank dense-recalled memories; MAP. — **differ**
  - Core mechanism: Distilled calibrated 0.6B/4B pointwise reranker in retrieve-then-rerank. — **partial**
  - Key insight: The reranker is the bottleneck (opposite of the proposal). — **differ**
  - Application domain: Agent memory. — **match**
  - Axes matching: 1 → **Level 4 — Low Overlap** (*three axes differ*)

- **Remaining 53 papers (Step 3, not deep-dived):** abstract/title-level overlap 0–1 (domain only, or a single partial on insight/mechanism, e.g., RankGPT, Generative Agents, BRIGHT, Imagine All The Relevance, GOAL-MEM, LoCoMo-Conv at 2 with framing only partial) → Level 4–5.

## Worst case
Minimum level = **Level 3 — Medium Overlap** (RootMem, InMind, PACE, LoCoMo-Plus each match on framing + domain). No paper matches on core mechanism *and* problem framing together. If partial axes were counted as half, RootMem would reach 3 (→ Level 2) and InMind 2.5, so the verdict sits at the Level 3/2 boundary.
