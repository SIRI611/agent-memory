# Step 2 — Search and Deduplicate

Timestamp: 2026-09-23 19:50 (UTC-6)

## Queries (sent to paper-search `search_papers.py`, 2023–2026, max 10/source)
- **Q1 Original-Problem:** `long-term memory LLM assistant implicit association personal fact retrieval` → 11 unique (S2=2, Crossref=10)
- **Q2 Broad-Domain:** `LLM agent long-term memory retrieval` → 18 unique (S2=10, Crossref=10)
- **Q3 Method-Signature:** `exhaustive LLM relevance scoring every memory without candidate retrieval` → 12 unique (S2=3, Crossref=10)

### Connector errors (verbatim, surfaced as required)
- `[arxiv] 406 Client Error: Not Acceptable for url: https://export.arxiv.org/api/query?...` (all 3 queries)
- `[dblp] Expecting value: line 1 column 1 (char 0)` (all 3)
- `[open_alex] 401 Client Error: Unauthorized for url: https://api.openalex.org/works?...` (all 3)
- `[openreview] openreview not installed. pip install openreview-py` (all 3)
→ Only Semantic Scholar and Crossref returned results. The CLI output does not print abstracts, so Step 3 for these rows is title/venue-level.

### Merge/dedup across the three queries
41 records → 36 unique by normalized title (5 cross-query duplicates: "A Long-Term Construction Memory System…", "PROMPT PERSISTENCE ATTACKS…", "Long-Term Memory Processes: Memory Retrieval", "Retrieval from long-term memory does not bypass working memory", "Retrieval Beats Cheap Structured Memory…").

## Supplementary live searches (because 4/6 connectors failed)
Via `user-academic.search_external` (OpenAlex/S2) and `user-semantic-scholar.search_papers` (several S2 calls were rate-limited: "Rate limit exceeded for /paper/search"). WebSearch returned "An error occurred while searching the web" for all 4 attempts.
Queries: "InMind implicit memory benchmark retrieval knowledge-conditioned relevance"; "System-One controlled agentic memory calibrated evaluator"; "implicit memory benchmark LLM personalization cue trigger"; "memory retrieval beyond semantic similarity implicit relevance user facts LLM reasoning-based retrieval"; "Jev TypeSafe probabilistic evaluation model" (0 results); "LLM judges relevance of every memory entry exhaustive scan no retriever personalized assistant"; "proactive personalization safety-critical user profile facts allergy medication LLM assistant memory"; "ImplicitMemBench"; "lightweight classifier score all stored memories query-time necessity…"; "PACE: Towards Surfacing Hidden Conflicts in User Requests". S2 citations of InMind: none indexed yet.

Off-topic hits from these catalogs (e.g., smart-grid agentic AI, shape-memory-alloy actuators, healthcare surveys, cybersecurity surveys) were discarded as noise (~35 records). On-topic, non-duplicate additions (19):
InMind (2607.24368); Jev-Mem (2609.23986, fetched directly by ID given by user); Towards Root Memories / RootMem–IMLogic (2606.23283); Beyond Similarity: Task-Aligned Retrieval / TAG (2605.27951); PACE (2609.03293); LoCoMo-Plus (2602.10715); When Users Don't Ask / LoCoMo-Conv (2609.03467); MemReranker (2605.06132); Goal-Oriented Reasoning / GOAL-MEM (2605.12213); Imagine All The Relevance (2503.23033); ImplicitMemBench (2604.08064); When Memory Updates but Behavior Does Not (2608.01619); From Profiling to Synthesis (2608.02171); AlpsBench (2603.26680); PersistBench (OpenAlex, arXiv 2026); ConvMemory v3 (S2, arXiv 2026); MAGMA (2026); A Simple Yet Strong Baseline for Long-Term Conversational Memory (2511.17208); User as Code: Executable Memory for Personalized Agents (2026).

## Model-recall additions (Source: model-recall; checked not already present)
1. Is ChatGPT Good at Search? Investigating LLMs as Re-Ranking Agents (RankGPT) — Sun et al., 2023 (EMNLP 2023), arXiv 2304.09542
2. Generative Agents: Interactive Simulacra of Human Behavior — Park et al., 2023 (UIST 2023), arXiv 2304.03442
3. Memory OS of AI Agent (MemoryOS) — Kang et al., 2025, arXiv 2506.06326 (venue left blank)
4. Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs (PrefEval) — Zhao et al., 2025 (ICLR 2025), arXiv 2502.09597
5. BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval — Su et al., 2024, arXiv 2407.12883

**Combined set: 36 (paper-search) + 19 (supplementary) + 5 (recall) = 60 papers → Step 3.**
