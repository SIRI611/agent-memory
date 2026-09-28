# Step 4 — High-Potential Candidates

Timestamp: 2026-09-23 20:02 (UTC-6)

Qualifying (overlap ≥2, or mechanism match, or same narrow subfield within 24 months): 37, 38, 39, 40, 41, 42, 43, plus 44 (mechanism partial: calibrated cheap memory scorer). 8 qualify → keep 7 (drop #43 LoCoMo-Conv: benchmark-only, framing is only partial and the implicit queries there still share topical cues with LoCoMo memories; it was downloaded and its intro read, see Step 5 note).

| # | Candidate | Reason selected |
|---|-----------|-----------------|
| 1 | Towards Root Memories / RootMem (2606.23283) | Highest abstract overlap (3): same "semantically distant yet logically critical" framing in personalized memory; LLM router whose scope (all units vs. candidates) was ambiguous from abstract |
| 2 | InMind (2607.24368) | The benchmark the proposal runs on; framing+domain match; explicitly discusses (and dismisses) exhaustive full-model scoring; must verify it did not test it |
| 3 | Jev-Mem (2609.23986) | User-named closest work; same scorer (Jev) used for query–candidate relevance; must verify whether candidate generation is similarity-anchored |
| 4 | TAG — Beyond Similarity (2605.27951) | Mechanism match: pairwise LLM applicability judged per rule, replacing similarity top-k; same "similarity ≠ applicability" insight |
| 5 | PACE (2609.03293) | Framing+domain match: latent user-KB facts making a request inappropriate, recent (2026-09) |
| 6 | LoCoMo-Plus (2602.10715) | Framing+domain: cue–trigger semantic disconnect; cited by InMind as nearest benchmark |
| 7 | MemReranker (2605.06132) | Mechanism-adjacent: small calibrated pointwise memory scorer; recent, same subfield; represents the "better reranker" alternative the proposal argues against |
