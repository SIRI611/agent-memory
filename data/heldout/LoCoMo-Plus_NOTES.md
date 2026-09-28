# LoCoMo-Plus — held-out benchmark notes

Source: arXiv 2602.10715, GitHub `xjtuleeyf/Locomo-Plus` (branch `main`), downloaded 2026-09-24 from raw.githubusercontent.com.

## Files (under `data/heldout/locomo_plus/`)

| path | size | notes |
|---|---|---|
| `locomo_plus.json` | 306 KB | **the Cognitive cue–trigger instances** |
| `locomo10.json` | 2.8 MB | original LoCoMo, 10 long conversations; used as the haystack the cue/trigger get stitched into |
| `build_conv.py` | 4.6 KB | stitches cue and trigger into a randomly chosen LoCoMo conversation |
| `unified_input.py` | 9.1 KB | merges LoCoMo's 5 categories with Cognitive into one eval JSON |
| `README.md` | 1.6 KB | upstream `data/README.md` |
| `REPO_README.md` | 4.3 KB | upstream top-level README |
| `evaluation_framework/llm_as_judge.py`, `prompt.py` | 7 KB, 4.6 KB | judge code and prompts (copied for reference) |

## Counts

- **401 Cognitive instances** (394 unique cues).
- `relation_type`: causal 101, state 100, goal 100, value 100.
- `cue_dialogue`: 357 are 2-turn (A then B) and 44 are 1-turn. `trigger_query` is always a single A turn.
- `time_gap` is free text, e.g., "two months later" (59), "three months later" (56), "one month later" (44), "six months later" (33), ...
- `model_name` (the generator): gpt-5.1 220, gemini-2.5-flash 90, gpt-5-nano 44, gpt-4o-mini 39, gpt-4o 8.
- Haystack `locomo10.json` contains 10 conversations with 19–32 sessions and 369–689 turns each. It also includes LoCoMo's own 5-category QA, which is not the focus here.

## Schema (`locomo_plus.json`: list of dicts)

Fields: `relation_type`, `cue_dialogue`, `trigger_query`, `time_gap`, `model_name`,
`scores{mpnet,bge,bm25,combined}`, `ranks{mpnet,bge,bm25,combined}`, `final_similarity_score`.
The scores are cue–trigger similarities, and the ranks place each pair within the generation pool. Low similarity is intended, i.e., the pairs are hard for surface retrieval.

Example (item 0):
```json
{
 "relation_type": "causal",
 "cue_dialogue": "A: After learning to say 'no', I've felt a lot less stressed overall.\nB: That's a great skill to develop; protecting your time is important.",
 "trigger_query": "A: I ended up volunteering for that project, and now I'm totally overwhelmed.",
 "time_gap": "two weeks later",
 "model_name": "gpt-4o-mini",
 "scores": {"mpnet": 0.400, "bge": 0.571, "bm25": 1.52, "combined": 0.509},
 "ranks": {"mpnet": 262, "bge": 235, "bm25": 33, "combined": 20},
 "final_similarity_score": 0.509
}
```
Other examples:
- state: cue "My sciatic pain shoots so sharply down my leg that I have to stand during long meetings." and trigger "The airline seat upgrade was totally worth it; I didn't spend the whole flight counting the minutes..."
- value: cue "I stood up to the bully ... confront injustice head-on." and trigger "My neighbor keeps playing loud music ... thinking of just sending an anonymous note instead of knocking on their door."

## Mapping

- **(a) Stored cue to recall:** `cue_dialogue`, a 1–2 turn A/B exchange stating a user state, goal, value, or causal experience. In the unified format, `evidence` = `cue_dialogue` with A/B mapped to the speaker names.
- **(b) Later trigger:** `trigger_query`, a single A utterance inserted `time_gap` after the cue. It does not ask a question; the model is expected to respond in a way that shows awareness of the cue.
- **(c) Distractors / haystack:** there are no per-instance distractor memories. `build_conv.py` inserts the cue and trigger as new sessions into **one randomly chosen** `locomo10.json` conversation (`random.choice`, **unseeded**, so stitching is not reproducible unless a seed is added). The haystack is that entire conversation: about 370–690 turns in 19–32 sessions. Instances do not share a context, and each one is stitched separately.
- **(d) Gold answer / rubric:** **there is no reference answer** for Cognitive. The judge (`prompt.py`, `PROMPT_TEMPLATES["Cognitive"]`, "Memory Awareness Judge") sees only the evidence (the cue) and the model prediction. It labels the prediction "correct" (1) if it "explicitly or implicitly reflects/uses the evidence" and "wrong" (0) otherwise, returning JSON `{"label", "reason"}`. The `relation_type` label is the only explanation-like field.
- **Multiple choice:** no. The format is open-ended only.

## Building the analogous eval (pool of user facts + target; retrieve top-5; judge application)

- Option A, which is closest to the paper: stitch each cue and trigger into a LoCoMo conversation (fix a seed in `build_conv.py`, or reimplement it). Chunk or extract memories from all sessions to form the pool; the target is the chunk or fact containing the cue. Query with `trigger_query`, retrieve the top-5, generate a response, and score it with the Cognitive judge prompt (evidence = cue). Recall@5 of the cue session or turn is directly measurable because the insertion position is known.
- Option B, a pure fact pool without dialogue: convert each `cue_dialogue` into a single user fact (e.g., the A turn). Build the pool as the target cue plus N distractors. Distractors can be the other 400 cues, which are same-style hard negatives, and/or LoCoMo facts (e.g., the `observation` / `session_summary` fields in `locomo10.json`). Then query with the trigger, retrieve the top-5, answer, and judge "applies target fact" using the cue as evidence. Because there is no gold answer, judging is reference-free and only checks for a connection to the cue.
- The provided similarity scores (`scores`, `ranks`) can be used to stratify the instances by how far apart the cue and trigger are on the surface.

## License

- The GitHub repo has **no LICENSE file** (the GitHub API reports `license: null`), and the README says "See the repository for license information", but no license is actually given. Treat it as research use with citation (bibtex is in `REPO_README.md`).
- `locomo10.json` is the original LoCoMo data (snap-research/LoCoMo), which is distributed upstream under CC BY-NC 4.0. That license was not re-verified here.
