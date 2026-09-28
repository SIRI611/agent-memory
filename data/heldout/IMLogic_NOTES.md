# IMLogic — held-out benchmark notes

Source: RootMem paper (arXiv 2606.23283). Repo: https://anonymous.4open.science/r/IMLogic-DBB3
(downloaded 2026-09-24 via `https://anonymous.4open.science/api/repo/IMLogic-DBB3/file/<path>`;
the directory listing API is `.../files?path=<dir>`).

## Files (under `data/heldout/imlogic/`)

| path | size | notes |
|---|---|---|
| `README.md` | 12 KB | upstream README |
| `qa/qa_user00.json` … `qa_user19.json` | 20 files, 3.2 MB total (92 KB – 363 KB each) | benchmark instances, one file per user |
| `dialogue/HaluMem-Medium.jsonl` | 33.5 MB | haystack / memory bank. **Not in 4open repo** (the 4open `HaluMem-Medium.jsonl.zip` is a 2-byte placeholder). Downloaded from HuggingFace `IAAR-Shanghai/HaluMem` (`HaluMem-Medium.jsonl`) |
| `dialogue/HaluMem_README.md` | 12 KB | HaluMem dataset card |
| `eval/*.py` | 4 files | reference eval (`evaluate_common.py`, `evaluate_on_memory.py`, `evaluate_on_conversation.py`, `judge_openended.py`) |
| `generate/*.py` | 3 files | generation / verification prompts |

Alignment: line *i* of `HaluMem-Medium.jsonl` (20 lines = 20 users) ↔ `qa_user{i:02d}.json`.
Verified: for all 2216 instances, both `memory_l` and `memory_s` appear verbatim as a
`memory_content` in the corresponding user's HaluMem memory bank. In 7/2216 cases, `memory_l` appears more than once in the bank (exact duplicate strings).

## Counts

- **2216 QA instances** across 20 users; per-user counts:
  226, 98, 73, 64, 79, 82, 85, 134, 75, 145, 128, 121, 148, 137, 64, 77, 100, 166, 117, 97.
- 1634 unique `memory_l`; 2216 unique (`memory_l`, `memory_s`) pairs.
- `category`: State Constraint 609, Motivational Orientation 588, Identity Shift 291,
  Goal/Mission Alignment 287, Contextual Fit 284, Resource Constraint 134, Other 23.
- `query_type`: Advice 1699, Recommendation 455, Conversation 62.
- Memory bank per user (all `memory_points` across sessions): 610–876 memories
  (mean ≈ 750; 14,948 total; types Persona 9116 / Event 4550 / Relationship 1282;
  3122 flagged `is_update="True"`). Sessions per user: 58–82; dialogue turns per user: 2342–3538.

## QA schema (`qa/qa_userXX.json`: list of dicts)

Fields: `memory_s`, `memory_l`, `query_type`, `query`, `options{Correct, Trap_Preference, Trap_Fabrication, Trap_Generic}`,
`diff_similarity_score` (float; missing in 152 items), `query_time` (missing in 1 item), `category`.
One item has only 3 options (no `Trap_Fabrication`). (The upstream README calls these fields `m_s`/`m_l`; the actual JSON keys are `memory_s`/`memory_l`.)

Example (`qa_user03.json[0]`):
```json
{
 "memory_s": "Sarah Garcia discussed her career aspirations and desire for continuous growth and learning within her current field.",
 "memory_l": "Sarah Garcia acknowledged the importance of her savings as a financial buffer during her transition period after the layoff, allowing her to focus on refining her platform's mission.",
 "query_type": "Advice",
 "query": "Should I enroll in that expensive executive leadership bootcamp to boost my career growth?",
 "options": {
  "Correct": "Given that you are currently in a transition period after a layoff and relying on savings as a financial buffer, it would be wiser to avoid large expenditures right now ...",
  "Trap_Preference": "Yes, you should enroll because it aligns perfectly with your strong desire for continuous growth ...",
  "Trap_Fabrication": "Absolutely, especially since your company just announced a full reimbursement program ...",
  "Trap_Generic": "Executive leadership bootcamps are generally excellent for networking ..."
 },
 "diff_similarity_score": -0.1112,
 "query_time": "Dec 20, 2037",
 "category": "Goal/Mission Alignment"
}
```

## HaluMem-Medium schema (one JSON object per line)

Top-level: `uuid`, `persona_info` (string, ~550–770 chars), `sessions`, `total_dialogue_token_length`, `total_question_count`, `token_cost`.
Session: `start_time`, `end_time`, `memory_points_count`, `memory_points`, `dialogue_turn_num`, `dialogue`, `dialogue_token_length`, `questions`, `question_count`.
memory_point example:
```json
{"index": 1, "memory_content": "User's name is Martin Mark", "memory_type": "Persona Memory", "is_update": "False",
 "original_memories": [], "timestamp": "Sep 04, 2025, 21:12:18", "event_source": 0, "importance": 0.75, "memory_source": "system"}
```
dialogue turn: `{"role": "user", "content": "...", "timestamp": "...", "dialogue_turn": 0}`.

## Mapping

- **(a) Stored memory to recall (target):** `memory_l`, the "logical" memory that should constrain the answer (e.g., savings buffer after a layoff). It is a verbatim entry in the user's HaluMem `memory_points`.
- **(b) Later query/trigger:** `query` (plus `query_time`). By construction it is surface-aligned with `memory_s` and has "zero leakage" of `memory_l` (per the generator/judger prompts).
- **(c) Distractors / haystack:** the adversarial distractor is `memory_s`, the surface-relevant "semantic" memory, which is present in the same bank. The full haystack is all of that user's `memory_points` (≈610–876 per user; the whole bank, not a per-instance sample). In the conversation-level setting, the raw `dialogue` turns are used instead (~2.3k–3.5k turns per user).
- **(d) Gold answer / rubric:** `options.Correct` is the gold answer. The three traps are labeled by failure mode: Preference = follows `memory_s`, Fabrication = hallucinated justification, Generic = no memory grounding. Open-ended judging (`eval/judge_openended.py`) gives the judge `memory_l` as the logical memory, `options.Correct` as the reference, the query, and the prediction, and returns `{"is_correct": bool, "reasoning": ...}`. The judge is strict: the answer must explicitly reflect the constraint, and generic answers count as wrong. There is no separate free-text explanation field.
- **Multiple choice:** yes. There are 4 options (A–D), shuffled with `random.Random(f"{seed}:{user_idx}:{q_idx}")` (`--shuffle-seed`), and the model outputs a single letter. The default `--top-k` is 5.

## Building the analogous eval (pool of user facts + target; retrieve top-5; judge application)

1. For user *i*, the pool is every `memory_content` in `HaluMem-Medium.jsonl` line *i*, which gives about 750 facts (optionally with timestamp/type, as in `evaluate_on_memory.build_memory_bank_from_memory`). The target fact is `memory_l`, and `memory_s` is a built-in hard negative.
2. Given `query` (prefixed with `[query_time]`, as in `build_prompt_query`), retrieve the top-5 facts and log retrieval metrics: target recall@5 (match `memory_content == memory_l`, while handling the 7 duplicate-string cases) and whether `memory_s` is ranked above `memory_l`.
3. Answer either in MCQ mode (accuracy = picks `Correct`; the trap distribution is diagnostic) or in open-ended mode, then judge "applies target fact" with the `judge_openended.py` prompt (inputs: `memory_l`, `options.Correct`, query, prediction).
4. Memory timestamps are in 2025 and `query_time`s are in the future (e.g., 2037). To use a smaller pool, subsample the bank per instance but always keep `memory_l` and `memory_s`.

## License

- The IMLogic repo (4open) has no LICENSE file and no license statement in its README, which only says "please cite the corresponding paper". It is an anonymous review release, so treat it as research use with citation, and ask the authors before redistributing.
- HaluMem (source of the dialogues and memory bank) is **CC BY-NC-ND 4.0** according to its HF dataset card: non-commercial, no derivatives, attribution required.
