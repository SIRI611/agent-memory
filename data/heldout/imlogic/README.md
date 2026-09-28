# IMLogic

IMLogic is a benchmark for evaluating whether a personalized assistant can use user-specific memories to resolve **implicit logical relevance** in query answering. The benchmark is designed around cases where a user query appears reasonable on the surface, but a different, previously stored memory should constrain or redirect the final answer.

This repository contains the benchmark data, the generation and verification prompts, and reference evaluation scripts for both **memory-level** and **conversation-level** settings. It supports both **multiple-choice QA (MCQ)** and **open-ended QA** evaluation.

## What Is Being Evaluated

IMLogic focuses on the following question:

> Given a current user query, can a model retrieve the **right past memory** and use it to produce a response that is logically aligned with the user's actual situation, rather than following surface-level preferences or generic advice?

The benchmark is built around pairs of memories:

- `m_s`: the semantic memory capturing a surface-level intention/preference
- `m_l`: the logical memory that should govern the answer

In other words, the model must identify that the relevant answer is not determined by the most obvious memory, but by the memory that is **query-conditioned and logically controlling**.

## Repository Contents

```text
IMLogic/
|- dataset/
|  |- dialogue/
|  |  `- HaluMem-Medium.jsonl
|  `- qa/
|     |- qa_user00.json
|     |- qa_user01.json
|     `- ...
|- generate/
|  |- memory_tagging.py
|  |- implicit_logical_memory_pair_mining.py
|  `- qa_generating.py
|- eval/
|  |- evaluate_common.py
|  |- evaluate_on_memory.py
|  |- evaluate_on_dialogue.py
|  |- evaluate_qa.py
|  `- judge_openended.py
`- README.md
```

Current release summary:

- `dataset/dialogue/HaluMem-Medium.jsonl` contains `20` dialogue records
- `dataset/qa/qa_userXX.json` contains `20` aligned QA files, one per dialogue record

The alignment rule is:

- dialogue item `0` corresponds to `qa_user00.json`
- dialogue item `1` corresponds to `qa_user01.json`
- and so on

## Benchmark Data Format

### Dialogue File

`dataset/dialogue/HaluMem-Medium.jsonl` is a JSONL file. Each line is one user trajectory and contains:

- `uuid`: user identifier
- `persona_info`: summary of persona metadata
- `sessions`: a list of multi-turn sessions

Each `session` typically includes:

- `start_time`, `end_time`
- `memory_points`: stored memories extracted or maintained by the source system
- `dialogue`: the original user-assistant interaction turns
- `questions`: auxiliary session-level questions in the source data

Each `memory_point` typically contains:

- `memory_content`
- `memory_type`
- `timestamp`
- `importance`
- `memory_source`
- `index`

Minimal schema sketch:

```json
{
  "uuid": "user-uuid",
  "persona_info": "...",
  "sessions": [
    {
      "start_time": "...",
      "end_time": "...",
      "memory_points": [
        {
          "index": 1,
          "memory_content": "...",
          "memory_type": "...",
          "timestamp": "...",
          "importance": 0.75
        }
      ],
      "dialogue": [
        {
          "role": "user",
          "content": "...",
          "timestamp": "...",
          "dialogue_turn": 0
        }
      ]
    }
  ]
}
```

### QA File

Each `dataset/qa/qa_userXX.json` file is a list of benchmark instances for the corresponding user.

Each QA item contains:

- `m_s`: semantic memory
- `m_l`: logical memory
- `query_type`: e.g., `Advice`, `Recommendation`, `Conversation`
- `query`: the user query to answer
- `options`: the MCQ candidates
- `query_time`: timestamp associated with the query
- `category`: evaluation category

The `options` object contains:

- `Correct`: the logically correct answer when `m_l` is used properly
- `Trap_Preference`: answer based on `m_s` only
- `Trap_Fabrication`: answer with hallucinated justification
- `Trap_Generic`: generic advice without relevant memory grounding

Minimal schema sketch:

```json
{
  "m_s": "...",
  "m_l": "...",
  "query_type": "Advice",
  "query": "...",
  "options": {
    "Correct": "...",
    "Trap_Preference": "...",
    "Trap_Fabrication": "...",
    "Trap_Generic": "..."
  },
  "query_time": "...",
  "category": "Goal/Mission Alignment"
}
```

For open-ended judgment, `m_l` is used as the logical memory in the judge prompt.

## Benchmark Construction Pipeline

The benchmark construction process is organized into three stages under `generate/`.

### 1. Memory Tagging

File: `generate/memory_tagging.py`

This script assigns each memory a top-level semantic tag using an LLM-based taxonomy. The current taxonomy includes:

- `Personal_Background`
- `Assets`
- `Past_Experience`
- `States`
- `Preferences`
- `Opinions`
- `Goals`
- `Plans`
- `Social_Relationships`
- `Others`

### 2. Implicit Logical Pair Mining

File: `generate/implicit_logical_memory_pair_mining.py`

This script mines candidate `(m_l, m_s)` pairs where:

- `m_l` is the logical memory that should constrain the answer
- `m_s` is the semantic memory that is surface-level relevant to the query

The mining prompt asks whether `m_s` is logically dominated by `m_l`, rather than merely being topically unrelated.

### 3. QA Generation and Verification

File: `generate/qa_generating.py`

This script generates benchmark questions and four candidate options. It also contains:

- a **generator prompt**
- a **judger prompt**
- a **reviser prompt**

These prompts are used to create and verify high-quality adversarial benchmark instances with:

- strong surface alignment to `m_s`
- zero leakage of `m_l` into the query
- balanced candidate options

## Evaluation Settings

IMLogic supports two evaluation settings.

### 1. Memory-Level Evaluation

File: `eval/evaluate_on_memory.py`

This setting assumes a memory bank already exists. The evaluation pipeline is:

1. build a memory bank from stored `memory_points`
2. retrieve relevant memories for the current query
3. answer the QA using an LLM

The script currently exposes a retrieval interface that users should implement:

```python
def retrieve_memories(query, memory_bank, qa_item, top_k):
    # TODO: implement memory retrieval from memory_bank.
    return []
```

### 2. Conversation-Level Evaluation

File: `eval/evaluate_on_conversation.py`

This setting starts from raw dialogue instead of a ready-made memory bank. The pipeline is:

1. extract memories from dialogue
2. retrieve relevant extracted memories
3. answer the QA using an LLM

The script currently exposes two interfaces that users should implement:

```python
def extract_memories_from_dialogue(dialogue_item):
    # TODO: implement memory extraction from dialogue turns.
    return []

def retrieve_memories(query, memory_bank, qa_item, top_k):
    # TODO: implement memory retrieval from extracted dialogue memories.
    return []
```

## QA Modes

The repository supports two answer modes.

### MCQ Mode

In MCQ mode, the model is given the query, retrieved context, and four candidate responses. It must output only the best option letter.

Implementation:

- prompt template: `eval/evaluate_common.py`
- runner: `eval/evaluate_on_memory.py` or `eval/evaluate_on_dialogue.py`
- CLI flag: `--answer-mode multiple_choice`

### Open-Ended Mode

In open-ended mode, the model is given the query and retrieved context and must generate one concise answer sentence.

Implementation:

- prompt template: `eval/evaluate_common.py`
- runner: `eval/evaluate_on_memory.py` or `eval/evaluate_on_dialogue.py`
- CLI flag: `--answer-mode open_ended`

## LLM Answering Prompts

The repository currently includes two evaluation-time prompts:

- **Cognitive Evaluation Agent** for MCQ selection
- **Cognitive Generative Agent** for open-ended generation

Both prompts are implemented in:

- `eval/evaluate_common.py`

They share the same basic structure:

- `Active_Root_Memory_Units`
- `global_state`
- retrieved context
- timestamped user query

The evaluation scripts expect an **OpenAI-compatible API**.

## Open-Ended LLM-as-Judge Evaluation

File: `eval/judge_openended.py`

This script evaluates open-ended model outputs using an LLM judge. For each QA item, it uses:

- `m_l` as the logical memory
- `options["Correct"]` as `Reference Answer`
- the original `query` as `User Question`
- the generated answer as `Model Prediction`

The judge prompt is intentionally strict:

- the answer must explicitly reflect the relevant constraint
- the answer must align with the logical direction of the reference answer
- generic answers should be marked incorrect

### Judge Output

The judge returns JSON of the form:

```json
{
  "is_correct": true,
  "reasoning": "..."
}
```

The saved result file includes:

- per-question judge result
- `m_l`
- `reference_answer`
- `model_prediction`
- aggregate accuracy statistics

## How to Run the Benchmark

### Prerequisites

The evaluation scripts assume:

- Python 3.10+
- `openai` Python package
- access to an OpenAI-compatible API endpoint

Typical environment variables:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=...
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="..."
$env:OPENAI_MODEL="..."
```

### 1. Memory-Level MCQ Evaluation

```bash
python eval/evaluate_on_memory.py \
  --answer-mode multiple_choice \
  --model gemini-3-pro-preview
```

### 2. Memory-Level Open-Ended Evaluation

```bash
python eval/evaluate_on_memory.py \
  --answer-mode open_ended \
  --model gemini-3-pro-preview
```

### 3. Conversation-Level MCQ Evaluation

```bash
python eval/evaluate_on_dialogue.py \
  --answer-mode multiple_choice \
  --model gemini-3-pro-preview
```

### 4. Conversation-Level Open-Ended Evaluation

```bash
python eval/evaluate_on_dialogue.py \
  --answer-mode open_ended \
  --model gemini-3-pro-preview
```

### 5. Judge Open-Ended Predictions

```bash
python eval/judge_openended.py \
  --predictions-file eval/evaluate_on_memory_results.json \
  --model gemini-3-pro-preview
```

### Useful Optional Flags

- `--user-index`: evaluate only one aligned user trajectory
- `--question-limit`: evaluate only the first `N` questions per user
- `--top-k`: maximum number of retrieved memories passed to the answer model
- `--global-state-file`: override the automatically constructed global state block
- `--output`: set a custom result file path

## Result Files

The evaluation scripts save JSON result files containing:

- `config`
- `summary`
- `users`

Per-user records include:

- question metadata
- model outputs
- candidate options for MCQ
- retrieved memories
- answer-level correctness for MCQ

Judge result files additionally include:

- LLM-as-judge verdicts for open-ended answers
- reasoning from the judge

## Reproducibility Notes

To reproduce the full IMLogic pipeline, researchers should specify:

1. the memory extraction method
2. the retrieval method
3. the answer model
4. the answer mode (`multiple_choice` or `open_ended`)
5. the judge model for open-ended evaluation

At the moment, the repository provides:

- benchmark instances
- generation scripts
- verification prompts
- evaluation scripts
- LLM-as-judge support for open-ended QA

The repository intentionally leaves the **memory extraction** and **memory retrieval** interfaces open, so different memory systems can be plugged into the same benchmark protocol.

## Planned Appendix

In the revision, we will add a dedicated appendix that formally describes:

- the benchmark format
- the benchmark construction process
- the evaluation protocol
- the distinction between `m_s` and query-conditioned `m_l` constraints

This appendix is intended to make IMLogic easier to use, analyze, and reproduce for future researchers.

## Citation and Usage

If you use IMLogic in your experiments, please cite the corresponding paper once the bibliographic entry is released. If you adapt the benchmark with a custom extractor, retriever, or judge model, we recommend documenting those components explicitly for reproducibility.
