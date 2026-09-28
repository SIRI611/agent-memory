from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_SYSTEM_PROMPT = "Follow the prompt exactly and return only the requested output."

MULTIPLE_CHOICE_PROMPT_TEMPLATE = """Role: Cognitive Evaluation Agent
You are an advanced conversational agent
equipped with a cognitive root memory system for personalization and logical reasoning.
Active_Root_Memory_Units:
Activated cognitive root memory units providing the logical framework for time-aware
personalization.
- Execution_Rules: The deterministic control laws and conflict resolution protocols.
- Personalized_Logical_Evidences: The temporal parameter matrix bounding the Agent's feasible decision region.
{global_state}
Retrieved Context:
The user's historical memories.
{context}
User Query:
The [Timestamp] at the beginning of the query indicates the exact time of the user's current request.
"{query}"
Candidate Responses:
{options_str}
Instructions:
Evaluate the candidate responses and select the optimal choice by adhering to the following rules:
1. Logic-Aware Personalization: Incorporate the Execution_Rules and Personalized_Logical_Evidences defined
in Active_Root_Memory_Units to contextualize and tailor the response.
2. Evidence-Grounded Alignment: Select the response that maximally aligns
with the user's actual needs, relying on the provided evidence.
Output Format:
Output only the letter of the best option (e.g., A, B, C, or D). Do not provide any explanation."""

OPEN_ENDED_PROMPT_TEMPLATE = """Role: Cognitive Generative Agent
You are an advanced conversational agent
equipped with a cognitive root memory system for personalization and logical reasoning.
Active_Root_Memory_Units:
Activated cognitive root memory units providing the logical framework for time-aware
personalization.
- Execution_Rules: The deterministic control laws and conflict resolution protocols.
- Personalized_Logical_Evidences: The temporal parameter matrix bounding the Agent's feasible decision region.
{global_state}
Retrieved Context:
The user's historical memories.
{context}
User Query:
The [Timestamp] at the beginning of the query indicates the exact time of the user's current request.
"{query}"
Instructions:
Generate a precise and accurate answer by adhering to the following rules:
1. Logic-Aware Personalization: Incorporate the Execution_Rules and Personalized_Logical_Evidences defined
in Active_Root_Memory_Units to contextualize and tailor the response.
2. Evidence-Grounded Alignment: Select the response that maximally aligns
with the user's actual needs, relying on the provided evidence.
Output Format:
Output ONLY a single, well-structured sentence (not more than 50 words). Do not provide any internal reasoning or explanation."""

MemoryBankBuilder = Callable[[dict[str, Any]], list[dict[str, Any]]]
Retriever = Callable[[str, list[dict[str, Any]], dict[str, Any], int], list[dict[str, Any]]]


@dataclass
class Prediction:
    mode: str
    option_key: str | None = None
    option_letter: str | None = None
    option_text: str = ""
    generated_text: str = ""
    candidate_options: list[dict[str, str]] | None = None
    reason: str = ""
    backend: str = ""
    raw_output: str = ""


def build_argument_parser(description: str, default_output_name: str) -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    default_dialogue = root / "dataset" / "dialogue" / "HaluMem-Medium.jsonl"
    default_qa_dir = root / "dataset" / "qa"
    default_output = root / "eval" / default_output_name

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dialogue-file",
        type=Path,
        default=default_dialogue,
        help="Path to the dialogue JSONL file.",
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=default_qa_dir,
        help="Directory containing qa_userXX.json files.",
    )
    parser.add_argument(
        "--user-index",
        type=int,
        default=None,
        help="Evaluate a single dialogue item. If omitted, evaluate all items.",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Only evaluate the first N questions for each user.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of recalled memories to pass to the answer stage.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=["multiple_choice", "open_ended"],
        default="multiple_choice",
        help="Answer the QA with either choice selection or free-form generation.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or "",
        help="API key for an OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or "",
        help="Base URL for an OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EVAL_MODEL_NAME") or os.getenv("OPENAI_MODEL") or "gemini-3-pro-preview",
        help="Model name for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM decoding temperature.",
    )
    parser.add_argument(
        "--global-state-file",
        type=Path,
        default=None,
        help="Optional text file used to override the auto-generated global state block.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Deterministic seed for shuffling multiple-choice options.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Where to save the evaluation result JSON.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the result JSON to disk.",
    )
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL line {line_number} in {path}") from exc
    return records


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def extract_user_number(path: Path) -> int:
    match = re.search(r"qa_user(\d+)\.json$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def resolve_qa_file(qa_dir: Path, user_index: int) -> Path:
    direct_path = qa_dir / f"qa_user{user_index:02d}.json"
    if direct_path.exists():
        return direct_path

    qa_files = sorted(qa_dir.glob("qa_user*.json"), key=extract_user_number)
    if user_index < 0 or user_index >= len(qa_files):
        raise FileNotFoundError(f"Cannot find QA file for user index {user_index} in {qa_dir}")
    return qa_files[user_index]


def create_llm_client(api_key: str, base_url: str) -> Any:
    if OpenAI is None or not api_key:
        return None

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def call_llm_text(client: Any, model: str, user_prompt: str, temperature: float) -> tuple[str | None, str | None]:
    if client is None:
        return None, "LLM client is not configured."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            return None, "LLM returned empty content."
        return content.strip(), None
    except Exception as exc:
        return None, str(exc)


def build_global_state(
    dialogue_item: dict[str, Any],
    memory_bank: list[dict[str, Any]],
    global_state_override: str | None,
) -> str:
    if global_state_override:
        return global_state_override

    persona_info = dialogue_item.get("persona_info", "").strip()
    session_count = len(dialogue_item.get("sessions", []))
    lines = [
        "Execution_Rules:",
        "- Prefer retrieved memories over generic assumptions.",
        "- If memories conflict, prioritize the most recent evidence-supported memory.",
        "- Do not fabricate facts that are not grounded in retrieved context.",
        "Personalized_Logical_Evidences:",
        f"- User UUID: {dialogue_item.get('uuid', 'unknown')}",
        f"- Session Count: {session_count}",
        f"- Available Memory Count: {len(memory_bank)}",
    ]
    if persona_info:
        lines.append(f"- Persona Summary: {persona_info}")
    return "\n".join(lines)


def format_context(recalled_memories: list[dict[str, Any]]) -> str:
    if not recalled_memories:
        return "(No retrieved memories.)"

    lines = []
    for index, memory in enumerate(recalled_memories, start=1):
        timestamp = memory.get("timestamp") or "Unknown time"
        memory_type = memory.get("memory_type") or "Unknown type"
        memory_content = memory.get("memory_content", "").strip()
        lines.append(f"{index}. [{timestamp}] ({memory_type}) {memory_content}")
    return "\n".join(lines)


def build_prompt_query(qa_item: dict[str, Any]) -> str:
    query = qa_item.get("query", "").strip()
    query_time = qa_item.get("query_time")
    if not query:
        return f"[{query_time}]" if query_time else ""
    if query_time and query_time not in query:
        return f"[{query_time}] {query}"
    return query


def sanitize_single_sentence(text: str) -> str:
    collapsed = " ".join(text.strip().split())
    if not collapsed:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", collapsed, maxsplit=1)[0].strip()
    words = sentence.split()
    if len(words) > 50:
        sentence = " ".join(words[:50]).rstrip(",;:")
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
    return sentence


def build_lettered_options(
    qa_item: dict[str, Any],
    user_index: int,
    question_index: int,
    shuffle_seed: int,
) -> tuple[str, dict[str, dict[str, str]]]:
    option_items = [
        (option_key, option_text)
        for option_key, option_text in qa_item.get("options", {}).items()
        if isinstance(option_text, str)
    ]

    rng = random.Random(f"{shuffle_seed}:{user_index}:{question_index}")
    rng.shuffle(option_items)

    letter_to_option: dict[str, dict[str, str]] = {}
    lines: list[str] = []
    for idx, (option_key, option_text) in enumerate(option_items):
        letter = OPTION_LETTERS[idx]
        letter_to_option[letter] = {"option_key": option_key, "option_text": option_text}
        lines.append(f"{letter}. {option_text}")
    return "\n".join(lines), letter_to_option


def parse_choice_letter(raw_output: str, valid_letters: set[str]) -> str | None:
    cleaned = raw_output.strip().upper()
    if cleaned in valid_letters:
        return cleaned
    match = re.search(r"\b([A-Z])\b", cleaned)
    if not match:
        return None
    letter = match.group(1)
    if letter not in valid_letters:
        return None
    return letter


def answer_multiple_choice_with_llm(
    qa_item: dict[str, Any],
    dialogue_item: dict[str, Any],
    memory_bank: list[dict[str, Any]],
    recalled_memories: list[dict[str, Any]],
    global_state_override: str | None,
    client: Any,
    model: str,
    temperature: float,
    user_index: int,
    question_index: int,
    shuffle_seed: int,
) -> Prediction:
    options_str, letter_to_option = build_lettered_options(
        qa_item=qa_item,
        user_index=user_index,
        question_index=question_index,
        shuffle_seed=shuffle_seed,
    )
    candidate_options = [
        {
            "letter": letter,
            "option_key": option_data["option_key"],
            "option_text": option_data["option_text"],
        }
        for letter, option_data in letter_to_option.items()
    ]

    if client is None:
        return Prediction(
            mode="multiple_choice",
            candidate_options=candidate_options,
            reason="LLM client is not configured.",
            backend="llm",
        )

    prompt = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        global_state=build_global_state(dialogue_item, memory_bank, global_state_override),
        context=format_context(recalled_memories),
        query=build_prompt_query(qa_item),
        options_str=options_str,
    )
    raw_output, error = call_llm_text(client, model, prompt, temperature)
    if error:
        return Prediction(
            mode="multiple_choice",
            candidate_options=candidate_options,
            reason=f"LLM call failed: {error}",
            backend="llm",
            raw_output=raw_output or "",
        )

    valid_letters = set(letter_to_option)
    selected_letter = parse_choice_letter(raw_output or "", valid_letters)
    if selected_letter is None:
        return Prediction(
            mode="multiple_choice",
            candidate_options=candidate_options,
            reason=f"LLM did not return a valid option letter: {raw_output!r}",
            backend="llm",
            raw_output=raw_output or "",
        )

    selected_option = letter_to_option[selected_letter]
    return Prediction(
        mode="multiple_choice",
        option_key=selected_option["option_key"],
        option_letter=selected_letter,
        option_text=selected_option["option_text"],
        candidate_options=candidate_options,
        reason="Selected by the multiple-choice cognitive evaluation prompt.",
        backend="llm",
        raw_output=raw_output or "",
    )


def answer_open_ended_with_llm(
    qa_item: dict[str, Any],
    dialogue_item: dict[str, Any],
    memory_bank: list[dict[str, Any]],
    recalled_memories: list[dict[str, Any]],
    global_state_override: str | None,
    client: Any,
    model: str,
    temperature: float,
) -> Prediction:
    if client is None:
        return Prediction(
            mode="open_ended",
            reason="LLM client is not configured.",
            backend="llm",
        )

    prompt = OPEN_ENDED_PROMPT_TEMPLATE.format(
        global_state=build_global_state(dialogue_item, memory_bank, global_state_override),
        context=format_context(recalled_memories),
        query=build_prompt_query(qa_item),
    )
    raw_output, error = call_llm_text(client, model, prompt, temperature)
    if error:
        return Prediction(
            mode="open_ended",
            reason=f"LLM call failed: {error}",
            backend="llm",
            raw_output=raw_output or "",
        )

    generated_text = sanitize_single_sentence(raw_output or "")
    return Prediction(
        mode="open_ended",
        generated_text=generated_text,
        reason="Generated by the open-ended cognitive generation prompt.",
        backend="llm",
        raw_output=raw_output or "",
    )


def answer_question(
    qa_item: dict[str, Any],
    dialogue_item: dict[str, Any],
    memory_bank: list[dict[str, Any]],
    recalled_memories: list[dict[str, Any]],
    global_state_override: str | None,
    answer_mode: str,
    client: Any,
    model: str,
    temperature: float,
    user_index: int,
    question_index: int,
    shuffle_seed: int,
) -> Prediction:
    if answer_mode == "open_ended":
        return answer_open_ended_with_llm(
            qa_item=qa_item,
            dialogue_item=dialogue_item,
            memory_bank=memory_bank,
            recalled_memories=recalled_memories,
            global_state_override=global_state_override,
            client=client,
            model=model,
            temperature=temperature,
        )

    return answer_multiple_choice_with_llm(
        qa_item=qa_item,
        dialogue_item=dialogue_item,
        memory_bank=memory_bank,
        recalled_memories=recalled_memories,
        global_state_override=global_state_override,
        client=client,
        model=model,
        temperature=temperature,
        user_index=user_index,
        question_index=question_index,
        shuffle_seed=shuffle_seed,
    )


def evaluate_single_user(
    user_index: int,
    dialogue_item: dict[str, Any],
    qa_items: list[dict[str, Any]],
    memory_bank_builder: MemoryBankBuilder,
    retriever: Retriever,
    top_k: int,
    answer_mode: str,
    global_state_override: str | None,
    client: Any,
    model: str,
    temperature: float,
    shuffle_seed: int,
    question_limit: int | None,
) -> dict[str, Any]:
    memory_bank = memory_bank_builder(dialogue_item)
    predictions: list[dict[str, Any]] = []
    answered_count = 0
    correct_count = 0
    scored_count = 0

    if question_limit is not None:
        qa_items = qa_items[:question_limit]

    for question_index, qa_item in enumerate(qa_items):
        recalled_memories = retriever(
            qa_item.get("query", ""),
            memory_bank,
            qa_item,
            top_k,
        )
        prediction = answer_question(
            qa_item=qa_item,
            dialogue_item=dialogue_item,
            memory_bank=memory_bank,
            recalled_memories=recalled_memories,
            global_state_override=global_state_override,
            answer_mode=answer_mode,
            client=client,
            model=model,
            temperature=temperature,
            user_index=user_index,
            question_index=question_index,
            shuffle_seed=shuffle_seed,
        )

        gold_answer = qa_item.get("correct_answer") or qa_item.get("options", {}).get("Correct", "")
        is_answered = bool(prediction.option_key) if answer_mode == "multiple_choice" else bool(prediction.generated_text)
        is_correct: bool | None
        if answer_mode == "multiple_choice":
            scored_count += 1
            is_correct = prediction.option_text == gold_answer if is_answered else False
            if is_correct:
                correct_count += 1
        else:
            is_correct = None

        if is_answered:
            answered_count += 1

        predictions.append(
            {
                "question_index": question_index,
                "query_type": qa_item.get("query_type"),
                "category": qa_item.get("category"),
                "query_time": qa_item.get("query_time"),
                "query": qa_item.get("query"),
                "prediction": {
                    "mode": prediction.mode,
                    "backend": prediction.backend,
                    "option_key": prediction.option_key,
                    "option_letter": prediction.option_letter,
                    "option_text": prediction.option_text,
                    "generated_text": prediction.generated_text,
                    "candidate_options": prediction.candidate_options,
                    "reason": prediction.reason,
                    "raw_output": prediction.raw_output,
                },
                "gold_answer": gold_answer,
                "is_answered": is_answered,
                "is_correct": is_correct,
                "recalled_memories": recalled_memories,
            }
        )

    total_questions = len(qa_items)
    accuracy_all = (correct_count / total_questions) if total_questions and answer_mode == "multiple_choice" else None
    accuracy_answered = (correct_count / answered_count) if answered_count and answer_mode == "multiple_choice" else None
    return {
        "user_index": user_index,
        "uuid": dialogue_item.get("uuid"),
        "memory_bank_size": len(memory_bank),
        "question_count": total_questions,
        "answered_count": answered_count,
        "scored_count": scored_count,
        "correct_count": correct_count if answer_mode == "multiple_choice" else None,
        "accuracy_all": accuracy_all,
        "accuracy_answered": accuracy_answered,
        "predictions": predictions,
    }


def build_summary(user_results: list[dict[str, Any]], answer_mode: str) -> dict[str, Any]:
    total_users = len(user_results)
    total_questions = sum(item["question_count"] for item in user_results)
    answered_count = sum(item["answered_count"] for item in user_results)
    scored_count = sum(item["scored_count"] for item in user_results)
    correct_count = sum(item["correct_count"] or 0 for item in user_results)
    if answer_mode == "multiple_choice":
        accuracy_all = (correct_count / total_questions) if total_questions else None
        accuracy_answered = (correct_count / answered_count) if answered_count else None
        final_correct_count: int | None = correct_count
    else:
        accuracy_all = None
        accuracy_answered = None
        final_correct_count = None
    return {
        "user_count": total_users,
        "question_count": total_questions,
        "answered_count": answered_count,
        "scored_count": scored_count,
        "correct_count": final_correct_count,
        "accuracy_all": accuracy_all,
        "accuracy_answered": accuracy_answered,
    }


def evaluate_dataset(
    dialogue_file: Path,
    qa_dir: Path,
    user_index: int | None,
    memory_bank_builder: MemoryBankBuilder,
    retriever: Retriever,
    top_k: int,
    answer_mode: str,
    global_state_override: str | None,
    client: Any,
    model: str,
    temperature: float,
    shuffle_seed: int,
    question_limit: int | None,
    pipeline_name: str,
) -> dict[str, Any]:
    dialogue_items = load_jsonl(dialogue_file)
    if user_index is not None:
        if user_index < 0 or user_index >= len(dialogue_items):
            raise IndexError(
                f"user_index={user_index} is out of range for {dialogue_file} "
                f"(size={len(dialogue_items)})"
            )
        indices = [user_index]
    else:
        indices = list(range(len(dialogue_items)))

    user_results: list[dict[str, Any]] = []
    for current_index in indices:
        qa_file = resolve_qa_file(qa_dir, current_index)
        qa_items = load_json(qa_file)
        user_result = evaluate_single_user(
            user_index=current_index,
            dialogue_item=dialogue_items[current_index],
            qa_items=qa_items,
            memory_bank_builder=memory_bank_builder,
            retriever=retriever,
            top_k=top_k,
            answer_mode=answer_mode,
            global_state_override=global_state_override,
            client=client,
            model=model,
            temperature=temperature,
            shuffle_seed=shuffle_seed,
            question_limit=question_limit,
        )
        user_result["qa_file"] = str(qa_file)
        user_results.append(user_result)

    return {
        "config": {
            "pipeline": pipeline_name,
            "dialogue_file": str(dialogue_file),
            "qa_dir": str(qa_dir),
            "user_index": user_index,
            "question_limit": question_limit,
            "answer_mode": answer_mode,
            "top_k": top_k,
            "model": model,
            "temperature": temperature,
            "shuffle_seed": shuffle_seed,
            "has_global_state_override": bool(global_state_override),
        },
        "summary": build_summary(user_results, answer_mode),
        "users": user_results,
    }


def save_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


def print_summary(results: dict[str, Any]) -> None:
    summary = results["summary"]
    config = results["config"]
    print(f"Pipeline: {config['pipeline']}")
    print(f"Answer mode: {config['answer_mode']}")
    print(f"Users: {summary['user_count']}")
    print(f"Questions: {summary['question_count']}")
    print(f"Answered: {summary['answered_count']}")
    if summary["correct_count"] is None:
        print("Correct: N/A")
        print("Accuracy(all): N/A")
        print("Accuracy(answered): N/A")
    else:
        print(f"Correct: {summary['correct_count']}")
        print(f"Accuracy(all): {summary['accuracy_all']:.4f}")
        if summary["accuracy_answered"] is None:
            print("Accuracy(answered): N/A")
        else:
            print(f"Accuracy(answered): {summary['accuracy_answered']:.4f}")
