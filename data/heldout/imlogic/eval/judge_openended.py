from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evaluate_common import create_llm_client, load_json, resolve_qa_file, save_results


JUDGE_PROMPT_TEMPLATE = """Role: Expert Memory Evaluator
You are an expert evaluator for a personalized memory assistant. Your goal is to determine if the [Model Prediction] correctly
applies the user's specific past constraints
to answer their current query.
Criteria:
A correct answer must satisfy ALL of the
following:
1. Specific Grounding: The prediction
MUST incorporate the specific fact or constraint from [Memory l].
2. Logical Alignment: The core recommendation must match the logical
direction of the [Reference Answer].
It must use [Memory l] to appropriately constrain, guide, or warn the user
against conflicting actions.
JUDGING Rules:
- Be strict about the inclusion of the specific constraint from [Memory l]. The
model must prove it retrieved the right
memory.
- If the prediction offers a compromise,
it is correct if it still explicitly leverages the constraint from [Memory l]
as the primary baseline.
- Mark as Incorrect if the prediction
gives generic advice, misses the specific detail in [Memory l], or blindly
agrees with a conflicting query.
Inputs:
Memory l: {memory_l}
Reference Answer: {reference}
User Question: {query}
Model Prediction: {prediction}
Output Format (JSON Only):
{{
  "is_correct": bool,
  "reasoning": "A concise explanation of why the answer is correct or why it failed."
}}"""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate open-ended predictions with an LLM-as-Judge prompt."
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        required=True,
        help="Path to an open-ended prediction result JSON produced by evaluate_on_memory or evaluate_on_dialogue.",
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=root / "dataset" / "qa",
        help="Directory containing qa_userXX.json files.",
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
        help="Judge model name for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge decoding temperature.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "eval" / "judge_openended_results.json",
        help="Where to save the judged result JSON.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the judged result JSON.",
    )
    return parser.parse_args()


def call_judge_json(client: Any, model: str, prompt: str, temperature: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only valid JSON that matches the requested schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            return None, "Judge returned empty content."
        return json.loads(content), None
    except Exception as exc:
        return None, str(exc)


def normalize_is_correct(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def build_summary(user_results: list[dict[str, Any]]) -> dict[str, Any]:
    question_count = sum(user["question_count"] for user in user_results)
    answered_count = sum(user["answered_count"] for user in user_results)
    judged_count = sum(user["judged_count"] for user in user_results)
    correct_count = sum(user["correct_count"] for user in user_results)
    accuracy_all = (correct_count / question_count) if question_count else None
    accuracy_answered = (correct_count / answered_count) if answered_count else None
    accuracy_judged = (correct_count / judged_count) if judged_count else None
    return {
        "user_count": len(user_results),
        "question_count": question_count,
        "answered_count": answered_count,
        "judged_count": judged_count,
        "correct_count": correct_count,
        "accuracy_all": accuracy_all,
        "accuracy_answered": accuracy_answered,
        "accuracy_judged": accuracy_judged,
    }


def judge_user_predictions(
    client: Any,
    model: str,
    temperature: float,
    qa_dir: Path,
    user_result: dict[str, Any],
) -> dict[str, Any]:
    user_index = user_result["user_index"]
    qa_file = Path(user_result.get("qa_file") or resolve_qa_file(qa_dir, user_index))
    qa_items = load_json(qa_file)

    judged_predictions: list[dict[str, Any]] = []
    answered_count = 0
    judged_count = 0
    correct_count = 0

    for prediction_record in user_result.get("predictions", []):
        question_index = prediction_record["question_index"]
        qa_item = qa_items[question_index]
        generated_text = prediction_record.get("prediction", {}).get("generated_text", "").strip()
        is_answered = bool(generated_text)
        if is_answered:
            answered_count += 1

        memory_l = qa_item.get("memory_l", "")
        reference = qa_item.get("correct_answer") or qa_item.get("options", {}).get("Correct", "")
        query = qa_item.get("query", "")

        judge_output: dict[str, Any] | None = None
        judge_error: str | None = None
        normalized_is_correct: bool | None = None
        reasoning = ""

        if is_answered:
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                memory_l=memory_l,
                reference=reference,
                query=query,
                prediction=generated_text,
            )
            judge_output, judge_error = call_judge_json(client, model, prompt, temperature)
            if judge_output is not None:
                normalized_is_correct = normalize_is_correct(judge_output.get("is_correct"))
                reasoning = str(judge_output.get("reasoning", "")).strip()
            else:
                reasoning = judge_error or "Judge did not return valid JSON."
            if normalized_is_correct is not None:
                judged_count += 1
            if normalized_is_correct is True:
                correct_count += 1
        else:
            reasoning = "Model prediction is empty."

        judged_predictions.append(
            {
                **prediction_record,
                "judge": {
                    "memory_l": memory_l,
                    "reference_answer": reference,
                    "model_prediction": generated_text,
                    "is_correct": normalized_is_correct,
                    "reasoning": reasoning,
                    "raw_output": judge_output,
                    "error": judge_error,
                },
            }
        )

    question_count = len(user_result.get("predictions", []))
    return {
        "user_index": user_index,
        "uuid": user_result.get("uuid"),
        "qa_file": str(qa_file),
        "question_count": question_count,
        "answered_count": answered_count,
        "judged_count": judged_count,
        "correct_count": correct_count,
        "accuracy_all": (correct_count / question_count) if question_count else None,
        "accuracy_answered": (correct_count / answered_count) if answered_count else None,
        "accuracy_judged": (correct_count / judged_count) if judged_count else None,
        "predictions": judged_predictions,
    }


def print_summary(results: dict[str, Any]) -> None:
    summary = results["summary"]
    print(f"Users: {summary['user_count']}")
    print(f"Questions: {summary['question_count']}")
    print(f"Answered: {summary['answered_count']}")
    print(f"Judged: {summary['judged_count']}")
    print(f"Correct: {summary['correct_count']}")
    if summary["accuracy_all"] is None:
        print("Accuracy(all): N/A")
    else:
        print(f"Accuracy(all): {summary['accuracy_all']:.4f}")
    if summary["accuracy_answered"] is None:
        print("Accuracy(answered): N/A")
    else:
        print(f"Accuracy(answered): {summary['accuracy_answered']:.4f}")
    if summary["accuracy_judged"] is None:
        print("Accuracy(judged): N/A")
    else:
        print(f"Accuracy(judged): {summary['accuracy_judged']:.4f}")


def main() -> None:
    args = parse_args()
    client = create_llm_client(args.api_key, args.base_url)
    if client is None:
        raise RuntimeError("LLM client is not configured. Provide --api-key or OPENAI_API_KEY.")

    predictions_data = load_json(args.predictions_file)
    answer_mode = predictions_data.get("config", {}).get("answer_mode")
    if answer_mode != "open_ended":
        raise ValueError(
            f"Predictions file must come from open_ended evaluation, got answer_mode={answer_mode!r}."
        )

    user_results = [
        judge_user_predictions(
            client=client,
            model=args.model,
            temperature=args.temperature,
            qa_dir=args.qa_dir,
            user_result=user_result,
        )
        for user_result in predictions_data.get("users", [])
    ]

    results = {
        "config": {
            "predictions_file": str(args.predictions_file),
            "qa_dir": str(args.qa_dir),
            "judge_model": args.model,
            "temperature": args.temperature,
            "source_pipeline": predictions_data.get("config", {}).get("pipeline"),
            "source_answer_mode": answer_mode,
        },
        "summary": build_summary(user_results),
        "users": user_results,
    }

    print_summary(results)
    if not args.no_save:
        save_results(args.output, results)
        print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
