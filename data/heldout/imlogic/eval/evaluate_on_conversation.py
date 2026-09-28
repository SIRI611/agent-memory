from __future__ import annotations

from typing import Any

from evaluate_common import (
    build_argument_parser,
    create_llm_client,
    evaluate_dataset,
    load_text,
    print_summary,
    save_results,
)


def flatten_dialogue_turns(dialogue_item: dict[str, Any]) -> list[dict[str, Any]]:
    dialogue_turns: list[dict[str, Any]] = []
    for session_index, session in enumerate(dialogue_item.get("sessions", [])):
        for turn_index, turn in enumerate(session.get("dialogue", [])):
            dialogue_turns.append(
                {
                    "session_index": session_index,
                    "turn_index": turn_index,
                    "role": turn.get("role", ""),
                    "content": turn.get("content", ""),
                    "timestamp": turn.get("timestamp"),
                    "dialogue_turn": turn.get("dialogue_turn"),
                }
            )
    return dialogue_turns


def extract_memories_from_dialogue(dialogue_item: dict[str, Any]) -> list[dict[str, Any]]:
    # TODO: implement memory extraction from dialogue turns.
    dialogue_turns = flatten_dialogue_turns(dialogue_item)
    _ = dialogue_turns
    return []


def retrieve_memories(
    query: str,
    memory_bank: list[dict[str, Any]],
    qa_item: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    # TODO: implement memory retrieval from extracted dialogue memories.
    _ = (query, memory_bank, qa_item, top_k)
    return []


def parse_args():
    parser = build_argument_parser(
        description="Evaluate QA by first extracting memories from dialogue, then using LLM answering.",
        default_output_name="evaluate_on_dialogue_results.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_state_override = load_text(args.global_state_file) if args.global_state_file else None
    client = create_llm_client(args.api_key, args.base_url)
    if client is None:
        raise RuntimeError("LLM client is not configured. Provide --api-key or OPENAI_API_KEY.")
    results = evaluate_dataset(
        dialogue_file=args.dialogue_file,
        qa_dir=args.qa_dir,
        user_index=args.user_index,
        memory_bank_builder=extract_memories_from_dialogue,
        retriever=retrieve_memories,
        top_k=args.top_k,
        answer_mode=args.answer_mode,
        global_state_override=global_state_override,
        client=client,
        model=args.model,
        temperature=args.temperature,
        shuffle_seed=args.shuffle_seed,
        question_limit=args.question_limit,
        pipeline_name="on_dialogue",
    )
    print_summary(results)
    if not args.no_save:
        save_results(args.output, results)
        print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
