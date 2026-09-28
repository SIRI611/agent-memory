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


def build_memory_bank_from_memory(dialogue_item: dict[str, Any]) -> list[dict[str, Any]]:
    memory_bank: list[dict[str, Any]] = []
    for session_index, session in enumerate(dialogue_item.get("sessions", [])):
        for memory in session.get("memory_points", []):
            memory_bank.append(
                {
                    "memory_content": memory.get("memory_content", ""),
                    "memory_type": memory.get("memory_type", ""),
                    "timestamp": memory.get("timestamp"),
                    "session_index": session_index,
                    "memory_index": memory.get("index"),
                }
            )
    return memory_bank


def retrieve_memories(
    query: str,
    memory_bank: list[dict[str, Any]],
    qa_item: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    # TODO: implement memory retrieval from memory_bank.
    _ = (query, memory_bank, qa_item, top_k)
    return []


def parse_args():
    parser = build_argument_parser(
        description="Evaluate QA with an existing memory bank plus LLM answering.",
        default_output_name="evaluate_on_memory_results.json",
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
        memory_bank_builder=build_memory_bank_from_memory,
        retriever=retrieve_memories,
        top_k=args.top_k,
        answer_mode=args.answer_mode,
        global_state_override=global_state_override,
        client=client,
        model=args.model,
        temperature=args.temperature,
        shuffle_seed=args.shuffle_seed,
        question_limit=args.question_limit,
        pipeline_name="on_memory",
    )
    print_summary(results)
    if not args.no_save:
        save_results(args.output, results)
        print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
