"""Cosine similarity between every query and every stored memory with all-MiniLM-L6-v2.

MiniLM is the smaller of the two embedders InMind reports. Run with the
.venv-embed interpreter, which has sentence-transformers installed.
"""

from __future__ import annotations

import json

from sentence_transformers import SentenceTransformer

from jev_persist.corpus import load_config
from jev_persist.query_relevance import pools, sha
from jev_persist.routing import EMBED_PATH

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    targets, memories = pools(load_config())
    model = SentenceTransformer(MODEL, device="cpu")
    memory_vectors = model.encode([m["text"] for m in memories], normalize_embeddings=True, batch_size=64)
    queries = sorted({t[field] for t in targets for field in ("query", "naive_query")})
    query_vectors = model.encode(queries, normalize_embeddings=True, batch_size=64)
    sims = {}
    for query, vector in zip(queries, query_vectors):
        scores = memory_vectors @ vector
        sims[sha(query)] = {m["text_hash"]: float(s) for m, s in zip(memories, scores)}
    EMBED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBED_PATH.write_text(json.dumps({"model": MODEL, "sims": sims}))
    print(f"queries={len(queries)} memories={len(memories)} -> {EMBED_PATH}")


if __name__ == "__main__":
    main()
