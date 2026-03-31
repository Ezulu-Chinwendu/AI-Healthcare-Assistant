import faiss
import pandas as pd
import re
import asyncio
import logging
import numpy as np
from functools import lru_cache
from symspellpy import SymSpell
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

class Config:
    # Paths
    FAISS_INDEX_PATH     = "models/medquad_faiss.index"
    METADATA_CSV_PATH    = "models/medquad_metadata.csv"
    SPELL_DICT_PATH      = "frequency_dictionary_en_82_765.txt"

    # Retrieval
    TOP_K_RETRIEVE       = 30    # candidates from FAISS + BM25
    TOP_K_RERANK         = 6     # final chunks fed to LLM
    BM25_ALPHA           = 0.5   # weight for dense vs sparse fusion (0=all BM25, 1=all dense)
    MIN_SIMILARITY       = 0.35  # cosine similarity floor — below this → "no info"
    MIN_RERANK_SCORE     = 0.0   # cross-encoder score floor (increase to tighten)

    # Context
    MAX_CONTEXT_CHARS    = 3000

    # Generation
    LLM_MODEL            = "google/flan-t5-large"   # upgrade from base → large (still local)
    MAX_NEW_TOKENS       = 300

    # Query
    MIN_QUERY_WORDS      = 2
    EMBEDDING_CACHE_SIZE = 512


cfg = Config()


# --------------------------------------------------
# LOAD MODELS  (called once at startup)
# --------------------------------------------------

logger.info("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

logger.info("Loading cross-encoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

logger.info(f"Loading LLM: {cfg.LLM_MODEL}...")
llm = pipeline(
    "text2text-generation",
    model=cfg.LLM_MODEL,
    device=-1,
    max_new_tokens=cfg.MAX_NEW_TOKENS,
    do_sample=False,
)

logger.info("Loading SymSpell...")
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary(cfg.SPELL_DICT_PATH, term_index=0, count_index=1)


# --------------------------------------------------
# LOAD VECTOR DATABASE + BUILD BM25
# --------------------------------------------------

logger.info("Loading FAISS index and metadata...")
faiss_index = faiss.read_index(cfg.FAISS_INDEX_PATH)
chunk_df    = pd.read_csv(cfg.METADATA_CSV_PATH).reset_index(drop=True)

logger.info("Building BM25 index...")
tokenized_corpus = [str(t).lower().split() for t in chunk_df["text"]]
bm25_index       = BM25Okapi(tokenized_corpus)

logger.info("All models loaded successfully.")


# --------------------------------------------------
# SPELL CORRECTION
# --------------------------------------------------

def normalize_query(query: str) -> str:
    """Fix spelling using SymSpell compound lookup."""
    suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
    corrected   = suggestions[0].term if suggestions else query
    if corrected != query:
        logger.info(f"Spell correction: '{query}' → '{corrected}'")
    return corrected


# --------------------------------------------------
# CLEAN QUERY
# --------------------------------------------------

def clean_query(query: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    query = query.lower()
    query = re.sub(r"[^a-z0-9\s]", "", query)
    return query.strip()


# --------------------------------------------------
# QUERY VALIDATION
# --------------------------------------------------

NONSENSE_WORDS = {"lol", "hi", "hello", "hey", "test", "okay", "ok", "thanks"}

MEDICAL_KEYWORDS = {
    "symptom", "symptoms", "disease", "disorder", "condition", "syndrome",
    "treatment", "therapy", "medication", "drug", "dose", "dosage",
    "pain", "cause", "causes", "diagnosis", "diagnose", "cure", "prevent",
    "side", "effect", "effects", "risk", "risks", "sign", "signs",
    "infection", "virus", "bacteria", "cancer", "chronic", "acute",
    "surgery", "vaccine", "allergy", "blood", "heart", "lung", "kidney",
    "liver", "brain", "nerve", "bone", "skin", "diet", "exercise",
    "what", "how", "why", "when", "which", "is", "are", "can", "does",
}

def is_valid_question(query: str) -> tuple[bool, str]:
    """
    Returns (is_valid: bool, reason: str).
    Allows common question words so 'what is diabetes' passes.
    """
    words = set(query.lower().split())

    if len(words) < cfg.MIN_QUERY_WORDS:
        return False, "Your question is too short. Please provide more detail."

    if words <= NONSENSE_WORDS:
        return False, "Please ask a clear medical question."

    # Check for at least one medical or interrogative keyword
    if not words & MEDICAL_KEYWORDS:
        return False, (
            "Your question doesn't appear to be medical. "
            "Please ask about symptoms, treatments, conditions, etc."
        )

    return True, ""


# --------------------------------------------------
# EMBEDDING CACHE
# --------------------------------------------------

@lru_cache(maxsize=cfg.EMBEDDING_CACHE_SIZE)
def _cached_embed(query: str) -> np.ndarray:
    """Cache query embeddings to avoid re-encoding repeated queries."""
    return embedding_model.encode([query], normalize_embeddings=True)


# --------------------------------------------------
# HYBRID RETRIEVAL  (Dense FAISS + Sparse BM25)
# --------------------------------------------------

def retrieve_hybrid(query: str, top_k: int = cfg.TOP_K_RETRIEVE):
    """
    Fuse dense (cosine similarity via FAISS) and sparse (BM25) scores
    using a weighted linear combination.

    Returns:
        results    - DataFrame of top candidates
        top_scores - array of fused similarity scores (0–1 range)
    """
    n = len(chunk_df)

    # --- Dense retrieval ---
    query_emb        = _cached_embed(query)
    distances, idxs  = faiss_index.search(query_emb, top_k)
    # For L2-normalised embeddings: cosine_sim = 1 - L2²/2
    dense_sims       = np.clip(1.0 - distances[0] / 2.0, 0, 1)
    dense_scores_arr = np.zeros(n)
    for idx, sim in zip(idxs[0], dense_sims):
        if 0 <= idx < n:
            dense_scores_arr[idx] = sim

    # --- Sparse retrieval (BM25) ---
    raw_bm25         = bm25_index.get_scores(query.split())
    # Normalise BM25 to [0, 1]
    bm25_max         = raw_bm25.max()
    sparse_scores    = (raw_bm25 / bm25_max) if bm25_max > 0 else raw_bm25

    # --- Fusion ---
    alpha            = cfg.BM25_ALPHA
    fused            = alpha * dense_scores_arr + (1 - alpha) * sparse_scores
    top_ids          = np.argsort(fused)[::-1][:top_k]

    results          = chunk_df.iloc[top_ids].copy()
    results          = results.drop_duplicates(subset="text")
    top_scores       = fused[top_ids[:len(results)]]

    return results, top_scores


# --------------------------------------------------
# RERANKING
# --------------------------------------------------

def rerank_chunks(query: str, results: pd.DataFrame, top_k: int = cfg.TOP_K_RERANK):
    """Score every candidate with a cross-encoder and keep the best top_k."""
    pairs  = [(query, str(text)) for text in results["text"]]
    scores = reranker.predict(pairs)

    results          = results.copy()
    results["score"] = scores
    results          = results.sort_values("score", ascending=False)
    results          = results[results["score"] >= cfg.MIN_RERANK_SCORE]

    return results.head(top_k)


# --------------------------------------------------
# BUILD CONTEXT
# --------------------------------------------------

def build_context(results: pd.DataFrame) -> str:
    """
    Concatenate reranked chunks into a single context string,
    numbering each chunk for easier referencing.
    """
    parts = []
    for i, (_, row) in enumerate(results.iterrows(), 1):
        parts.append(f"[{i}] {str(row['text']).strip()}")
    context = "\n\n".join(parts)
    return context[:cfg.MAX_CONTEXT_CHARS]


# --------------------------------------------------
# PROMPT BUILDER
# --------------------------------------------------

SYSTEM_INSTRUCTIONS = """You are a medical information assistant.
Rules:
1. Answer ONLY using information from the numbered context below.
2. If the context does not contain the answer, say exactly: "The available medical sources do not provide this information."
3. Never suggest specific drug dosages.
4. Always recommend consulting a qualified doctor for personal medical advice.
5. Be concise, accurate, and factual."""

def build_prompt(context: str, query: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


# --------------------------------------------------
# FORMAT ANSWER
# --------------------------------------------------

def format_answer(answer: str) -> str:
    """Capitalise sentences and strip extra whitespace."""
    answer    = answer.strip()
    sentences = re.split(r"(?<=[.!?]) +", answer)
    sentences = [s.capitalize() for s in sentences if s]
    return " ".join(sentences)


# --------------------------------------------------
# CONFIDENCE CHECK
# --------------------------------------------------

def confidence_check(top_similarity: float, top_rerank_score: float) -> str | None:
    """
    Returns a warning string if confidence is low, otherwise None.
    Keeps the hallucination guard separate from hard rejection.
    """
    if top_similarity < cfg.MIN_SIMILARITY:
        return "The available medical sources do not provide this information."
    if top_rerank_score < 0.1:
        return (
            "Low confidence: the retrieved sources may not directly address "
            "your question. Please consult a healthcare professional."
        )
    return None


# --------------------------------------------------
# MAIN RAG FUNCTION
# --------------------------------------------------

def ask_medical_question(query: str) -> str:
    """
    Full RAG pipeline:
      1. Spell-correct & clean query
      2. Validate query
      3. Hybrid retrieve (dense + BM25)
      4. Confidence / hallucination guard
      5. Cross-encoder rerank
      6. Build prompt & generate answer
      7. Format & return with sources
    """

    # 1. Pre-process
    query = normalize_query(query)
    query = clean_query(query)
    logger.info(f"Processed query: '{query}'")

    # 2. Validate
    valid, reason = is_valid_question(query)
    if not valid:
        return reason

    # 3. Hybrid retrieval
    results, fused_scores = retrieve_hybrid(query, top_k=cfg.TOP_K_RETRIEVE)
    if results.empty:
        return "The available medical sources do not provide this information."

    top_similarity = float(fused_scores[0]) if len(fused_scores) else 0.0

    # 4. Rerank
    reranked = rerank_chunks(query, results, top_k=cfg.TOP_K_RERANK)
    if reranked.empty:
        return "The available medical sources do not provide this information."

    top_rerank_score = float(reranked["score"].iloc[0])

    # 5. Confidence guard
    warning = confidence_check(top_similarity, top_rerank_score)
    if warning and top_similarity < cfg.MIN_SIMILARITY:
        return warning   # hard reject — similarity too low

    # 6. Build context & prompt
    context = build_context(reranked)
    prompt  = build_prompt(context, query)
    logger.info("Generating answer...")

    # 7. Generate
    response = llm(prompt)
    answer   = response[0]["generated_text"].strip()
    answer   = format_answer(answer)

    # 8. Append low-confidence note if needed
    if warning:
        answer += f"\n\n⚠️ {warning}"

    # 9. Cite sources
    sources = ", ".join(reranked["source"].dropna().unique())
    disclaimer = "\n\n⚕️ This is general medical information only. Always consult a qualified healthcare professional."

    return f"{answer}\n\nSource(s): {sources}{disclaimer}"


# --------------------------------------------------
# ASYNC WRAPPER  (optional — for web / API use)
# --------------------------------------------------

async def ask_medical_question_async(query: str) -> str:
    """Non-blocking wrapper for use in async frameworks (FastAPI, etc.)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ask_medical_question, query)


# --------------------------------------------------
# CLI ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    print("Medical RAG Pipeline ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue
        print("\n" + ask_medical_question(user_input))
        print("\n" + "-" * 60 + "\n")