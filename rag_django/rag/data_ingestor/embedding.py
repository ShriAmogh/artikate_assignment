from sentence_transformers import SentenceTransformer

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )
    return _embedding_model
