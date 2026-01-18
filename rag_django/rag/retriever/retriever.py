from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from chromadb import PersistentClient


class ChromaRetriever:

    def __init__(
        self,
        chroma_dir: str,
        collection_name: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.client = PersistentClient(path=chroma_dir)
        #self.collection = self.client.get_collection(name=collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.cross_encoder = CrossEncoder(reranker_model_name)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20
    ) -> List[Dict]:

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = self._format_results(results)

        reranked = self._rerank_with_cross_encoder(
            query=query,
            documents=retrieved,
            top_k=top_k
        )

        return reranked


    def _format_results(self, results) -> List[Dict]:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        formatted = []

        for doc, meta, dist in zip(documents, metadatas, distances):
            formatted.append({
                "content": doc,
                "metadata": meta,
                "vector_score": float(dist)
            })
        # print(formatted)
        # print()
        return formatted

    def _rerank_with_cross_encoder(
        self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:


        if not documents:
            return []

        pairs = [
            (query, doc["content"])
            for doc in documents
        ]

        scores = self.cross_encoder.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["cross_score"] = float(score)

        documents.sort(
            key=lambda x: x["cross_score"],
            reverse=True
        )
        # for doc in documents:
        #     print(doc)
        #     print()
        return documents[:top_k]
