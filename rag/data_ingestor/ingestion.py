import os
import fitz
import uuid
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FolderPDFIngestor:


    def __init__(
        self,
        folder_path: str,
        chroma_dir: str,
        collection_name: str,
        chunk_size: int,
        overlap_size: int,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.folder_path = folder_path
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size
        )

        self.client = PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )


    def _list_pdfs(self) -> List[str]:
        return [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.lower().endswith(".pdf")
        ]

    def _load_pdf(self, path: str) -> List[Dict]:
        doc = fitz.open(path)
        pages = []

        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append({
                    "text": text,
                    "page": i + 1
                })
        return pages


    def ingest(self):
        documents = []
        metadatas = []
        ids = []

        for pdf_path in self._list_pdfs():
            pdf_name = os.path.basename(pdf_path)
            doc_id = str(uuid.uuid4())

            pages = self._load_pdf(pdf_path)

            for page in pages:
                chunks = self.splitter.split_text(page["text"])

                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_{page['page']}_{idx}"

                    documents.append(chunk)
                    metadatas.append({
                        "doc_id": doc_id,
                        "source": pdf_name,
                        "page": page["page"],
                        "chunk_index": idx,
                    })
                    ids.append(chunk_id)

        if not documents:
            return {"status": "no_documents"}

        embeddings = self.embedding_model.encode(
            documents,
            batch_size=64,
            show_progress_bar=True
        )

        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )

        return {
            "status": "success",
            "documents": len(set(m["doc_id"] for m in metadatas)),
            "chunks": len(documents)
        }
