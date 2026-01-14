import os
import fitz
import uuid
from typing import List, Dict
#from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
torch.set_default_device("cpu")
from rag.data_ingestor.embedding import get_embedding_model 



class FolderPDFIngestor:
    def __init__(self, folder_path: str, chroma_dir: str, collection_name: str):
        self.folder_path = folder_path
        self.embedding_model = get_embedding_model()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=350,
            length_function=len,
            is_separator_regex=False,

        )

        self.client = PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name= collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.documents = []
        self.metadatas = []
        self.ids = []

    def _list_pdfs(self):
        return [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.lower().endswith(".pdf")
        ]

    def _load_pdf(self, path):
        doc = fitz.open(path)
        pages = []

        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append({"text": text, "page": i + 1})

        return pages

    def _prepare_chunks(self):
        for pdf_path in self._list_pdfs():
            pdf_name = os.path.basename(pdf_path)
            doc_id = str(uuid.uuid4())

            for page in self._load_pdf(pdf_path):
                chunks = self.splitter.split_text(page["text"])

                for idx, chunk in enumerate(chunks):
                    self.documents.append(chunk)
                    self.metadatas.append({
                        "doc_id": doc_id,
                        "source": pdf_name,
                        "page": page["page"],
                        "chunk_index": idx,
                    })
                    self.ids.append(f"{doc_id}_{page['page']}_{idx}")

    def _batched(self, batch_size):
        for i in range(0, len(self.documents), batch_size):
            yield i

    def ingest(self):
        self._prepare_chunks()

        if not self.documents:
            return {"status": "no_documents"}

        BATCH_SIZE = 64

        for i in self._batched(BATCH_SIZE):
            batch_docs = self.documents[i:i + BATCH_SIZE]
            batch_metas = self.metadatas[i:i + BATCH_SIZE]
            batch_ids = self.ids[i:i + BATCH_SIZE]

            embeddings = self.embedding_model.encode(
                batch_docs,
                normalize_embeddings=True,
                show_progress_bar=True
            )

            self.collection.add(
                documents=batch_docs,
                embeddings=embeddings.tolist(),
                metadatas=batch_metas,
                ids=batch_ids
            )

        return {
            "status": "success",
            "documents": len(set(m["doc_id"] for m in self.metadatas)),
            "chunks": len(self.documents)
        }

    def ingest_with_progress(self):
        """Generator for UI progress"""
        self._prepare_chunks()

        total = len(self.documents)
        if total == 0:
            yield 100
            return

        BATCH_SIZE = 64
        processed = 0

        for i in self._batched(BATCH_SIZE):
            batch_docs = self.documents[i:i + BATCH_SIZE]
            batch_metas = self.metadatas[i:i + BATCH_SIZE]
            batch_ids = self.ids[i:i + BATCH_SIZE]

            embeddings = self.embedding_model.encode(
                batch_docs,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            self.collection.add(
                documents=batch_docs,
                embeddings=embeddings.tolist(),
                metadatas=batch_metas,
                ids=batch_ids
            )

            processed += len(batch_docs)
            yield int((processed / total) * 100)
