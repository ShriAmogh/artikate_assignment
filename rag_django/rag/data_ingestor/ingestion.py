import os
import fitz
import uuid
from typing import List, Dict
from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
torch.set_default_device("cpu")
from rag.data_ingestor.embedding import get_embedding_model

# Optional: only import if table extraction is enabled
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class FolderPDFIngestor:
    def __init__(
        self, 
        folder_path: str, 
        chroma_dir: str, 
        collection_name: str,
        extract_tables: bool = False 
    ):
        self.folder_path = folder_path
        self.embedding_model = get_embedding_model()
        self.extract_tables = extract_tables and PDFPLUMBER_AVAILABLE

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=350,
            length_function=len,
            is_separator_regex=False,
        )

        self.client = PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
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

    def _extract_tables(self, pdf_path: str) -> List[Dict]:
        """Extract tables from PDF using pdfplumber"""
        if not self.extract_tables:
            return []
        
        tables_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            # Convert to markdown
                            markdown = self._table_to_markdown(table)
                            # Create searchable text
                            searchable = self._table_to_searchable_text(table)
                            
                            if searchable:  # Only add if has content
                                tables_data.append({
                                    'page': page_num + 1,
                                    'markdown': markdown,
                                    'searchable': searchable,
                                    'table_idx': table_idx
                                })
        except Exception as e:
            print(f"Warning: Table extraction failed for {pdf_path}: {e}")
        
        return tables_data

    def _table_to_markdown(self, table: List[List]) -> str:
        """Convert table to markdown format"""
        if not table or len(table) < 1:
            return ""
        
        headers = table[0]
        markdown = "| " + " | ".join(str(h) if h else "" for h in headers) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        for row in table[1:]:
            markdown += "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |\n"
        
        return markdown

    def _table_to_searchable_text(self, table: List[List]) -> str:
        """Create searchable text from table"""
        if not table or len(table) < 2:
            return ""
        
        text_parts = []
        headers = table[0]
        
        for row in table[1:]:
            for idx, cell in enumerate(row):
                if cell and idx < len(headers) and headers[idx]:
                    text_parts.append(f"{headers[idx]}: {cell}")
        
        return " | ".join(text_parts)

    def _prepare_chunks(self):
        for pdf_path in self._list_pdfs():
            pdf_name = os.path.basename(pdf_path)
            doc_id = str(uuid.uuid4())

            # Process regular text chunks
            for page in self._load_pdf(pdf_path):
                chunks = self.splitter.split_text(page["text"])

                for idx, chunk in enumerate(chunks):
                    self.documents.append(chunk)
                    self.metadatas.append({
                        "doc_id": doc_id,
                        "source": pdf_name,
                        "page": page["page"],
                        "chunk_index": idx,
                        "type": "text"
                    })
                    self.ids.append(f"{doc_id}_{page['page']}_text_{idx}")

            # Process tables if enabled
            if self.extract_tables:
                tables = self._extract_tables(pdf_path)
                for table in tables:
                    self.documents.append(table['searchable'])
                    self.metadatas.append({
                        "doc_id": doc_id,
                        "source": pdf_name,
                        "page": table['page'],
                        "chunk_index": table['table_idx'],
                        "type": "table",
                        "table_markdown": table['markdown']
                    })
                    self.ids.append(f"{doc_id}_table_{table['page']}_{table['table_idx']}")

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
            "chunks": len(self.documents),
            "tables": len([m for m in self.metadatas if m.get("type") == "table"])
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