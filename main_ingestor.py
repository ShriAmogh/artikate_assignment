from rag.data_ingestor.ingestion import FolderPDFIngestor
from config import chunk_size, overlap_size

ingestor = FolderPDFIngestor(
    folder_path="G:/Artikate_assignment/rag/data/",
    chroma_dir="vector_store/",
    collection_name="science_class_9",
    chunk_size= chunk_size,
    overlap_size= overlap_size
)

result = ingestor.ingest()
print(result)
