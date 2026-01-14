from rag_django.rag.data_ingestor.ingestion import FolderPDFIngestor
from config import chunk_size, overlap_size

#for testing rag system before integratiing with django. 
ingestor = FolderPDFIngestor(
    folder_path="G:/Artikate_assignment/rag/data/",
    chroma_dir="vector_store/",
    collection_name="science_class_9",
    chunk_size= 500,
    chunk_overlap= 350
)

result = ingestor.ingest()
print(result)
