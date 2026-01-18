from celery import shared_task
import os
from .models import IngestionJob
from rag.data_ingestor.ingestion import FolderPDFIngestor


@shared_task(bind=True)
def ingest_folder_task(self, folder_path: str, job_id: int, doc_id: str):
    job = IngestionJob.objects.get(id=job_id)
    job.status = "running"
    job.progress = 0
    job.save()

    try:
        # Persist each user's Chroma DB under vector_store/<collection_name>
        chroma_dir = os.path.join("vector_store", doc_id)
        os.makedirs(chroma_dir, exist_ok=True)

        ingestor = FolderPDFIngestor(
            folder_path=folder_path,
            chroma_dir=chroma_dir,
            collection_name=doc_id,
        )

        current = 0
        for step_progress in ingestor.ingest_with_progress():
            current = step_progress
            job.progress = current
            job.save(update_fields=["progress"])

        # ingestion_with_progress populated ingestor.metadatas/documents
        documents_count = len(set(m["doc_id"] for m in getattr(ingestor, "metadatas", [])))
        chunks_count = len(getattr(ingestor, "documents", []))

        job.status = "completed"
        job.progress = 100
        job.message = f"ingested {documents_count} documents ({chunks_count} chunks) into collection={doc_id} at {chroma_dir}"
        job.save()

        print(f"DEBUG: {job.message}")

    except Exception as e:
        job.status = "failed"
        job.message = str(e)
        job.save()
        raise
