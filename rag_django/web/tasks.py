from celery import shared_task
from .models import IngestionJob
from rag.data_ingestor.ingestion import FolderPDFIngestor

@shared_task(bind=True)
def ingest_folder_task(self, folder_path: str, job_id: int, doc_id : str):
    job = IngestionJob.objects.get(id=job_id)
    job.status = "running"
    job.progress = 0
    job.save()

    try:
        ingestor = FolderPDFIngestor(
            folder_path=folder_path,
            chroma_dir="vector_store/",
            collection_name= doc_id
        )

        total_steps = 100
        current = 0

        for step_progress in ingestor.ingest_with_progress():
            current = step_progress
            job.progress = current
            job.save(update_fields=["progress"])

        job.status = "completed"
        job.progress = 100
        job.save()

    except Exception as e:
        job.status = "failed"
        job.message = str(e)
        job.save()
        raise
