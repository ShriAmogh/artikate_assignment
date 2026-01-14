from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .tasks import ingest_folder_task
from rag.services import ask_rag
from .models import IngestionJob
import uuid

def index(request):
    if request.method == "POST":
        files = request.FILES.getlist("files")
        upload_dir = "media/uploads/"

        fs = FileSystemStorage(location=upload_dir)

        for f in files:
            fs.save(f.name, f)

        doc_id = str(uuid.uuid4())

        request.session['doc_id'] = doc_id
        print(f"doc_id set in session: {doc_id}")

        job = IngestionJob.objects.create(
            status="pending",
            progress=0
        )

        # #  celery execution
        # ingest_folder_task.delay(
        #     folder_path=upload_dir,
        #     job_id=job.id
        # )

        #currently sync task
        ingest_folder_task(
            folder_path=upload_dir,
            job_id=job.id,
            doc_id = doc_id
        )


        return render(
            request,
            "web/index.html",
            {"job_id": job.id}
        )

    return render(request, "web/index.html")



def ask(request):
    answer = None
    sources = []

    if request.method == "POST":
        question = request.POST.get("question")
        doc_id = request.session.get('doc_id')
        print(f"checking doc id here   {doc_id}")
        print(f"doc_id retrieved from session: {doc_id}")
        response = ask_rag(question, doc_id)
        answer = response["answer"]
        sources = response["sources"]

    return render(request, "web/index.html", {
        "answer": answer,
        "sources": sources
    })

from django.http import JsonResponse

def ingestion_status(request, job_id):
    job = IngestionJob.objects.get(id=job_id)
    return JsonResponse({
        "status": job.status,
        "progress": job.progress,
        "message": job.message
    })


