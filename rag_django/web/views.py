from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .tasks import ingest_folder_task
from rag.services.services import ask_rag
from .models import IngestionJob
#import uuid
from rag.services.user_index import get_or_create_user_index
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@login_required(login_url="/login/")
def dashboard(request):
    if request.method == "POST":
        
        files = request.FILES.getlist("files")

        if not files:
            return render(
                request,
                "web/index.html",
                {"error" : "Add atleast one document before indexing."}
            )
        upload_dir = "media/uploads/"

        fs = FileSystemStorage(location=upload_dir)

        for f in files:
            fs.save(f.name, f)

        user = request.user
        user_index = get_or_create_user_index(user)

        # doc_id = str(uuid.uuid4())

        # request.session['doc_id'] = doc_id
        # print(f"doc_id set in session: {doc_id}")

        job = IngestionJob.objects.create(
            status="pending",
            progress=0
        )

        # #  celery execution
        # ingest_folder_task.delay(
        #     folder_path=upload_dir,
        #     job_id=job.id,
        #     doc_id = user_index.collection_name
        # )

        #currently sync task
        ingest_folder_task(
            folder_path=upload_dir,
            job_id=job.id,
            doc_id = user_index.collection_name
        )


        return render(
            request,
            "web/index.html",
            {"job_id": job.id}
        )

    return render(request, "web/index.html")

@login_required(login_url="/login/")
@csrf_exempt
def ask(request):
    answer = None
    sources = []

    if request.method == "POST":
        question = request.POST.get("question")
        if not question:
            return render(request, "web/index.html",
                          {"error": "Please enter a question according to the uploaded documents."})
        user = request.user
        user_index = get_or_create_user_index(user)
        doc_id = user_index.collection_name
        print(f"Currently using collection as  {doc_id}")
        response = ask_rag(question, doc_id)
        answer = response["answer"]
        sources = response["sources"]

    return render(request, "web/index.html", {
        "answer": answer,
        "sources": sources
    })


def ingestion_status(request, job_id):
    job = IngestionJob.objects.get(id=job_id)
    return JsonResponse({
        "status": job.status,
        "progress": job.progress,
        "message": job.message
    })


