# artikate_assignment

A Retrieval-Augmented Generation (RAG) application built using Django, ChromaDB, Sentence Transformers, and Google Gemini, enabling users to upload PDFs, index them, and ask grounded questions strictly based on their documents.


Application Flow (End-to-End)
    1. Upload PDFs

  User uploads PDFs via Django UI

  Files stored in media/uploads/

2. Indexing Pipeline
  PDF → Text → Chunks → Embeddings → ChromaDB


Steps:

  PDFs parsed using PyMuPDF
  
  Text chunked using RecursiveCharacterTextSplitter
  
  Embeddings generated in safe batches
  
  Stored in ChromaDB with metadata:

    document name
    
    page number
    
    chunk index
    
    Indexing is append-only — new documents extend the same collection.

3. Retrieval Pipeline
  Question → Embedding → Vector Search → Reranking


  Top-k chunks retrieved via cosine similarity

  CrossEncoder reranks results

  Highest confidence chunks selected

4. Answer Beautification

  Retrieved chunks passed to Gemini

  Prompt enforces:

    No hallucination
    
    No paraphrasing lists
    
    No truncation

  Outputs structured answer + sources

5. Final Output

  User sees:
  
  ✅ Extracted Answer
  
  📌 Source Document + Page Number

Authentication & Persistence (Future Scope)

  Planned enhancements:
  
    Django authentication
    
    One Chroma collection per user
    
    Documents persist across sessions
    
    Incremental uploads without re-indexing

To use Async Indexing with Celery 

The architecture supports async indexing using:

  Celery for background jobs
  
  Redis as broker
  
  USe : ingest_folder_task.delay(upload_dir, user_id)


Currently disabled for simplicity.
Enabling it requires only deployment-level changes.



First Install the dependencies

  pip install -r requirements.txt

Create a .env file:

  GOOGLE_API_KEY=your_google_gemini_api_key

To run this : 

git clone https://github.com/ShriAmogh/artikate_assignment.git

cd rag_django

python manage.py migrate

python .\manage.py runserver

Open: http://127.0.0.1:8000



