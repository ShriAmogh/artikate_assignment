# artikate_assignment

A Retrieval-Augmented Generation (RAG) application built using Django, ChromaDB, Sentence Transformers, and Google Gemini, enabling users to upload PDFs, index them, and ask grounded questions strictly based on their documents.


Application Flow (End-to-End)
1. "/login/" --> routes to the login page (if user exist in local storage )
2. "/signup/" --> to create the user and store the auth in local storage (using django User model)
3. "/dashboard/" --> main page where user uploads documents and gets saved according to the user id (collection name) in the vector storage.
   PDF → Text → Chunks → Embeddings → ChromaDB
4. "/ask/" --> to ask the question according to the document (currently using vector search and re-ranking method)
   


Steps:
-User name and hashed password are stored in the Postgressql. 
    
-login checks for the username (username is unique, easy to read )
    
 -PDFs parsed using PyMuPDF
  
  -Text chunked using RecursiveCharacterTextSplitter
  
  -Embeddings generated in safe batches and gets saves separately, no two user documents get merged.  
  
  -Stored in ChromaDB with metadata:

    document name
    
    page number
    
    chunk index


**Retrieval Pipeline**
  Question → Embedding → Vector Search → Reranking


  Top-k chunks retrieved via cosine similarity

  CrossEncoder reranks results

  Highest confidence chunks selected

**Answer Beautification**

  Retrieved chunks passed to Gemini

  Prompt enforces:

    No hallucination
    
    No paraphrasing lists
    
    No truncation

  Outputs structured answer + sources

**Final Output**

  User sees:
  
  ✅ Extracted Answer
  
  📌 Source Document + Page Number




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



