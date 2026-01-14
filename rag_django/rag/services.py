from rag.retriever.retriever import ChromaRetriever
from rag.ans_builder.beautify_answer import AnswerBeautifier

beautifier = AnswerBeautifier()


def ask_rag(question: str, doc_id: str):
    if not doc_id:
        return {
            "answer": "No session found. Please upload documents first.",
            "sources": []
        }
    
    retriever = ChromaRetriever(
        chroma_dir="vector_store/",
        collection_name=doc_id
    )
    
    results = retriever.retrieve(
        query=question,
        fetch_k=20,
        top_k=5
    )

    if not results:
        return {
            "answer": "No relevant information found.",
            "sources": []
        }
    #to view all results
    for result in results:
        print(f"\n Retrieved doc : {result['content'][:200]}")

    beautified_answer = beautifier.generate_answer(
        question=question,
        top_documents=results
    )
    print(f"\n beautified_answer : {beautified_answer}\n")
    return beautified_answer
