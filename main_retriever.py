from rag.retriever.retriever import ChromaRetriever
from rag.ans_builder.beautify_answer import AnswerBeautifier


def main():
    retriever = ChromaRetriever(
        chroma_dir="vector_store/",
        collection_name="science_class_9"
    )

    beautifier = AnswerBeautifier()

    print("RAG Assistant started")
    print("Type 'exit' to quit \n")

    while True:
        try:
            query = input("Enter question: ").strip()

            if not query:
                continue

            if query.lower() in {"exit", "quit"}:
                print("Exiting RAG system...")
                break

            results = retriever.retrieve(
                query=query,
                fetch_k=20,
                top_k=5
            )

            if not results:
                print("No relevant context found.\n")
                continue

            top_doc = results[0] 

            final_answer = beautifier.generate_answer(
                question=query,
                top_document=top_doc
            )

            print("\nAnswer:")
            print(final_answer["answer"])
            print("\nSources:")
            for src in final_answer["sources"]:
                print(f"- {src}")
            print("\n" + "-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n Keyboard interrupt received. Exiting safely...")
            break

        except Exception as e:
            print("Error occurred:", str(e))
            print("Continuing...\n")


if __name__ == "__main__":
    main()
