from version_3_chroma_DB import get_vectorstore, build_chain

store = get_vectorstore()      # first run builds the index, later runs reuse it
chain = build_chain(store)

print("Ask a question (type 'exit' to quit)")
while True:
    question = input("\nYou: ").strip()
    if question.lower() in ("exit", "quit"):
        break
    if not question:
        continue
    print("\nAnswer:", chain.invoke(question))