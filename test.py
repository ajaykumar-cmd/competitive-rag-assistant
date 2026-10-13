from common import load_documents, split_documents

docs = load_documents("data")
print("Documents:", len(docs))
for d in docs[:5]:
    print(d.metadata)

chunks = split_documents(docs)
print("Chunks:", len(chunks))