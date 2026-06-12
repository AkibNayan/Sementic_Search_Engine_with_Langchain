import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import vector_store
from embeddings import embeddings


# Below is a minimal helper for demonstration purposes
def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]


file_path = "nke-10k-2023.pdf"
docs = load_pdf_pages(file_path)
# print(len(docs))


# Recursively splits a document using common separators like new lines
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

# print(len(all_splits))

# Index the chunks into the vector store
ids = vector_store.add_documents(documents=all_splits)


# Return documents based on similarity to a string query
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])


# Async query
async def main():
    results = await vector_store.asimilarity_search("When was Nike incorporated?")
    print(results[0])
main()

# Return scores:
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}")
print(doc)


# Return documents based on similarity to an embedded query:
embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])
