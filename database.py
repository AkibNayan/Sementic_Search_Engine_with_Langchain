from langchain_core.vectorstores import InMemoryVectorStore
from embeddings import embeddings


vector_store = InMemoryVectorStore(embedding=embeddings)



