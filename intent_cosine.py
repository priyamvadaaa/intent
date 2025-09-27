import numpy as np
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import time

trigger_phrase = [
    "based on the book",
    "based on the document",
    "based on my document",
    "based on my text",
    "document i provide",
    "with the help of my text",
    "context provided",
    "using the document",
    "using my text",
    "context",
    "document",
]
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

connection = "postgresql+psycopg://intent:second@localhost:6022/new"

trigger_docs = [
    Document(page_content=phrase, metadata={"label": "trigger"})
    for phrase in trigger_phrase
]

ts = PGVector(
    embeddings=embeddings,
    collection_name="trigger_phrases",
    connection=connection,
    use_jsonb=True,
)

ts.add_documents(trigger_docs)


def cosine_similarity(vec1,vec2):
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

def detect_intent_pg(query,threshold=0.65):
    start_time_cosine = time.perf_counter()

    query_lower=query.lower()

    #excat string match
    for phrase in trigger_phrase:
        if phrase in query_lower:
            # print(f"Exact phrase found in query{phrase}")
            return "rag"

    # embedding match
    query_emb=embeddings.embed_query(query)
    retriever=ts.as_retriever()
    docs=retriever.invoke(query)

    best_sim=-1
    best_phrase=None

    for doc in docs:
        phrase_embedding=embeddings.embed_query(doc.page_content)
        similarity=cosine_similarity(query_emb,phrase_embedding)
        # print(f"Similarity with {doc.page_content}: {similarity:.3f}")
        if similarity>best_sim:
            best_sim=similarity
            best_phrase=doc.page_content

    end_time_cosine = time.perf_counter()
    tot_cos=end_time_cosine-start_time_cosine
    if best_sim>=threshold:
        # print(f"emb match found with {best_phrase} (similarity={best_sim:.3f})")
        return "rag"

    # print(f"No trigger phrase detected for query: '{query}'")
    return "slm"

#
# queries = [
#         "Tell me about blackholes",
#         "Explain quantum entanglement using the document I provide",
#         "What is photosynthesis",
#         "Tell me about dogs with the help of my textbook"
#     ]
#
# for q in queries:
#     intent = detect_intent_pg(q)
#     print(f"Query: '{q}' → Detected intent: {intent}")


