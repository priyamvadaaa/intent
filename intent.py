from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# ======== Embeddings ========
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ======== Trigger Phrases ========
TRIGGER_PHRASES = [
    "based on the book",
    "based on the document",
    "based on my text",
    "document i provide",
    "with the help of my text",
    "context provided",
    "using the document",
    "using my text"
    "context"
]

trigger_docs = [
    Document(page_content=phrase, metadata={"label": "trigger"})
    for phrase in TRIGGER_PHRASES
]

# ======== PostgreSQL Connection ========
connection = "postgresql+psycopg://intent:second@localhost:6022/new"

trigger_store = PGVector(
    embeddings=embeddings,
    collection_name="trigger_phrases",
    connection=connection,
    use_jsonb=True,
)

# ======== Add trigger phrases to PGVector (run once) ========
# trigger_store.add_documents(trigger_docs)


# ======== Intent Detection (only check similarity if phrase is present) ========
def detect_intent_with_pgvector(query: str):
    query_lower = query.lower()

    # First check if query contains any trigger phrase
    for phrase in TRIGGER_PHRASES:
        if phrase in query_lower:
            print(f"\nQuery contains trigger phrase: '{phrase}'")
            return "general_rag"

    # If no trigger phrase present, it's an SLM query
    print(f"\nNo trigger phrase detected in query: '{query}'")
    return "slm"


# ======== Example Usage ========
if __name__ == "__main__":
    queries = [
        "Tell me about blackholes",
        "Explain quantum entanglement using the document I provide",
        "What is photosynthesis",
        "Tell me about dogs with the help of my textbook"
    ]

    for q in queries:
        intent = detect_intent_with_pgvector(q)
        print(f"Query: '{q}' → Detected intent: {intent}")
