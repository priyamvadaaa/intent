from dotenv import load_dotenv
load_dotenv()

import os
import time
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

CACHE_FILE = "llm_cache.json"

# -----------------------------
# Persistent Cache Load
# -----------------------------
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        _llm_cache = json.load(f)
else:
    _llm_cache = {}

# -----------------------------
# Load LLM Chain (once)
# -----------------------------
def load_llm_chain():
    print("Loading documents...")
    # Load documents from a file or folder
    loader = TextLoader("state_of_the_union.txt", encoding="utf-8")  # replace with your file
    documents = loader.load()

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # or your preferred embeddings
    connection = "postgresql+psycopg://llmuser:llmslm@localhost:6024/slm"
    vs = PGVector(
            embeddings=embeddings,
            collection_name="llm_embedding",
            connection=connection,
            use_jsonb=True,
        )
    print("Creating vector store...")
    vs.add_documents(docs)
    print("Loading LLM model...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", model_kwargs={"temperature": 0.7})

    print("Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vs.as_retriever(),
        chain_type="stuff"  # or "map_reduce", "refine" depending on your use-case
    )

    return qa_chain

print("Loading LLM and QA chain... (this happens only once)")
qa_chain = load_llm_chain()

# -----------------------------
# Query Function
# -----------------------------
def ans_llm(query):
    if query in _llm_cache:
        print("Cache hit!")
        return _llm_cache[query]

    start_time_llm = time.perf_counter()
    out = qa_chain.invoke({"query": query})  # works now with RetrievalQA
    end_time_llm = time.perf_counter()

    tot_llm = end_time_llm - start_time_llm
    result = f"Answer from LLM is {out} and time taken is {tot_llm:.3f} seconds"

    _llm_cache[query] = result

    # Save cache persistently
    with open(CACHE_FILE, "w") as f:
        json.dump(_llm_cache, f)

    print(f"Cache stored for query: {query}")
    return result


# -----------------------------
# Test
# -----------------------------
print(ans_llm("what did the president say based on document"))

# from dotenv import load_dotenv
# load_dotenv()
# import os
# import getpass
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import TextLoader
# from langchain.chains import RetrievalQA
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_postgres import PGVector
# from langchain_core.caches import InMemoryCache
# from langchain_core.globals import set_llm_cache
# import time
#
# # set_llm_cache(InMemoryCache())  # Enable caching
#
# # Persistent globals
# _llm = None
# _qa_chain = None
# _vs = None
#
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# connection = "postgresql+psycopg://llmuser:llmslm@localhost:6024/slm"
#
# raw_docs = TextLoader("hamster.txt").load()
# text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_split.split_documents(raw_docs)
#
# _vs = PGVector(
#     embeddings=embeddings,
#     collection_name="llm_embedding",
#     connection=connection,
#     use_jsonb=True,
# )
#
# if not os.environ.get("GOOGLE_API_KEY"):
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for GeminiAI: ")
#
# def load_llm_chain():
#     global _llm, _qa_chain
#     if _llm is None:
#         _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", streaming=True)
#         prompt = ChatPromptTemplate.from_template("""You are a helpful assistant, answer the user queries based on the context provided,
#         if you can't find the context try to answer as factually as you can:
#         Question: {question}
#         Context : {context}
#         Answer
#         """)
#         _qa_chain = RetrievalQA.from_llm(
#             _llm,
#             retriever=_vs.as_retriever(),
#             prompt=prompt
#         )
#     return _qa_chain
#
# def ans_llm(query):
#     qa_chain = load_llm_chain()
#     start_time_llm = time.perf_counter()
#     out = qa_chain.invoke({"query": query})
#     end_time_llm = time.perf_counter()
#     tot_llm = end_time_llm - start_time_llm
#     print(f"Answer from llm is {out} and by llm is {tot_llm:.3f}")
#     return f"Answer from llm is {out} and by llm is {tot_llm:.3f}"
#
#
# # from dotenv import load_dotenv
# # load_dotenv()
# # import os
# # import getpass
# # from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_community.document_loaders import TextLoader
# # from langchain.chains import RetrievalQA
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_postgres import PGVector
# # from langchain_core.output_parsers import StrOutputParser
# # import time
# # from langchain_core.runnables import RunnableLambda
# #
# # from langchain_core.caches import InMemoryCache
# # from langchain_core.globals import set_llm_cache
# #
# # set_llm_cache(InMemoryCache())
# #
# # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# # #sudo docker run --name llm -e POSTGRES_USER=llmuser -e POSTGRES_PASSWORD=llmslm -e POSTGRES_DB=slm -p 6024:5432 -d pgvector/pgvector:pg16
# # connection = "postgresql+psycopg://llmuser:llmslm@localhost:6024/slm"
# #
# # # raw_docs=TextLoader("state_of_the_union.txt").load()
# # raw_docs=TextLoader("hamster.txt").load()
# #
# # text_split=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
# # docs=text_split.split_documents(raw_docs)
# #
# # vs = PGVector(
# #     embeddings=embeddings,
# #     collection_name="llm_embedding",
# #     connection=connection,
# #     use_jsonb=True,
# # )
# #
# # # vs.add_documents(docs)
# #
# # if not os.environ.get("GOOGLE_API_KEY"):
# #    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for GeminiAI: ")
# #
# # def ans_llm(query):
# #     start_time_llm = time.perf_counter()
# #     llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",streaming=True)
# #     prompt=ChatPromptTemplate.from_template("""You are a helpful assistant, answer the user queries based on the context provided,
# #     if you can't find the context try to answer as factually as you can:
# #     Question: {question}
# #     Context : {context}
# #     Answer
# #     """)
# #
# #     qa_chain= RetrievalQA.from_llm(
# #         llm,
# #         retriever=vs.as_retriever(),
# #         prompt=prompt
# #     )
# #     #query="how many gm in 1 kg"
# #     out=qa_chain.invoke({"query":query})
# #     end_time_llm = time.perf_counter()
# #     tot_llm=end_time_llm-start_time_llm
# #     print(f"Answer from llm is{out} and by llm is {tot_llm:.3f}")
# #     return f"Answer from llm is{out} and by llm is {tot_llm:.3f}"
# #
# # # ans_llm("how many gm in 1 kg")
# # # ans_llm("what did the president say based on document")
# # ans_llm("Where can we order free covid tests based on the context")