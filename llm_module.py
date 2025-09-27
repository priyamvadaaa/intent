from dotenv import load_dotenv
load_dotenv()
import os
import getpass
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
import time
from langchain_core.runnables import RunnableLambda



embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#sudo docker run --name llm -e POSTGRES_USER=llmuser -e POSTGRES_PASSWORD=llmslm -e POSTGRES_DB=slm -p 6024:5432 -d pgvector/pgvector:pg16
connection = "postgresql+psycopg://llmuser:llmslm@localhost:6024/slm"

# raw_docs=TextLoader("state_of_the_union.txt").load()
raw_docs=TextLoader("hamster.txt").load()

text_split=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs=text_split.split_documents(raw_docs)

vs = PGVector(
    embeddings=embeddings,
    collection_name="llm_embedding",
    connection=connection,
    use_jsonb=True,
)

# vs.add_documents(docs)

if not os.environ.get("GOOGLE_API_KEY"):
   os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for GeminiAI: ")

def ans_llm(query):
    start_time_llm = time.perf_counter()
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    prompt=ChatPromptTemplate.from_template("""You are a helpful assistant, answer the user queries based on the context provided,
    if you can't find the context try to answer as factually as you can: 
    Question: {question}
    Context : {context}
    Answer
    """)

    qa_chain= RetrievalQA.from_llm(
        llm,
        retriever=vs.as_retriever(),
        prompt=prompt
    )
    #query="how many gm in 1 kg"
    out=qa_chain.invoke({"query":query})
    end_time_llm = time.perf_counter()
    tot_llm=end_time_llm-start_time_llm
    return f"Answer from llm is{out} and by llm is {tot_llm:.3f}"

# ans_llm("how many gm in 1 kg")