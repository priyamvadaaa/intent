from sympy.polys.polyconfig import query
from transformers import pipeline
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import time
from langchain_core.runnables import RunnableLambda

pipe = pipeline("text-generation", model="Qwen/Qwen2-0.5B",temperature=0,do_sample=False,max_new_tokens=50)
hf=HuggingFacePipeline(pipeline=pipe)
prompt = ChatPromptTemplate.from_template("""
You are a precise Q&A assistant. Only give factual, concise answers.If you are unsure about answers just say "I don't know" No extra information and text.
Question: {question}
Answer:
""")

def ans_slm(query):
    start_time_slm = time.perf_counter()

    chain = prompt | hf
    # query = "How many hours are in a week"
    out = chain.invoke({"question": query})
    # print(out.split("Answer:\n")[-1])
    end_time_slm = time.perf_counter()
    tot_slm=end_time_slm - start_time_slm
    return f"{out.split("Answer:\n")[-1]} and time by slm is {tot_slm:.3f}"

# ans_slm("what is the answer to question 10 in my maths book")
# ans_slm("what is the rotational speed of earth")
# ans_slm("How many days are there in a month")


