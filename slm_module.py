from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
import time

# Enable LangChain caching
set_llm_cache(InMemoryCache())

# Persistent globals
print("Loading SLM model at import time...")
_slm_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2-0.5B",
    temperature=0,
    do_sample=False,
    max_new_tokens=50
)
_hf = HuggingFacePipeline(pipeline=_slm_pipeline)
_prompt = ChatPromptTemplate.from_template("""
    You are a precise Q&A assistant. Only give factual, concise answers.
    If you are unsure about answers just say "I don't know". No extra information and text.
    Question: {question}
    Answer:
""")
_slm_cache = {}
print("SLM model loaded.")

def ans_slm(query):
    if query in _slm_cache:
        print("SLM cache hit!")
        return _slm_cache[query]

    start_time_slm = time.perf_counter()
    chain = _prompt | _hf
    out = chain.invoke({"question": query})
    end_time_slm = time.perf_counter()

    tot_slm = end_time_slm - start_time_slm
    answer = out.split("Answer:\n")[-1].strip()

    result = f"{answer} and time by slm is {tot_slm:.3f}"
    _slm_cache[query] = result
    print(result)
    return result




# from sympy.polys.polyconfig import query
# from transformers import pipeline
# from langchain_huggingface import ChatHuggingFace
# from langchain_huggingface import HuggingFacePipeline
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
# import time
# from langchain_core.runnables import RunnableLambda
# from langchain.globals import set_llm_cache
# from langchain_community.cache import InMemoryCache
#
#
#
# pipe = pipeline("text-generation", model="Qwen/Qwen2-0.5B",temperature=0,do_sample=False,max_new_tokens=50)
# hf=HuggingFacePipeline(pipeline=pipe)
# prompt = ChatPromptTemplate.from_template("""
#     You are a precise Q&A assistant. Only give factual, concise answers.If you are unsure about answers just say "I don't know" No extra information and text.
#     Question: {question}
#     Answer:
#     """)
#
# # Enable LangChain in-memory cache globally
# set_llm_cache(InMemoryCache())
#
# def ans_slm(query):
#     start_time_slm = time.perf_counter()
#     chain = prompt | hf
#     out = chain.invoke({"question": query})
#     end_time_slm = time.perf_counter()
#     tot_slm = end_time_slm - start_time_slm
#     answer = out.split("Answer:\n")[-1]
#     print(f"{answer} and time by slm is {tot_slm:.3f}")
#     return f"{answer} and time by slm is {tot_slm:.3f}"
#
# # ans_slm("what is the answer to question 10 in my maths book")
# # ans_slm("what is the rotational speed of earth")
# # ans_slm("How many days are there in a month")
# ans_slm("tell me about small language models")
