# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper
from sympy.physics.units import temperature
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B",temperature=0.9)
message="what is the rotational speed of earth"
out=pipe(message)
print(out)

'''LLAMA (check if access is granted)
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")'''
''' QWEN
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

pipe = pipeline("text-generation", model="Qwen/Qwen2-0.5B",temperature=0,do_sample=False)
hf=HuggingFacePipeline(pipeline=pipe)
prompt = ChatPromptTemplate.from_template("""
You are a precise Q&A assistant. Only give factual, concise answers. No extra text.
Question: {question}
Answer:
""")


chain= prompt | hf
query="How many hours are in a week"
out = chain.invoke({"question": query})
print(out.split("Answer:\n")[-1])
'''

'''GEMMA
from sympy.polys.polyconfig import query
from transformers import pipeline
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

model="google/gemma-2-2b-it"
pipe = pipeline(
    "text-generation",
    model=model,
    device="cpu",
    max_new_tokens=20,
)

hf=HuggingFacePipeline(pipeline=pipe)

def ans_slm(query):
    prompt=ChatPromptTemplate.from_template("""Answer the user queries in precise manner and only give the factual answer,
    Do not add extra explanations, greetings, or conversational text.
    if you don't know the answer just say "I don't know"
    Question: {question}
    Answer:
    """)

    chain=prompt | hf
    # query="what is the answer to question 10 in my maths book"
    out=chain.invoke({"question":query})
    print(out)
    # return out.split('Answer:')[-1].strip()

# ans_slm("what is the answer to question 10 in my maths book")
# ans_slm("what is the rotational speed of earth")
ans_slm("How many days are there in a month")
'''