from intent_cosine import detect_intent_pg
from llm_module import ans_llm
from slm_module import ans_slm
import time
from langchain.chains import RouterChain


# query=("Who are the members of the European")
def detect_model(query):
    start_time_pred = time.perf_counter()
    pred, intent=detect_intent_pg(query)
    if pred=="slm":
        response=ans_slm(query)
        if not response or response=="I don't know":
            response=ans_llm(query)
    elif pred=="rag":
        response=ans_llm(query)
    else:
        response="None"
    end_time_pred = time.perf_counter()

    tot=end_time_pred-start_time_pred
    return f"{response}and by intent recognition is {tot:.3f} and by cosine is {intent}"

# detect_model("What did the president say about economy")
# detect_model("Who are the members of the European")

