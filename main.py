from flask import Flask, request
from flask_restful import Resource, Api
from intent_slm_llm import detect_model
import time


app=Flask(__name__)
api=Api(app)

class Home(Resource):
    def get(self):
        return "Hello, from the APi"

class Intent(Resource):
    def post(self):
        data=request.get_json()
        query=data.get("query","")
        if not query:
            return {"error":"No query"},400
        start_time_main = time.perf_counter()

        result=detect_model(query)
        end_time_main = time.perf_counter()
        tot_main= round((end_time_main - start_time_main), 3)
        return {"query":query, "response":result, "time by api":tot_main},200



api.add_resource(Home,'/')
api.add_resource(Intent,"/intent")

if __name__=="__main__":
    app.run(debug=True)