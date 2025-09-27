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

        result=detect_model(query)
        return {"query":query, "response":result},200



api.add_resource(Home,'/')
api.add_resource(Intent,"/intent")

if __name__=="__main__":
    app.run(debug=True)