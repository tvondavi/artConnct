import requests
import json
from flask import Flask, render_template, request, jsonify
from workhorse import fromRAG_gen
from pprint import pprint

app = Flask(__name__)

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json',
}


@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET','POST'])
def generate_prompt():
    prompt = request.form["msg"]
    input = prompt
    # return generate_Response(input)
    return fromRAG_gen(input)

##############------ If we want to run the model with just the model information and no RAG ------#####################
# def generate_Response(text):
    # data = {
    #     "model": "SocBot",
    #     "prompt": text,
    #     "stream": False
    # }

    # response = requests.post(url, headers=headers, data=json.dumps(data))
    # if response.status_code == 200:
    #     response_text = response.text
    #     data = json.loads(response_text)
    #     actual_response = data["response"]
    #     return actual_response
    # else:
    #     print("Error: ", response.status_code, response.text)


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 3000)