from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PORT = 54325
HOST = '0.0.0.0'

print("モデル準備完了")

app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello():
    return jsonify({"response":"開通済み"})

if __name__ == "__main__":
    app.run(debug=True, host=HOST, port=PORT)
