from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PORT = 54325
HOST = '0.0.0.0'

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")

def generate_prompt(text):
    return "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
        bos_token=tokenizer.bos_token,
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=text,
        e_inst=E_INST,
    )

if torch.cuda.is_available():
    model = model.to("cuda")

print("モデル準備完了")

app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello():
    text = request.form.get('text', None)
    print(f"input: {text}")
    if request.method != 'POST' or not text:
        return {"error":"リクエストが不正です。"}
    else:
        prompt = generate_prompt(request.form['text'])
        
    with torch.no_grad():
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
    print(f'output: {output}')
    return jsonify({"response":output})

if __name__ == "__main__":
    app.run(debug=True, host=HOST, port=PORT)
