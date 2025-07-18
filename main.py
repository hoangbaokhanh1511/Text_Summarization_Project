from flask import Flask, request, render_template, jsonify
from model.inference import Summarizer

app = Flask(__name__)
summarizer = Summarizer()
@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')
@app.route('/summarize', methods = ['POST'])
def summarize():
    data = request.get_json()
    input_text = data.get('text', '')
    if input_text:
        summary = summarizer.summarize(input_text, max_length=1024, min_length=60)  
    else:
        summary = "No text provided for summarization."
    return jsonify({'summary': summary})
