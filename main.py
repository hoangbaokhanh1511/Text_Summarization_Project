from flask import Flask, request, render_template
from model.inference import Summarizer

app = Flask(__name__)
summarizer = Summarizer()  # Load model khi khởi động

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    if request.method == 'POST':
        input_text = request.form['text']
        if input_text:
            summary = summarizer.summarize(input_text)
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
