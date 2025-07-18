from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

class Summarizer:
    def __init__(self, model_name="google/pegasus-cnn_dailymail"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text, max_length=1024, min_length=60):
        inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Xử lý các khả năng khác nhau
        summary = summary.replace("/n", " ")
        summary = summary.replace("\u003cn\u003e", " ")
        summary = re.sub(r'\s*<n>\s*', ' ', summary)
        summary = ' '.join(summary.split())  # loại bỏ khoảng trắng thừa

        return summary
        