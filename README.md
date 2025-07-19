# 🏆 **TEXT-SUMMARIZATION** 

## 🔍 Project Overview

Text summarization is a classic and challenging problem in Natrual Language Model (NLP), where the system must understand, compress and regenerate human-level summaries. There are two main types: 
    
- **Extractive Summarization:** Selecting key sentences from the origin text. 
- **Abstractive Summarixation (used in this project):** Generating new sentences that convey the essence of the source 

## 🚀 Model Selection and Evaluation
Before deciding on the final model, i conducted a comparative evaluation of three powerful transformer-based summarization models: **Pegasus**, **Bart** and **T5**. 

All models were evaluated using the **CNN/DailyMail dataset** and scored based on **ROUGE** metrics to measure summarization quality. After thorough analysis, **Pegasus** was selected for its superior performance in capturing the semantics of long-form news articles and generating highly fluent summaries. 

## 🧠 Final Approach

The final system using **PEGASUS model fine-tuned on news summarization tasks**, capable of generating high-quality English summaries of international news articles. The implementation leverages HuggingFace's Transformer library for seamless model intergration and evaluation.  

## 📌 Installation
```bash
git clone https://github.com/hoangbaokhanh1511/Text_Summarization_Project.git
cd Text_Summarization_Project
python -m venv .venv 
.venv\Scripts\Activate
pip install -r requirements.txt
```