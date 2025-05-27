# app.py

from transformers import pipeline
import gradio as gr
from fastapi import FastAPI
from uvicorn import Server, Config

# Load model from Hugging Face
model_name = "pruthvya/kokani-en-model"
translator = pipeline("translation", model=model_name)

# Gradio UI
def translate(text):
    result = translator(text, max_length=50, num_beams=5, early_stopping=True)
    return result[0]['translation_text']

# Web Interface
interface = gr.Interface(fn=translate, inputs="text", outputs="text", title="Kokani → English Translator")

# FastAPI endpoint (optional)
app = FastAPI(title="Kokani Translation API")

@app.get("/translate")
def translate_text(text: str):
    result = translator(text, max_length=50, num_beams=5, early_stopping=True)
    return {"input": text, "translation": result[0]["translation_text"]}

# To run both UI and API:
if __name__ == "__main__":
    import uvicorn
    interface.launch(share=False)
    uvicorn.run(app, host="0.0.0.0", port=8000)