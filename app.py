from transformers import pipeline
import gradio as gr

# Load model from Hugging Face
model_name = "pruthvya/kokani-en-model"
translator = pipeline("translation", model=model_name)

# Define translation function
def kokani_to_english(text):
    result = translator(text, max_length=50, num_beams=5, early_stopping=True)
    return result[0]['translation_text']

# Create Gradio interface
demo = gr.Interface(
    fn=kokani_to_english,
    inputs="text",
    outputs="text",
    title="Kokani to English Translator",
    description="Translate Kokani text to English using a fine-tuned model.",
    examples=[
        ["माझी सेवा ठीक आसा"],
        ["तुम्ही कसे आहात?"]
    ],
    theme="default"
)

# Launch Gradio app
if __name__ == "__main__":
    demo.launch()
