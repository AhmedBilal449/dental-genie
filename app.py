# app.py
import gradio as gr
from inference import DentalInference

# Instantiate the inference class
inference_model = DentalInference()

def process_xray(patient_name, xray_image, comments=""):
    labeled_image, result_text = inference_model.predict(xray_image)
    return labeled_image, result_text

interface = gr.Interface(
    fn=process_xray,
    inputs=[
        gr.Textbox(label="Patient Name"),
        gr.Image(label="Upload X-ray", type="numpy"),
        gr.Textbox(label="Add Comments (Optional)", lines=2)
    ],
    outputs=[
        gr.Image(label="Labeled X-ray"),
        gr.Textbox(label="Result")
    ],
    title="Dental X-ray Analysis",
    description="Upload a dental X-ray to detect teeth, quadrants, and diseases."
)

interface.launch()
