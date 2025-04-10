# app.py
import gradio as gr
import numpy as np
from inference import DentalInference
from llm_integration import get_llm_response

inference_model = DentalInference()

def process_xray(patient_name, xray_image, comments):
    image_np = np.array(xray_image)
    labeled_image, detection_data = inference_model.predict(image_np)
    
    # Format detection results
    result_text = "\n".join([
        f"Quadrant {d['quadrant']}, Tooth {d['tooth']}: {d['disease']}"
        for d in detection_data
    ]) if detection_data else "No dental issues detected"
    
    explanation, _ = get_llm_response(detection_data, patient_name, comments)
    return labeled_image, result_text, explanation, detection_data

def respond_to_question(message, chat_history, detection_data, patient_name, comments):
    explanation, updated_history = get_llm_response(
        detection_data, 
        patient_name, 
        comments, 
        chat_history + [(message, "")]
    )
    updated_history.append((message, explanation))
    return "", updated_history

with gr.Blocks() as interface:
    gr.Markdown("# AI Dental Assistant")
    gr.Markdown("Upload dental X-ray for analysis and chat with dental AI assistant")

    with gr.Row():
        with gr.Column():
            patient_name = gr.Textbox(label="Patient Name", placeholder="Enter patient name...")
            xray_image = gr.Image(label="Upload X-ray", type="pil")
            comments = gr.Textbox(label="Additional Comments", placeholder="Any symptoms or notes?...", lines=3)
            submit_btn = gr.Button("Analyze X-ray")
            
        with gr.Column():
            result_image = gr.Image(label="Analyzed X-ray", interactive=False)
            result_text = gr.Textbox(label="Detection Results", interactive=False)
            explanation = gr.Textbox(label="Expert Explanation", interactive=False)
            
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="Follow-up Questions", height=300)
            msg = gr.Textbox(label="Type your question", placeholder="Ask about your dental health...")
            clear = gr.ClearButton([msg, chatbot])

    detection_data = gr.State()
    
    submit_btn.click(
        fn=process_xray,
        inputs=[patient_name, xray_image, comments],
        outputs=[result_image, result_text, explanation, detection_data]
    )
    
    msg.submit(
        fn=respond_to_question,
        inputs=[msg, chatbot, detection_data, patient_name, comments],
        outputs=[msg, chatbot]
    )

interface.launch()