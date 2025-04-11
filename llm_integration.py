import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ['GEMINI_API_KEY'])

def get_llm_response(detections, patient_name="", comments="", chat_history=None):
    if not detections and not chat_history:
        return "No dental issues detected! Keep up with regular checkups!", []
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    base_prompt = f"""Act as a dental expert assistant. Current patient: {patient_name or 'unnamed patient'}
Initial findings: {[d['disease'] for d in detections] if detections else 'No issues found'}
Patient comments: {comments or 'None'}"""
    
    if chat_history is None:
        prompt = f"""{base_prompt}
        
        Provide:
        1. Overview of findings
        2. Explanations in layman's terms
        3. Recommended actions
        4. Prevention advice
        Keep response under 250 words."""
    else:
        prompt = f"""{base_prompt}
        
        Chat History:
        {format_chat_history(chat_history)}
        
        Current Question: {chat_history[-1][0]}
        
        Provide a concise, expert answer to the current question focusing on:
        - Relevant dental concepts
        - Practical advice
        - Safety considerations
        Keep response under 150 words."""

    try:
        response = model.generate_content(prompt)
        return response.text, chat_history or []
    except Exception as e:
        return f"Error generating response: {str(e)}", chat_history or []

def format_chat_history(history):
    return "\n".join([f"Patient: {entry[0]}\nAssistant: {entry[1]}" for entry in history])