"""
R.A.D.A.R - Automatic Number Plate Recognition (ANPR) Enterprise Dashboard
================================================================================
Version: 4.6.0 Enterprise Ultra
AI Vision Model: openai/gpt-oss-120b (via Groq)
Architecture: Streamlit + YOLOv8 + EasyOCR + LangChain + Plotly
================================================================================
"""

import os
import sys
import time
import base64
import subprocess
from datetime import datetime
import json
import uuid

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import streamlit as st
import plotly.express as px

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# =====================================================================
# 1. APPLICATION INITIALIZATION & SYSTEM CONFIGURATION
# =====================================================================

load_dotenv()

st.set_page_config(
    page_title="R.A.D.A.R | Enterprise ANPR",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_enterprise_css():
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #050810 !important; color: #E2E8F0 !important; }
        .stApp { background: radial-gradient(circle at top left, #1a2035, #050810 60%, #000000 100%); }
        h1, h2, h3, h4, h5, h6 { color: #00FF66 !important; font-weight: 800 !important; letter-spacing: -0.5px; text-shadow: 0 0 10px rgba(0, 255, 102, 0.2); }
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: #0B101E; }
        ::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 5px; border: 2px solid #0B101E; }
        ::-webkit-scrollbar-thumb:hover { background: #00FF66; }
        .stButton > button { background: linear-gradient(135deg, #00FF66 0%, #00B259 100%); color: #000000 !important; font-weight: 800 !important; border: none !important; border-radius: 6px !important; padding: 0.75rem 2rem !important; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; text-transform: uppercase; letter-spacing: 1.5px; box-shadow: 0 4px 15px 0 rgba(0, 255, 102, 0.3) !important; width: 100%; }
        .stButton > button:hover { transform: translateY(-3px) !important; box-shadow: 0 8px 25px rgba(0, 255, 102, 0.5) !important; background: linear-gradient(135deg, #33FF88 0%, #00CC66 100%); }
        [data-testid="stMetric"] { background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(0, 255, 102, 0.1); border-radius: 12px; padding: 20px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5); transition: transform 0.3s ease, border-color 0.3s ease; }
        [data-testid="stMetric"]:hover { transform: translateY(-5px) scale(1.02); border-color: rgba(0, 255, 102, 0.5); box-shadow: 0 10px 40px rgba(0, 255, 102, 0.15); }
        [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 3rem !important; font-weight: 800 !important; font-family: 'JetBrains Mono', monospace; text-shadow: 0 0 15px rgba(255,255,255,0.2); }
        [data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 1rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 1.5px; }
        [data-testid="stSidebar"] { background-color: rgba(5, 8, 16, 0.95) !important; backdrop-filter: blur(20px); border-right: 1px solid rgba(255, 255, 255, 0.05); }
        .ai-table-wrapper { border: 1px solid #1E293B; border-radius: 12px; background: #0B101E; overflow: hidden; box-shadow: 0 15px 35px rgba(0,0,0,0.6); margin-top: 2rem; margin-bottom: 2rem; }
        .ai-row { display: flex; border-bottom: 1px solid #1E293B; }
        .ai-row:last-child { border-bottom: none; }
        .ai-header { flex: 0 0 250px; background: #0F172A; color: #F8FAFC; font-weight: 800; font-size: 1.1rem; padding: 1.5rem; display: flex; align-items: center; border-right: 1px solid #1E293B; text-transform: uppercase; letter-spacing: 1px; }
        .ai-content { flex: 1; padding: 1.5rem; color: #CBD5E1; display: flex; align-items: center; background: #050810; }
        .ai-content-text { font-size: 1.15rem; line-height: 1.7; }
        .ai-content-code { font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; color: #FF3366; background: rgba(255, 51, 102, 0.1); padding: 10px; border-radius: 6px; border: 1px solid rgba(255, 51, 102, 0.2); }
        .ai-frames-container { display: flex; gap: 20px; flex-wrap: nowrap; overflow-x: auto; padding-bottom: 10px; }
        .ai-frame-card { background: #000000; border: 1px solid #334155; border-radius: 8px; padding: 8px; display: flex; flex-direction: column; align-items: center; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
        .ai-frame-img { height: 140px; width: auto; object-fit: contain; border-radius: 4px; }
        .ai-frame-caption { color: #94A3B8; font-size: 0.85rem; margin-top: 10px; font-weight: 700; text-transform: uppercase; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

inject_enterprise_css()

# =====================================================================
# 3. ROBUST SESSION STATE MANAGEMENT
# =====================================================================

def initialize_session_state():
    state_variables = {
        'is_processing': False,
        'pipeline_completed': False,
        'current_video_name': None,
        'video_uploaded': False,
        'system_logs': [],
        'groq_api_key': os.getenv("GROQ_API_KEY", ""),
        'confidence_threshold': 0.35,
        'app_session_id': str(uuid.uuid4())[:8],
        'start_time': datetime.now()
    }
    for key, value in state_variables.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

def log_system_event(message, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    icon = "ℹ️" if level == "INFO" else "✅" if level == "SUCCESS" else "⚠️" if level == "WARNING" else "❌"
    st.session_state['system_logs'].insert(0, f"[{timestamp}] {icon} {message}")
    if len(st.session_state['system_logs']) > 50: st.session_state['system_logs'].pop()

def purge_system_data():
    files_to_remove = ['temp_upload.mp4', 'output_anpr_final.mp4', 'test_results.csv', 'test_results_interpolated.csv']
    deleted_count = 0
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception:
                pass
    st.session_state['pipeline_completed'] = False
    st.session_state['video_uploaded'] = False
    st.session_state['system_logs'] = []
    log_system_event("SYSTEM PURGE: Deleted temporary files.", "SUCCESS")
    return deleted_count

# =====================================================================
# 4. COMPUTER VISION & DATA PROCESSING UTILITIES
# =====================================================================

def encode_frame(frame, frame_id):
    """Helper to resize and encode an OpenCV frame to Base64 for the API."""
    resized_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    success, buffer = cv2.imencode('.jpg', resized_frame, encode_param)
    if success:
        return {"id": frame_id, "image_bgr": resized_frame, "base64": base64.b64encode(buffer).decode('utf-8')}
    return None

def extract_optimal_frames_for_llm(video_path, max_frames=3):
    if not os.path.exists(video_path): return []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return []
    
    intervals = [int(total_frames * (1.0 / (max_frames + 1) * i)) for i in range(1, max_frames + 1)]
    extracted_data = []
    for frame_id in intervals:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            encoded = encode_frame(frame, frame_id)
            if encoded: extracted_data.append(encoded)
    cap.release()
    return extracted_data

def extract_specific_frame_for_llm(video_path, frame_number):
    if not os.path.exists(video_path): return []
    cap = cv2.VideoCapture(video_path)
    # Ensure it's an integer before passing to OpenCV
    frame_id = int(float(frame_number))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    
    extracted_data = []
    if ret:
        encoded = encode_frame(frame, frame_id)
        if encoded: extracted_data.append(encoded)
            
    cap.release()
    return extracted_data

# =====================================================================
# 5. LANGCHAIN & GROQ VISION INTEGRATION
# =====================================================================

def query_vision_agent(user_query, frames_data):
    api_key = st.session_state.get('groq_api_key', '')
    if not api_key: return "Error: Unauthorized. Please provide a valid Groq API Key."
    if not frames_data: return "Error: No frames available for analysis."

    try:
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1, groq_api_key=api_key, max_tokens=1024)
        message_content = [{"type": "text", "text": f"You are a Traffic Analyst AI. Answer the user query based ONLY on what you see in the images.\n\nUser Query: {user_query}"}]
        
        for frame in frames_data:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame['base64']}"}
            })
            
        final_message = HumanMessage(content=message_content)
        log_system_event("Transmitting payload to Vision model...", "INFO")
        response = llm.invoke([final_message])
        log_system_event("Received AI response.", "SUCCESS")
        return response.content
    except Exception as e:
        log_system_event(f"API ERROR: {str(e)}", "ERROR")
        return f"Error from Groq API: {str(e)}"

def render_image2_ui(question, frames, answer):
    st.markdown('<div class="ai-table-wrapper">', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-row"><div class="ai-header">Question</div><div class="ai-content"><span class="ai-content-text" style="font-weight: 600;">{question}</span></div></div>', unsafe_allow_html=True)
    
    frames_html = ""
    for i in range(4):
        if i < len(frames):
            frames_html += f'<div class="ai-frame-card"><img class="ai-frame-img" src="data:image/jpeg;base64,{frames[i]["base64"]}"><div class="ai-frame-caption">Frame ID: {frames[i]["id"]}</div></div>'
        else:
            frames_html += f'<div class="ai-frame-card" style="width: 180px; height: 175px; justify-content: flex-end; background: #050810;"><div class="ai-frame-caption">Null</div></div>'
            
    st.markdown(f'<div class="ai-row"><div class="ai-header">Retrieved Video</div><div class="ai-content"><div class="ai-frames-container">{frames_html}</div></div></div>', unsafe_allow_html=True)
    answer_class = "ai-content-code" if "Error" in answer else "ai-content-text"
    st.markdown(f'<div class="ai-row"><div class="ai-header">Generated Answer</div><div class="ai-content"><span class="{answer_class}">{answer}</span></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# 6. PIPELINE EXECUTION ENGINE 
# =====================================================================

def execute_anpr_pipeline(video_path):
    st.session_state['is_processing'] = True
    try:
        with st.status("🚀 EXECUTING PIPELINE...", expanded=True) as status:
            st.write("Initializing YOLOv8 Engine...")
            result1 = subprocess.run([sys.executable, "src/main.py", video_path], capture_output=True, text=True)
            if result1.returncode != 0: raise Exception(f"main.py crashed. Error: {result1.stderr}")
            st.write("✅ Object Tracking & OCR Detection Complete.")
            
            st.write("🧠 Applying Pandas Voting & SciPy Interpolation...")
            result2 = subprocess.run([sys.executable, "src/visualization.py", video_path], capture_output=True, text=True)
            if result2.returncode != 0: raise Exception(f"visualization.py crashed. Error: {result2.stderr}")
            st.write("✅ Video Overlays Rendered.")
            status.update(label="✅ PIPELINE COMPLETE", state="complete", expanded=False)
            
        log_system_event("Pipeline Execution Successful.", "SUCCESS")
        st.session_state['pipeline_completed'] = True
        st.session_state['is_processing'] = False
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"❌ **CRITICAL PIPELINE FAILURE:** {str(e)}")
        log_system_event(f"CRITICAL ERROR: {str(e)}", "ERROR")
        st.session_state['is_processing'] = False

# =====================================================================
# 7. SIDEBAR NAVIGATION & SYSTEM SETTINGS
# =====================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("""<div style="text-align: center; padding-bottom: 10px;"><h2 style="color: #00FF66;">R.A.D.A.R</h2><p style="color: #94A3B8; font-size: 0.85rem;">Automatic Detection And Recognition</p></div>""", unsafe_allow_html=True)
        menu_options = ["🎥 Core ANPR Engine", "📊 Analytics Dashboard", "🗄️ Database Explorer", "🤖 AI Vision Agent", "🩺 System Diagnostics"]
        selected_menu = st.radio("Navigation", options=menu_options, label_visibility="collapsed")
        
        with st.expander("🔑 API Key Management", expanded=True):
            show_key = st.checkbox("Expose Key in UI", value=False)
            key_type = "default" if show_key else "password"
            api_key_input = st.text_input("Groq API Key", value=st.session_state['groq_api_key'], type=key_type, label_visibility="collapsed")
            if api_key_input != st.session_state['groq_api_key']: st.session_state['groq_api_key'] = api_key_input
                
        if st.button("🗑️ PURGE SYSTEM DATA"):
            purge_system_data()
            st.rerun()
    return selected_menu

# =====================================================================
# 8. MAIN VIEW RENDERING FUNCTIONS
# =====================================================================

def render_view_core_engine():
    st.markdown("<h1>🎥 Core ANPR Execution Engine</h1>", unsafe_allow_html=True)
    col_input, col_output = st.columns(2, gap="large")
    with col_input:
        uploaded_video = st.file_uploader("Select Video Source (MP4, AVI)", type=["mp4", "avi"])
        if uploaded_video is not None:
            with open("temp_upload.mp4", "wb") as f: f.write(uploaded_video.getbuffer())
            st.video("temp_upload.mp4")
            if st.button("⚡ EXECUTE PIPELINE"): execute_anpr_pipeline("temp_upload.mp4")
    with col_output:
        if st.session_state['pipeline_completed'] and os.path.exists("output_anpr_final.mp4"):
            st.video("output_anpr_final.mp4")
            with open("output_anpr_final.mp4", "rb") as file: st.download_button("💾 EXPORT", data=file, file_name="Analyzed.mp4", mime="video/mp4")

def render_view_analytics():
    st.markdown("<h1>📊 Analytics</h1>", unsafe_allow_html=True)
    if not os.path.exists("test_results_interpolated.csv"): return st.warning("No data.")
    df = pd.read_csv("test_results_interpolated.csv")
    valid_ocr_df = df[df['license_number_score'] > 0]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Unique Vehicles", df['car_id'].nunique())
    m2.metric("Total Frames", df['frame_nmr'].max() + 1)
    m3.metric("Avg Confidence", f"{(valid_ocr_df['license_number_score'].mean() * 100):.1f}%" if not valid_ocr_df.empty else "0%")

def render_view_database():
    st.markdown("<h1>🗄️ Database Explorer</h1>", unsafe_allow_html=True)
    if not os.path.exists("test_results_interpolated.csv"): return st.warning("No data.")
    df = pd.read_csv("test_results_interpolated.csv")
    st.dataframe(df, use_container_width=True, height=500)

def render_view_ai_agent():
    st.markdown("<h1>🤖 Visual AI Agent</h1>", unsafe_allow_html=True)
    
    target_video = "temp_upload.mp4" 
    if not os.path.exists(target_video): return st.warning("⚠️ No video data available.")

    user_query = st.text_input("Enter Inquiry:", placeholder="e.g., 'What color is the car with license plate EF10DZT?'")
    
    if st.button("EXECUTE AI QUERY") and user_query:
        with st.spinner("Analyzing Database for Query Context..."):
            extracted_frames = []
            plate_found = False
            
            # Context-Aware Logic: Clean string matching
            if os.path.exists("test_results_interpolated.csv"):
                df = pd.read_csv("test_results_interpolated.csv")
                
                # Filter out '0's and NaNs to only search actual reads
                valid_df = df[(df['license_number'].notna()) & (df['license_number'] != '0') & (df['license_number'] != 0)]
                unique_plates = valid_df['license_number'].unique()
                
                # Clean the user query (remove spaces, convert to uppercase)
                clean_query = user_query.upper().replace(" ", "").replace("-", "")
                
                for plate in unique_plates:
                    clean_plate = str(plate).upper().replace(" ", "").replace("-", "")
                    
                    # Ensure we don't accidentally match tiny substrings
                    if clean_plate in clean_query and len(clean_plate) >= 4:
                        plate_found = True
                        st.info(f"🔎 Database Match: Found plate **{plate}** in query. Extracting frame with highest OCR score...")
                        
                        # Find the absolute best frame for this specific car
                        car_data = valid_df[valid_df['license_number'] == plate]
                        best_row_idx = car_data['license_number_score'].idxmax()
                        best_frame_nmr = car_data.loc[best_row_idx]['frame_nmr']
                        
                        extracted_frames = extract_specific_frame_for_llm(target_video, best_frame_nmr)
                        break
                        
            # Fallback
            if not plate_found:
                st.info("No specific database plate matched. Extracting situational frames...")
                extracted_frames = extract_optimal_frames_for_llm(target_video, max_frames=3)

        if extracted_frames:
            with st.spinner("Transmitting payload to Vision model..."):
                answer = query_vision_agent(user_query, extracted_frames)
            render_image2_ui(user_query, extracted_frames, answer)

def render_view_diagnostics():
    st.markdown("<h1>🩺 System Diagnostics</h1>", unsafe_allow_html=True)
    for log in st.session_state['system_logs']: st.text(log)

def main():
    selected_view = render_sidebar()
    if "Core ANPR" in selected_view: render_view_core_engine()
    elif "Analytics" in selected_view: render_view_analytics()
    elif "Database" in selected_view: render_view_database()
    elif "AI Vision" in selected_view: render_view_ai_agent()
    elif "Diagnostics" in selected_view: render_view_diagnostics()

if __name__ == "__main__":
    main()