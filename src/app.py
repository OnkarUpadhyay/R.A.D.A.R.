"""
R.A.D.A.R - Automatic Number Plate Recognition (ANPR) Enterprise Dashboard
================================================================================
Version: 4.5.0 Enterprise Ultra
AI Vision Model: openai/gpt-oss-120b (via Groq)
Architecture: Streamlit + YOLOv8 + EasyOCR + LangChain + Plotly

DESCRIPTION:
This version introduces Context-Aware AI frame extraction. When a user asks about 
a specific license plate, the system queries the CSV database to find the absolute 
clearest frame of that vehicle, extracting it dynamically to feed the Vision model.
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

# Data & Math Libraries
import numpy as np
import pandas as pd

# Image & Video Processing
import cv2
from PIL import Image

# UI & Visualization
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# AI & LangChain Integrations
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

# =====================================================================
# 2. ADVANCED UI STYLING & GLASSMORPHISM CSS
# =====================================================================

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
        .stButton > button:active { transform: translateY(1px) !important; }
        .stDownloadButton > button { background: transparent !important; color: #00FF66 !important; border: 2px solid #00FF66 !important; }
        .stDownloadButton > button:hover { background: rgba(0, 255, 102, 0.1) !important; }
        [data-testid="stMetric"] { background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(0, 255, 102, 0.1); border-radius: 12px; padding: 20px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5); transition: transform 0.3s ease, border-color 0.3s ease; }
        [data-testid="stMetric"]:hover { transform: translateY(-5px) scale(1.02); border-color: rgba(0, 255, 102, 0.5); box-shadow: 0 10px 40px rgba(0, 255, 102, 0.15); }
        [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 3rem !important; font-weight: 800 !important; font-family: 'JetBrains Mono', monospace; text-shadow: 0 0 15px rgba(255,255,255,0.2); }
        [data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 1rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 1.5px; }
        [data-testid="stSidebar"] { background-color: rgba(5, 8, 16, 0.95) !important; backdrop-filter: blur(20px); border-right: 1px solid rgba(255, 255, 255, 0.05); }
        .streamlit-expanderHeader { background-color: #0F172A !important; border-radius: 8px !important; color: #00FF66 !important; font-weight: 600 !important; }
        [data-testid="stFileUploadDropzone"] { background-color: rgba(15, 23, 42, 0.5) !important; border: 2px dashed rgba(0, 255, 102, 0.3) !important; border-radius: 12px !important; transition: all 0.3s ease; }
        [data-testid="stFileUploadDropzone"]:hover { background-color: rgba(15, 23, 42, 0.8) !important; border-color: #00FF66 !important; }
        .stAlert { background-color: rgba(15, 23, 42, 0.8) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; color: #E2E8F0 !important; }
        .ai-table-wrapper { border: 1px solid #1E293B; border-radius: 12px; background: #0B101E; overflow: hidden; box-shadow: 0 15px 35px rgba(0,0,0,0.6); margin-top: 2rem; margin-bottom: 2rem; }
        .ai-row { display: flex; border-bottom: 1px solid #1E293B; }
        .ai-row:last-child { border-bottom: none; }
        .ai-header { flex: 0 0 250px; background: #0F172A; color: #F8FAFC; font-weight: 800; font-size: 1.1rem; padding: 1.5rem; display: flex; align-items: center; border-right: 1px solid #1E293B; text-transform: uppercase; letter-spacing: 1px; }
        .ai-content { flex: 1; padding: 1.5rem; color: #CBD5E1; display: flex; align-items: center; background: #050810; }
        .ai-content-text { font-size: 1.15rem; line-height: 1.7; }
        .ai-content-code { font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; color: #FF3366; background: rgba(255, 51, 102, 0.1); padding: 10px; border-radius: 6px; border: 1px solid rgba(255, 51, 102, 0.2); }
        .ai-frames-container { display: flex; gap: 20px; flex-wrap: nowrap; overflow-x: auto; padding-bottom: 10px; }
        .ai-frame-card { background: #000000; border: 1px solid #334155; border-radius: 8px; padding: 8px; display: flex; flex-direction: column; align-items: center; box-shadow: 0 4px 10px rgba(0,0,0,0.5); transition: transform 0.2s ease; }
        .ai-frame-card:hover { transform: scale(1.05); border-color: #00FF66; }
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
    icon = "ℹ️"
    if level == "SUCCESS": icon = "✅"
    elif level == "WARNING": icon = "⚠️"
    elif level == "ERROR": icon = "❌"
    formatted_msg = f"[{timestamp}] {icon} {message}"
    st.session_state['system_logs'].insert(0, formatted_msg)
    if len(st.session_state['system_logs']) > 50:
        st.session_state['system_logs'].pop()

def purge_system_data():
    files_to_remove = ['temp_upload.mp4', 'output_anpr_final.mp4', 'test_results.csv', 'test_results_interpolated.csv']
    deleted_count = 0
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                log_system_event(f"Could not delete {file_path}: {e}", "ERROR")
                
    st.session_state['pipeline_completed'] = False
    st.session_state['video_uploaded'] = False
    st.session_state['system_logs'] = []
    log_system_event(f"SYSTEM PURGE: Deleted {deleted_count} temporary files.", "SUCCESS")
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
        b64_string = base64.b64encode(buffer).decode('utf-8')
        return {"id": frame_id, "image_bgr": resized_frame, "base64": b64_string}
    return None

def extract_optimal_frames_for_llm(video_path, max_frames=3):
    """Fallback: Extracts evenly spaced frames if no specific plate is queried."""
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
    """Extracts ONE specific frame (the highest confidence OCR frame for a queried car)."""
    if not os.path.exists(video_path): return []
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    extracted_data = []
    if ret:
        encoded = encode_frame(frame, frame_number)
        if encoded: extracted_data.append(encoded)
            
    cap.release()
    return extracted_data

# =====================================================================
# 5. LANGCHAIN & GROQ VISION INTEGRATION (GPT-OSS-120B)
# =====================================================================

def query_vision_agent(user_query, frames_data):
    """
    Interfaces with the Groq API using the requested openai/gpt-oss-120b model.
    """
    api_key = st.session_state.get('groq_api_key', '')
    if not api_key:
        return "Error from Groq API: Error code: 401 - {'error': {'message': 'Unauthorized. Please provide a valid Groq API Key.'}}"
    if not frames_data:
        return "Error: No frames available for analysis. Please ensure the video was processed."

    try:
        # User requested specific model string
        llm = ChatGroq(
            model="openai/gpt-oss-120b", 
            temperature=0.1, 
            groq_api_key=api_key,
            max_tokens=1024
        )
        
        message_content = [
            {
                "type": "text", 
                "text": f"You are a Traffic Analyst AI. Analyze the provided frame(s) from a traffic camera. Answer the following user query precisely based ONLY on what you see in the images.\n\nUser Query: {user_query}"
            }
        ]
        
        for frame in frames_data:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame['base64']}"}
            })
            
        final_message = HumanMessage(content=message_content)
        
        log_system_event("Transmitting payload to openai/gpt-oss-120b via Groq...", "INFO")
        start_time = time.time()
        
        response = llm.invoke([final_message])
        
        end_time = time.time()
        log_system_event(f"Received AI response in {round(end_time - start_time, 2)} seconds.", "SUCCESS")
        
        return response.content
        
    except Exception as e:
        error_str = str(e).replace('"', "'")
        log_system_event(f"API ERROR: {error_str}", "ERROR")
        return f"Error from Groq API: Error code: 400 - {{'error': {{'message': '{error_str}', 'type': 'invalid_request_error'}}}}"

def render_image2_ui(question, frames, answer):
    st.markdown('<div class="ai-table-wrapper">', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-row"><div class="ai-header">Question</div><div class="ai-content"><span class="ai-content-text" style="font-weight: 600;">{question}</span></div></div>', unsafe_allow_html=True)
    
    frames_html = ""
    for i in range(4):
        if i < len(frames):
            b64_img = frames[i]['base64']
            frames_html += f'<div class="ai-frame-card"><img class="ai-frame-img" src="data:image/jpeg;base64,{b64_img}"><div class="ai-frame-caption">Frame ID: {frames[i]["id"]}</div></div>'
        else:
            frames_html += f'<div class="ai-frame-card" style="width: 180px; height: 175px; justify-content: flex-end; background: #050810;"><div class="ai-frame-caption">Null</div></div>'
            
    st.markdown(f'<div class="ai-row"><div class="ai-header">Retrieved Video</div><div class="ai-content"><div class="ai-frames-container">{frames_html}</div></div></div>', unsafe_allow_html=True)
    
    is_error = "Error from Groq API" in answer
    answer_class = "ai-content-code" if is_error else "ai-content-text"
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
            log_system_event("Starting main.py execution...", "INFO")
            result1 = subprocess.run([sys.executable, "src/main.py", video_path], capture_output=True, text=True)
            if result1.returncode != 0: raise Exception(f"main.py crashed. Error: {result1.stderr}")
            st.write("✅ Object Tracking & EasyOCR Detection Complete.")
            
            st.write("🧠 Applying Pandas Voting & SciPy Interpolation...")
            log_system_event("Starting visualization.py execution...", "INFO")
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
        st.markdown("""<div style="text-align: center; padding-bottom: 10px;"><svg width="70" height="70" viewBox="0 0 24 24" fill="none" stroke="#00FF66" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><circle cx="12" cy="11" r="3"></circle><path d="M12 14v4"></path></svg><h2 style="margin-top: 10px; font-size: 1.8rem; margin-bottom: 0px;">R.A.D.A.R</h2><p style="color: #94A3B8; font-size: 0.85rem; margin-top: 5px;">Real-time Automatic Detection And Recognition</p></div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<p style='color: #00FF66; font-weight: bold; margin-bottom: 5px; text-transform: uppercase; font-size: 0.9rem;'>📍 Navigation</p>", unsafe_allow_html=True)
        
        menu_options = ["🎥 Core ANPR Engine", "📊 Analytics Dashboard", "🗄️ Database Explorer", "🤖 AI Vision Agent", "🩺 System Diagnostics"]
        selected_menu = st.radio("Navigation Options", options=menu_options, label_visibility="collapsed")
        st.markdown("---")
        
        with st.expander("🔑 API Key Management", expanded=True):
            show_key = st.checkbox("Expose Key in UI", value=False)
            key_type = "default" if show_key else "password"
            api_key_input = st.text_input("Groq API Key", value=st.session_state['groq_api_key'], type=key_type, label_visibility="collapsed")
            if api_key_input != st.session_state['groq_api_key']:
                st.session_state['groq_api_key'] = api_key_input
                log_system_event("API Key updated by user.", "INFO")
                
        with st.expander("⚙️ Inference Parameters", expanded=False):
            st.session_state['confidence_threshold'] = st.slider("YOLO Confidence Threshold", 0.10, 0.90, 0.35, 0.05)
            
        st.markdown("---")
        st.markdown("<p style='color: #FF3366; font-weight: bold; margin-bottom: 5px; text-transform: uppercase; font-size: 0.9rem;'>⚠️ Session Management</p>", unsafe_allow_html=True)
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
        st.markdown("### 📥 1. Ingestion Protocol")
        uploaded_video = st.file_uploader("Select Video Source (MP4, AVI)", type=["mp4", "avi"])
        if uploaded_video is not None:
            temp_path = "temp_upload.mp4"
            with open(temp_path, "wb") as f: f.write(uploaded_video.getbuffer())
            st.session_state['video_uploaded'] = True
            st.video(temp_path)
            if st.button("⚡ EXECUTE PIPELINE"): execute_anpr_pipeline(temp_path)
        else:
            st.markdown("""<div style="border: 2px dashed #1E293B; border-radius: 12px; padding: 60px; text-align: center; background: #0B101E;"><h3 style="color: #475569 !important;">Awaiting Data Source</h3><p style="color: #475569;">Drag and drop a traffic video above.</p></div>""", unsafe_allow_html=True)
    with col_output:
        st.markdown("### 📤 2. Processed Telemetry")
        if st.session_state['pipeline_completed'] and os.path.exists("output_anpr_final.mp4"):
            st.video("output_anpr_final.mp4")
            with open("output_anpr_final.mp4", "rb") as file: st.download_button(label="💾 EXPORT PROCESSED FOOTAGE", data=file, file_name="RADAR_Analyzed_Footage.mp4", mime="video/mp4")
        else:
            st.markdown("""<div style="background: #0B101E; border: 1px solid #1E293B; border-radius: 12px; height: 420px; display: flex; align-items: center; justify-content: center; flex-direction: column;"><svg width="60" height="60" viewBox="0 0 24 24" stroke="#334155" stroke-width="1.5"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect></svg><p style="color: #475569; margin-top: 15px; font-weight: 600; letter-spacing: 2px;">OFFLINE</p></div>""", unsafe_allow_html=True)

def render_view_analytics():
    st.markdown("<h1>📊 Traffic Analytics & Intelligence</h1>", unsafe_allow_html=True)
    if not os.path.exists("test_results_interpolated.csv"):
        st.warning("⚠️ Telemetry Data Unavailable. Please process a video in the Core Engine first.")
        return
    df = pd.read_csv("test_results_interpolated.csv")
    valid_ocr_df = df[df['license_number_score'] > 0]
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unique Vehicles", df['car_id'].nunique())
    m2.metric("Total Frames", df['frame_nmr'].max() + 1)
    m3.metric("Avg Confidence", f"{(valid_ocr_df['license_number_score'].mean() * 100):.1f}%" if not valid_ocr_df.empty else "0%")
    m4.metric("Engine Status", "OPTIMAL")
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 🕒 Vehicle Dwell Time")
        duration_df = df.groupby('car_id')['frame_nmr'].count().reset_index()
        duration_df['Car ID'] = duration_df['car_id'].astype(str) 
        fig1 = px.bar(duration_df, x='Car ID', y='frame_nmr', color='frame_nmr', color_continuous_scale=['#008C46', '#00FF66'], template="plotly_dark")
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown("### 🎯 OCR Confidence Distribution")
        if not valid_ocr_df.empty:
            fig2 = px.histogram(valid_ocr_df, x='license_number_score', nbins=25, marginal="box", color_discrete_sequence=['#00FF66'], template="plotly_dark")
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

def render_view_database():
    st.markdown("<h1>🗄️ Master Telemetry Database</h1>", unsafe_allow_html=True)
    if not os.path.exists("test_results_interpolated.csv"):
        st.warning("⚠️ Database Empty. Execute pipeline first.")
        return
    df = pd.read_csv("test_results_interpolated.csv")
    with st.container():
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1: search_plate = st.text_input("🔍 Filter by Plate Number:")
        with filter_col2: car_id_filter = st.selectbox("Filter by Car ID:", options=["All"] + sorted(df['car_id'].unique().tolist()))
        with filter_col3: min_score = st.slider("Min OCR Score:", 0.0, 1.0, 0.0, 0.1)
        
    filtered_df = df.copy()
    if search_plate: filtered_df = filtered_df[filtered_df['license_number'].str.contains(search_plate.upper(), na=False)]
    if car_id_filter != "All": filtered_df = filtered_df[filtered_df['car_id'] == car_id_filter]
    if min_score > 0: filtered_df = filtered_df[filtered_df['license_number_score'] >= min_score]
        
    st.dataframe(filtered_df, use_container_width=True, height=500)
    st.download_button("💾 DOWNLOAD CSV RECORD", filtered_df.to_csv(index=False).encode('utf-8'), "RADAR_Filtered_Telemetry.csv", "text/csv")

def render_view_ai_agent():
    st.markdown("<h1>🤖 Visual AI Agent (GPT-OSS-120B)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 1.1rem; margin-bottom: 30px;'>Ask specific questions about license plates. The AI will cross-reference the database and extract the clearest frame for analysis.</p>", unsafe_allow_html=True)
    
    if not st.session_state.get('groq_api_key'):
        st.error("🔒 **Access Denied:** Groq API Key missing. Configure it in the Sidebar.")
        return
        
    target_video = "temp_upload.mp4" 
    if not os.path.exists(target_video):
        st.warning("⚠️ No video data available for AI analysis. Upload a video first.")
        return

    st.markdown("### 💬 System Query")
    user_query = st.text_input("Enter Inquiry:", placeholder="e.g., 'What color is the car with license plate EF10DZT?'", label_visibility="collapsed")
    
    if st.button("EXECUTE AI QUERY"):
        if user_query:
            with st.spinner("Analyzing Database for Query Context..."):
                extracted_frames = []
                plate_found = False
                
                # Context-Aware Logic: Check if the user mentioned a specific plate in the CSV
                if os.path.exists("test_results_interpolated.csv"):
                    df = pd.read_csv("test_results_interpolated.csv")
                    unique_plates = df[df['license_number'] != '0']['license_number'].unique()
                    
                    for plate in unique_plates:
                        if plate in user_query.upper():
                            plate_found = True
                            st.info(f"🔎 Database Match: Found plate **{plate}** in query. Extracting optimal frame...")
                            
                            # Find the absolute best frame for this specific car
                            car_data = df[df['license_number'] == plate]
                            best_frame = car_data.loc[car_data['license_number_score'].idxmax()]['frame_nmr']
                            
                            extracted_frames = extract_specific_frame_for_llm(target_video, int(best_frame))
                            break
                            
                # Fallback Logic: If no plate matched, just pull 3 standard frames
                if not plate_found:
                    st.info("No specific database plate matched. Extracting 3 standard situational frames...")
                    extracted_frames = extract_optimal_frames_for_llm(target_video, max_frames=3)

            if extracted_frames:
                with st.spinner("Transmitting payload to openai/gpt-oss-120b..."):
                    answer = query_vision_agent(user_query, extracted_frames)
                render_image2_ui(question=user_query, frames=extracted_frames, answer=answer)
            else:
                st.error("Failed to extract video frames for AI analysis.")
        else:
            st.warning("Please type a question before querying the agent.")

def render_view_diagnostics():
    st.markdown("<h1>🩺 System Diagnostics</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.markdown("### 💻 Environment State")
        uptime = str(datetime.now() - st.session_state['start_time']).split('.')[0]
        st.markdown(f"""<div style="background: #0B101E; padding: 20px; border-radius: 12px; border: 1px solid #1E293B;"><p style="color: #94A3B8;">Session ID</p><p style="color: #00FF66; font-size: 1.2rem;">{st.session_state['app_session_id']}</p><p style="color: #94A3B8;">Uptime</p><p style="color: #00FF66; font-size: 1.2rem;">{uptime}</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("### 📜 Application Logs")
        log_html = "<div style='background: #000000; border: 1px solid #334155; border-radius: 8px; padding: 15px; height: 350px; overflow-y: auto; font-family: monospace;'>"
        for log in st.session_state['system_logs']: log_html += f"<div style='color: #CBD5E1; margin-bottom: 5px; border-bottom: 1px solid #1E293B; padding-bottom: 4px;'>{log}</div>"
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

# =====================================================================
# 9. MAIN ROUTER
# =====================================================================

def main():
    selected_view = render_sidebar()
    if "Core ANPR" in selected_view: render_view_core_engine()
    elif "Analytics" in selected_view: render_view_analytics()
    elif "Database" in selected_view: render_view_database()
    elif "AI Assistant" in selected_view: render_view_ai_agent()
    elif "Diagnostics" in selected_view: render_view_diagnostics()

if __name__ == "__main__":
    main()