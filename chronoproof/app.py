import streamlit as st
import os
import subprocess
import pandas as pd
import sys  

st.set_page_config(page_title="ChronoProof: Video Temporal Reasoning", layout="wide")

st.title("🛡️ ChronoProof: Video Understanding & Temporal Reasoning")
st.markdown("Upload a classroom or surveillance video, extract behavioral events via YOLO, and query temporal relationships with strict mathematical proofs.")

# 1. Video Upload & Processing Sidebar
st.sidebar.header("1. Video Processing")
uploaded_file = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded video into the 'data' directory
    os.makedirs("data", exist_ok=True)
    video_path = os.path.join("data", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Saved: {uploaded_file.name}")

    if st.sidebar.button("Run YOLO Perception Engine"):
        with st.spinner("Running YOLOv8m models & ByteTrack on video... Please wait."):
            
            # Here we dynamically modify or call your perception script. 
            # We will run a python snippet that imports and executes your processor function directly on this file.
            cmd = [
                sys.executable, "-c", 
                f"import sys; sys.path.append('.'); from video_to_perception import process_classroom_video; process_classroom_video(r'{video_path}', r'data/generated_events_2.csv')"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.sidebar.success("Perception pipeline complete! Generated generated_events_2.csv.")
            else:
                st.sidebar.error("Error running perception script:")
                st.sidebar.code(result.stderr)

# 2. Main Panel: Query Interface
st.header("2. Ask Temporal Questions")
default_question = "Who was interacting with a cell phone?"
user_question = st.text_input("Enter your question about the video timeline:", default_question)

if st.button("Query Engine"):
    target_csv = "data/generated_events_2.csv"
    if not os.path.exists(target_csv):
        st.warning("Please run the YOLO Perception Engine first to generate the event database.")
    else:
        with st.spinner("Querying ChronoProof engine and generating proof..."):
            result = subprocess.run(
                [sys.executable, "ask.py", user_question],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                st.subheader("Answer & Timestamp Proof")
                st.code(result.stdout)
                
                if os.path.exists("out/proof.json"):
                    with open("out/proof.json", "r") as pf:
                        st.download_button("Download proof.json", pf, file_name="proof.json", mime="application/json")
            else:
                st.error(f"Error running query: {result.stderr}")

# 3. Display Live Dataset Preview Table
st.subheader("3. Current Event Dataset Table (`generated_events_2.csv`)")
csv_path = "data/generated_events_2.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    # Interactive filters or search for the table
    st.dataframe(df, use_container_width=True)
    st.caption(f"Total events logged: {len(df)}")
else:
    st.info("No event dataset found yet. Upload a video and click 'Run YOLO Perception Engine' in the sidebar.")