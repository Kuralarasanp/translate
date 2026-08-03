import streamlit as st
import whisper
import tempfile
import os
import shutil
import subprocess
import traceback

st.set_page_config(page_title="Translate to English")

st.title("🎤 Translate to English")

# ===========================
# Debug Information
# ===========================
st.subheader("🔍 Environment Information")

st.write("Python Version:")
import sys
st.code(sys.version)

st.write("FFmpeg Path:")
st.code(str(shutil.which("ffmpeg")))

try:
    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        text=True
    )
    st.write("FFmpeg Version:")
    st.code(result.stdout[:500])
except Exception:
    st.error("Unable to execute FFmpeg")
    st.code(traceback.format_exc())

# ===========================
# Load Whisper Model
# ===========================
@st.cache_resource
def load_model():
    return whisper.load_model("small")

# ===========================
# Upload Audio
# ===========================
uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a", "ogg"]
)

if uploaded_file is not None:

    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

        st.write("Temporary Audio File:")
        st.code(audio_path)

        st.write("File Exists:")
        st.code(str(os.path.exists(audio_path)))

        st.write("File Size:")
        st.code(str(os.path.getsize(audio_path)))

        model = load_model()

        with st.spinner("Translating... Please wait..."):

            result = model.transcribe(
                audio_path,
                task="translate",
                language="es",
                fp16=False
            )

        st.success("Completed!")

        st.subheader("English Translation")
        st.write(result["text"])

    except Exception:
        st.error("❌ Full Error")
        st.code(traceback.format_exc())

    finally:
        try:
            if "audio_path" in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
