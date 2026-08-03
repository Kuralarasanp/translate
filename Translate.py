import streamlit as st
import whisper
import tempfile
import os
import traceback
import shutil

st.set_page_config(page_title="Translate to English")

st.title("🎤 Translate to English")

# -------------------------------
# Check FFmpeg
# -------------------------------
ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path is None:
    st.error("""
❌ FFmpeg is not installed or not available in PATH.

Please make sure your Streamlit Cloud project contains:

packages.txt

ffmpeg

Then reboot the app.
""")
    st.stop()

# -------------------------------
# Load Whisper model
# -------------------------------
@st.cache_resource
def load_model():
    return whisper.load_model("small")

try:
    model = load_model()
except Exception:
    st.error("❌ Failed to load Whisper model.")
    st.code(traceback.format_exc())
    st.stop()

# -------------------------------
# Upload file
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a", "ogg"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    try:

        with st.spinner("Translating..."):

            result = model.transcribe(
                audio_path,
                task="translate",
                language="es",
                fp16=False
            )

        st.success("✅ Translation Completed")
        st.write(result["text"])

    except FileNotFoundError as e:
        st.error("❌ FFmpeg executable not found.")
        st.code(str(e))

    except Exception:
        st.error("❌ Translation Failed")
        st.code(traceback.format_exc())

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
