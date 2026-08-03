from faster_whisper import WhisperModel
import streamlit as st
import tempfile
import os

st.set_page_config(page_title="Translate to English")

st.title("🎤 Translate to English")

@st.cache_resource
def load_model():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a", "ogg"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded_file.read())
        audio_path = f.name

    model = load_model()

    with st.spinner("Translating... Please wait..."):
        segments, info = model.transcribe(
            audio_path,
            task="translate",
            language="es"
        )

    text = "".join(segment.text for segment in segments)

    st.success("Completed")
    st.write(text)

    os.remove(audio_path)
