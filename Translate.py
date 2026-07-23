from faster_whisper import WhisperModel
import streamlit as st
import tempfile
import os

st.title("Translate to English")

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a", "ogg"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded_file.read())
        audio_path = f.name

    @st.cache_resource
    def load_model():
        return WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )

    model = load_model()

    with st.spinner("Translating..."):
        segments, info = model.transcribe(
            audio_path,
            task="translate",
            language="es"
        )

    text = "".join(segment.text for segment in segments)

    st.success("Completed")
    st.write(text)

    os.remove(audio_path)
