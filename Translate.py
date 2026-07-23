import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(
    page_title="Translate to English",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Translate to English")

st.write("Upload a Spanish audio (.wav) file to translate it into English.")

uploaded_file = st.file_uploader(
    "Browse Audio File",
    type=["wav", "mp3", "m4a", "ogg"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.info("Loading Whisper model...")

    model = whisper.load_model("small")

    with st.spinner("Translating... Please wait."):

        result = model.transcribe(
            temp_path,
            task="translate",
            language="es",
            fp16=False
        )

    st.success("Translation Complete")

    st.subheader("English Translation")

    st.write(result["text"])

    os.remove(temp_path)
