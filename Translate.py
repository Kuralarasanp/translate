import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="Translate to English")

st.title("🎤 Translate to English")

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a", "ogg"]
)

@st.cache_resource
def load_model():
    return whisper.load_model("small")

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

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

    os.remove(audio_path)
