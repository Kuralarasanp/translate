import os
import tempfile

import streamlit as st
import whisper

st.set_page_config(page_title="Audio Transcriber", page_icon="🎙️", layout="centered")

st.title("🎙️ Audio Transcriber")
st.caption("Upload an audio file, then transcribe or translate it with Whisper.")


@st.cache_resource(show_spinner=False)
def load_model(model_size: str):
    return whisper.load_model(model_size)


# --- Sidebar options ---
with st.sidebar:
    st.header("Options")
    model_size = st.selectbox(
        "Model size",
        ["tiny", "base", "small", "medium", "large"],
        index=1,  # "base" - safer default for Streamlit Cloud's free-tier RAM limit
        help="Bigger = more accurate but slower and more RAM/GPU hungry. "
        "On Streamlit Community Cloud's free tier (~1GB RAM), 'small' or larger may crash.",
    )
    task = st.radio(
        "Task",
        ["translate", "transcribe"],
        format_func=lambda t: "Translate to English" if t == "translate" else "Transcribe (original language)",
        index=0,
    )
    language = st.selectbox(
        "Source language",
        ["es", "en", "fr", "pt", "auto-detect"],
        index=0,
        help="Auto-detect lets Whisper guess the spoken language.",
    )

# --- File upload ---
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
)

transcribe_clicked = st.button("Transcribe", type="primary", disabled=uploaded_file is None)

if transcribe_clicked and uploaded_file is not None:
    # Save the upload to a temp file so Whisper (ffmpeg) can read it from disk
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner(f"Loading '{model_size}' model..."):
            model = load_model(model_size)

        with st.spinner("Transcribing... this can take a bit depending on file length."):
            lang_arg = None if language == "auto-detect" else language
            result = model.transcribe(tmp_path, task=task, language=lang_arg)

        st.success("Done.")
        st.subheader("Result")
        st.text_area("Transcribed text", result["text"], height=250)

        st.download_button(
            "Download as .txt",
            data=result["text"],
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}.txt",
            mime="text/plain",
        )
    except Exception as e:
        st.error(f"Something went wrong: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
elif uploaded_file is None:
    st.info("Upload an audio file to get started.")