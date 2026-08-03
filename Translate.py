import traceback
import streamlit as st
import tempfile
import os

# Check faster_whisper import
try:
    from faster_whisper import WhisperModel
    st.success("✅ faster_whisper imported successfully.")
except Exception as e:
    st.error("❌ Failed to import faster_whisper.")
    st.code(traceback.format_exc())
    raise

st.title("🎤 Translate to English")

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
        try:
            model = WhisperModel(
                "small",
                device="cpu",
                compute_type="int8"
            )
            return model
        except Exception:
            st.error("❌ Error while loading WhisperModel")
            st.code(traceback.format_exc())
            raise

    model = load_model()

    try:
        with st.spinner("Translating... Please wait..."):
            segments, info = model.transcribe(
                audio_path,
                task="translate",
                language="es"
            )

        text = "".join(segment.text for segment in segments)

        st.success("Completed")
        st.write(text)

    except Exception:
        st.error("❌ Error during transcription")
        st.code(traceback.format_exc())
        raise

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
