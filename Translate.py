import streamlit as st
import tempfile
import os
import sys
import traceback
import shutil

st.set_page_config(page_title="Translate to English", page_icon="🎤")

st.title("🎤 Translate to English")
st.caption("Upload a call recording (Spanish or auto-detected) and get an English translation.")
st.caption(f"Running on Python {sys.version.split()[0]}")

# -------------------------------
# Import whisper (wrapped so a broken/incompatible dependency
# shows a real error instead of crashing the whole app blank)
# -------------------------------
try:
    import whisper
except Exception:
    st.error("❌ Failed to import the `whisper` package. This usually means one of "
              "its dependencies (torch, numba, triton) isn't compatible with the "
              "Python version this app is running on, shown above.")
    st.code(traceback.format_exc())
    st.stop()

# -------------------------------
# Check FFmpeg
# -------------------------------
ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path is None:
    st.error("""
❌ FFmpeg is not installed or not available in PATH.

Make sure your repo has a `packages.txt` file (next to requirements.txt) containing:

```
ffmpeg
```

Then reboot the app from the Streamlit Cloud dashboard (Manage app → Reboot).
""")
    st.stop()

# -------------------------------
# Sidebar options
# -------------------------------
with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox(
        "Model size",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="Larger = more accurate but much slower on CPU. "
             "On Streamlit Cloud's free tier, 'base' is usually the best speed/quality tradeoff. "
             "'small'+ can be very slow or time out on long recordings."
    )
    lang_mode = st.radio(
        "Source language",
        ["Auto-detect", "Force Spanish (es)"],
        index=0
    )
    max_mb = st.number_input("Max upload size (MB) guard", min_value=1, max_value=500, value=100)

# -------------------------------
# Load Whisper model (cached per model_size)
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_model(size: str):
    return whisper.load_model(size)

try:
    with st.spinner(f"Loading '{model_size}' model (first run downloads it, can take a bit)..."):
        model = load_model(model_size)
except Exception:
    st.error("❌ Failed to load Whisper model.")
    st.code(traceback.format_exc())
    st.stop()

# -------------------------------
# Upload file
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a", "ogg", "flac", "aac"]
)

if uploaded_file:
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > max_mb:
        st.error(f"❌ File is {size_mb:.1f} MB, which exceeds the {max_mb} MB guard set in the sidebar.")
        st.stop()

    # Preserve the real extension so ffmpeg/whisper handle it correctly
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    try:
        transcribe_kwargs = {"task": "translate", "fp16": False, "verbose": False}
        if lang_mode == "Force Spanish (es)":
            transcribe_kwargs["language"] = "es"

        # --- Hook Whisper's internal frame-progress tqdm into a real
        # Streamlit progress bar (verbose=False is what activates it) ---
        import whisper.transcribe as _whisper_transcribe_module

        progress_bar = st.progress(0, text="Translating... 0%")

        class _StProgressBar:
            def __init__(self, total=0, unit=None, disable=False, **kwargs):
                self.total = total or 1
                self.n = 0

            def update(self, amount):
                self.n = min(self.total, self.n + amount)
                pct = self.n / self.total
                progress_bar.progress(pct, text=f"Translating... {pct * 100:.0f}%")

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        _original_tqdm = _whisper_transcribe_module.tqdm.tqdm
        _whisper_transcribe_module.tqdm.tqdm = _StProgressBar
        try:
            result = model.transcribe(audio_path, **transcribe_kwargs)
        finally:
            _whisper_transcribe_module.tqdm.tqdm = _original_tqdm
            progress_bar.empty()

        st.success("✅ Translation Completed")

        detected_lang = result.get("language")
        if detected_lang:
            st.caption(f"Detected source language: `{detected_lang}`")

        st.write(result["text"])

        st.download_button(
            "Download as .txt",
            data=result["text"],
            file_name=os.path.splitext(uploaded_file.name)[0] + "_translation.txt",
            mime="text/plain",
        )

    except FileNotFoundError as e:
        st.error("❌ FFmpeg executable not found.")
        st.code(str(e))

    except Exception:
        st.error("❌ Translation Failed")
        st.code(traceback.format_exc())

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
