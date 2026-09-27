import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os

# --- Konfiguracja strony ---
st.set_page_config(
    page_title="Polski Transkrybator SRT",
    page_icon="🇵🇱",
    layout="centered"
)

st.title("🇵🇱 Transkrybator audio/wideo → SRT")
st.caption("Wgraj plik, a aplikacja wygeneruje napisy w formacie .srt w języku polskim.")

# --- Ładowanie modelu (raz na sesję serwera) ---
@st.cache_resource(show_spinner="Ładowanie modelu... (tylko za pierwszym razem)")
def load_model():
    # Używamy modelu "small" dla równowagi szybkości i jakości na CPU.
    # Streamlit Cloud ma ograniczoną pamięć, więc "small" jest bezpiecznym wyborem.
    return WhisperModel("small", device="cpu", compute_type="int8")

model = load_model()

def format_time(seconds: float) -> str:
    """Sekundy → format SRT: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# --- Upload ---
uploaded_file = st.file_uploader(
    "Wybierz plik audio lub wideo",
    type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "mov", "avi", "mkv", "webm"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)
    st.write(f"**Plik:** `{uploaded_file.name}`  |  **Rozmiar:** {uploaded_file.size / 1024 / 1024:.2f} MB")

    if st.button("🚀 Rozpocznij transkrypcję", type="primary"):
        suffix = os.path.splitext(uploaded_file.name)[1]
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            with st.spinner("Trwa transkrypcja... To może potrwać kilka minut."):
                segments, info = model.transcribe(
                    tmp_path,
                    language="pl",
                    task="transcribe",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    initial_prompt="To jest transkrypcja nagrania w języku polskim."
                )

                srt_lines = []
                for i, seg in enumerate(segments, start=1):
                    srt_lines.append(str(i))
                    srt_lines.append(f"{format_time(seg.start)} --> {format_time(seg.end)}")
                    srt_lines.append(seg.text.strip())
                    srt_lines.append("")

                srt_content = "\n".join(srt_lines)

            st.success(f"✅ Gotowe! Wykryty język: **{info.language}** (pewność: {info.language_probability:.2f})")

            base_name = os.path.splitext(uploaded_file.name)[0]
            st.download_button(
                label="📥 Pobierz plik .srt",
                data=srt_content.encode("utf-8"),
                file_name=f"{base_name}.srt",
                mime="application/x-subrip",
                type="primary"
            )

            with st.expander("👀 Podgląd napisów"):
                st.text_area("Zawartość SRT", srt_content, height=300)

        except Exception as e:
            st.error(f"Błąd: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

st.divider()
st.caption("Aplikacja działa bez bazy danych – pliki są przetwarzane tymczasowo i usuwane po zakończeniu.")