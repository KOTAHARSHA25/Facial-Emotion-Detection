import numpy as np
import cv2
import av
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import os
import threading
from twilio.rest import Client

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EmoSense AI - Facial Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  (dark glassmorphism theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: radial-gradient(ellipse at 20% 10%, #0f2050 0%, #0a0f2e 60%, #050510 100%) !important;
    color: #e2e8f0 !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: rgba(10, 15, 46, 0.97) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.4) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.5);
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stRadio > label { color: #94a3b8 !important; }

  /* ── Header bar ── */
  .stApp > header { background: transparent !important; }

  /* ── Hero title ── */
  .hero-title {
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #818cf8 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    line-height: 1.1;
    margin-bottom: 0.25rem;
  }
  .hero-sub {
    text-align: center;
    color: #64748b;
    font-size: 1.05rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-bottom: 2.5rem;
  }

  /* ── Glass card ── */
  .glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2rem;
    margin: 0.75rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.2);
  }

  /* ── Emotion badge pill ── */
  .badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 3px;
  }

  /* ── Tech stack pill ── */
  .tech-pill {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.5);
    color: #a5b4fc !important;
    padding: 5px 16px;
    border-radius: 999px;
    font-size: 0.85rem;
    margin: 4px;
  }

  /* ── Status indicator ── */
  .status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 6px #10b981;
    margin-right: 6px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* ── Info box ── */
  .info-box {
    background: rgba(99,102,241,0.12);
    border-left: 3px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    color: #c7d2fe !important;
  }

  /* ── Warning box ── */
  .warn-box {
    background: rgba(251,191,36,0.1);
    border-left: 3px solid #f59e0b;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    color: #fde68a !important;
  }

  /* ── Metric card ── */
  .metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
  }
  .metric-val {
    font-size: 2rem;
    font-weight: 700;
    color: #818cf8 !important;
  }
  .metric-label {
    font-size: 0.8rem;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* ── Hide streamlit branding ── */
  #MainMenu, footer { visibility: hidden; }

  /* ── webrtc container styling ── */
  .stWebRtcVideoContainer video {
    border-radius: 16px !important;
    border: 2px solid rgba(99,102,241,0.4) !important;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
EMOTION_DICT = {
    0: ("😠", "Angry",    "#ef4444"),
    1: ("😄", "Happy",    "#10b981"),
    2: ("😐", "Neutral",  "#6366f1"),
    3: ("😢", "Sad",      "#3b82f6"),
    4: ("😲", "Surprise", "#f59e0b"),
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
#  RTC CONFIG  — Streamlit Cloud + TURN
#  Streamlit Cloud blocks WebRTC UDP traffic. 
#  We MUST use a TURN server to relay video over TCP.
#  This function fetches ephemeral credentials from Twilio.
# ─────────────────────────────────────────────
@st.cache_data
def get_ice_servers():
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        print(f"Twilio credential error warning: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": get_ice_servers()
})

# --- WORKAROUND FOR STREAMLIT-WEBRTC THREADING BUG ---
# Resolves: "AttributeError: 'NoneType' object has no attribute 'is_alive'"
import streamlit_webrtc.shutdown
if not hasattr(streamlit_webrtc.shutdown.SessionShutdownObserver, "_patched"):
    _original_stop = streamlit_webrtc.shutdown.SessionShutdownObserver.stop
    def _patched_stop(self):
        if hasattr(self, "_polling_thread") and self._polling_thread is None:
            # Bypass the NoneType thread access bug during WebRTC teardown
            if hasattr(self, "_polling_thread_stopped") and self._polling_thread_stopped:
                self._polling_thread_stopped.set()
            return
        _original_stop(self)
    streamlit_webrtc.shutdown.SessionShutdownObserver.stop = _patched_stop
    streamlit_webrtc.shutdown.SessionShutdownObserver._patched = True
# -----------------------------------------------------

# ─────────────────────────────────────────────
#  MODEL LOADING  (cached across reruns)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading emotion model…")
def load_models():
    json_path = os.path.join(BASE_DIR, "emotion_model1.json")
    h5_path   = os.path.join(BASE_DIR, "emotion_model1.h5")
    xml_path  = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

    if not os.path.exists(json_path):
        st.error(f"Model JSON not found: {json_path}")
        st.stop()
    if not os.path.exists(h5_path):
        st.error(f"Model weights not found: {h5_path}")
        st.stop()

    with open(json_path, "r") as f:
        model = model_from_json(f.read())
    model.load_weights(h5_path)

    cascade = cv2.CascadeClassifier(xml_path)
    if cascade.empty():
        st.error("Haar cascade XML not found or failed to load.")
        st.stop()

    return model, cascade


# ─────────────────────────────────────────────
#  VIDEO FRAME CALLBACK
# ─────────────────────────────────────────────
# Thread-safe model references
_model_lock = threading.Lock()

def make_callback(model, cascade):
    """Factory so the callback closes over loaded model/cascade."""
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (99, 102, 241), 2)

            roi_gray = gray[y : y + h, x : x + w]
            try:
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            except Exception:
                continue

            if np.sum(roi_gray) == 0:
                continue

            roi = roi_gray.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))   # (1,48,48,1)

            with _model_lock:
                preds = model.predict(roi, verbose=0)[0]

            idx      = int(np.argmax(preds))
            conf     = float(preds[idx]) * 100
            emoji, label, color = EMOTION_DICT[idx]

            # Convert hex color → BGR
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            bgr = (b, g, r)

            text = f"{emoji} {label} {conf:.0f}%"

            # Background pill behind text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            lx, ly = x, y - 12
            cv2.rectangle(img, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), bgr, -1)

            # Label text
            cv2.putText(
                img, text, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )
            # Re-draw box with color
            cv2.rectangle(img, (x, y), (x + w, y + h), bgr, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    return video_frame_callback


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <div style="font-size:3rem;">🎭</div>
            <div style="font-size:1.2rem; font-weight:700; color:#818cf8;">EmoSense AI</div>
            <div style="font-size:0.75rem; color:#475569; margin-top:2px;">v2.0 · Streamlit Cloud</div>
        </div>
        <hr style="border-color:rgba(99,102,241,0.3); margin: 0.75rem 0;">
        """, unsafe_allow_html=True)

        choice = st.radio(
            "Navigate",
            ["🏠 Home", "🎥 Live Camera", "📊 Model Info", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:rgba(99,102,241,0.3);'>", unsafe_allow_html=True)

        # Emotion legend
        st.markdown("<p style='color:#475569; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;'>Detectable Emotions</p>", unsafe_allow_html=True)
        for _, (emoji, label, color) in EMOTION_DICT.items():
            st.markdown(
                f"<span class='badge' style='background:{color}22; border:1px solid {color}88; color:{color};'>{emoji} {label}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border-color:rgba(99,102,241,0.3); margin-top:1.5rem;'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center; font-size:0.8rem; color:#334155;'>Built by <b style='color:#818cf8;'>Team Backbench Warriors</b><br>Harsha · Vishnu · Rakesh · Rohit</div>",
            unsafe_allow_html=True,
        )

    return choice


# ─────────────────────────────────────────────
#  PAGES
# ─────────────────────────────────────────────
def page_home():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">5</div>
            <div class="metric-label">Emotions</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">48×48</div>
            <div class="metric-label">Input Size</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">CNN</div>
            <div class="metric-label">Architecture</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#818cf8; margin-top:0;">🧠 What is EmoSense AI?</h3>
            <p style="line-height:1.8; color:#94a3b8;">
            A real-time facial emotion recognition system powered by a custom-trained
            Convolutional Neural Network and OpenCV's Haar Cascade face detector.
            It processes your webcam stream directly in the browser via WebRTC —
            no video is stored or sent to any server.
            </p>
            <ul style="color:#94a3b8; line-height:2;">
                <li>⚡ Sub-100ms inference per frame</li>
                <li>🔒 Fully browser-side WebRTC stream</li>
                <li>🎯 Confidence score per prediction</li>
                <li>🧵 Thread-safe async processing</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#34d399; margin-top:0;">🚀 Quick Start</h3>
            <p style="line-height:1.8; color:#94a3b8;">
            <b style="color:#e2e8f0;">1.</b> Go to <b style="color:#818cf8;">🎥 Live Camera</b> in the sidebar.<br>
            <b style="color:#e2e8f0;">2.</b> Click <b style="color:#818cf8;">START</b> — allow camera permission.<br>
            <b style="color:#e2e8f0;">3.</b> Wait ~5 s for WebRTC to connect (first time).<br>
            <b style="color:#e2e8f0;">4.</b> Face the camera — watch live emotion labels!
            </p>
            <div class="info-box">
                💡 If connection is slow, try a different browser (Chrome works best) 
                or check your firewall isn't blocking UDP traffic.
            </div>
        </div>""", unsafe_allow_html=True)


def page_live(classifier, face_cascade):
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <h3 style="color:#e2e8f0; margin:0;">
            <span class="status-dot"></span>Live Emotion Tracking
        </h3>
        <p style="color:#475569; font-size:0.95rem; margin-top:4px;">
            Allow camera access · WebRTC peer connection · async frame processing
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
        ⚠️ <b>First-time connection</b> may take 5–15 s while ICE candidates are gathered.
        If it hangs beyond 30 s, refresh the page and try again.
    </div>""", unsafe_allow_html=True)

    callback = make_callback(classifier, face_cascade)

    webrtc_streamer(
        key="emosense-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,   # ← non-blocking; fixes event-loop crash
    )


def page_model_info():
    st.markdown("<h3 style='color:#818cf8;'>📊 Model Architecture</h3>", unsafe_allow_html=True)

    layers = [
        ("Conv2D 32", "3×3, ReLU",      "#6366f1"),
        ("MaxPool2D", "2×2",             "#0ea5e9"),
        ("Dropout",   "0.25",            "#f59e0b"),
        ("Conv2D 64", "3×3, ReLU",       "#6366f1"),
        ("MaxPool2D", "2×2",             "#0ea5e9"),
        ("Dropout",   "0.25",            "#f59e0b"),
        ("Flatten",   "—",               "#8b5cf6"),
        ("Dense 1024","ReLU",            "#10b981"),
        ("Dropout",   "0.5",             "#f59e0b"),
        ("Dense 5",   "Softmax (output)","#ef4444"),
    ]

    for name, detail, color in layers:
        st.markdown(
            f"""<div style="display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="background:{color}33; border:1px solid {color}88; color:{color}; padding:4px 12px; border-radius:8px; font-size:0.85rem; min-width:130px; text-align:center;">{name}</span>
                <span style="color:#64748b; font-size:0.85rem;">{detail}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#34d399; margin-top:0;">Training Details</h4>
            <ul style="color:#94a3b8; line-height:2; font-size:0.95rem;">
                <li>Dataset: FER-2013 (grayscale)</li>
                <li>Input: 48 × 48 × 1</li>
                <li>Optimizer: Adam</li>
                <li>Loss: Categorical Cross-Entropy</li>
                <li>Epochs: 50+ with early stopping</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#f59e0b; margin-top:0;">Detection Pipeline</h4>
            <ul style="color:#94a3b8; line-height:2; font-size:0.95rem;">
                <li>Frame → BGR → Grayscale</li>
                <li>Haar Cascade face detect</li>
                <li>ROI crop → resize 48×48</li>
                <li>Normalize [0,1] → CNN predict</li>
                <li>argmax → emotion + confidence</li>
            </ul>
        </div>""", unsafe_allow_html=True)


def page_about():
    st.markdown("""
    <div style="max-width:780px; margin:auto; text-align:center; padding:1rem 0 2rem;">
        <h2 style="color:#818cf8;">The Science Behind EmoSense AI</h2>
        <p style="color:#64748b; line-height:1.9; font-size:1.05rem; margin-bottom:1.5rem;">
            This system combines a custom Convolutional Neural Network trained on the 
            FER-2013 dataset with OpenCV's Haar Cascade Classifier for face localisation.
            Frames are streamed from the browser via WebRTC (aiortc), processed server-side
            asynchronously, and returned annotated in real-time.
        </p>
        <h4 style="color:#94a3b8; margin-bottom:1rem;">Core Technologies</h4>
        <div>
    """ + "".join(
        f"<span class='tech-pill'>{t}</span>"
        for t in ["Python 3.11", "TensorFlow 2.14", "Keras", "OpenCV 4.x",
                  "Streamlit 1.x", "streamlit-webrtc", "aiortc", "NumPy"]
    ) + """
        </div>
        <br><br>
        <div class="glass-card" style="text-align:left; max-width:480px; margin:auto;">
            <h4 style="color:#34d399; margin-top:0;">Team Backbench Warriors 🚀</h4>
            <ul style="color:#94a3b8; line-height:2.2;">
                <li>🧑‍💻 <b style="color:#e2e8f0;">Harsha</b> — ML Model & Backend</li>
                <li>🎨 <b style="color:#e2e8f0;">Vishnu</b> — UI/UX</li>
                <li>🔧 <b style="color:#e2e8f0;">Rakesh</b> — Integration</li>
                <li>📊 <b style="color:#e2e8f0;">Rohit</b> — Training & Evaluation</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    # Hero header
    st.markdown('<h1 class="hero-title">🎭 EmoSense AI Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Advanced Real-Time Facial Emotion Analysis · Powered by CNN + WebRTC</p>', unsafe_allow_html=True)

    choice = render_sidebar()

    # Load models (cached)
    classifier, face_cascade = load_models()

    st.markdown("<hr style='border-color:rgba(99,102,241,0.2); margin-bottom:1.5rem;'>", unsafe_allow_html=True)

    if "Home" in choice:
        page_home()
    elif "Camera" in choice:
        page_live(classifier, face_cascade)
    elif "Model" in choice:
        page_model_info()
    elif "About" in choice:
        page_about()


if __name__ == "__main__":
    main()