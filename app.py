import numpy as np
import cv2
import av
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

st.set_page_config(page_title="EmoSense AI - Facial Emotion Detection", page_icon="🎭", layout="wide")

# Custom CSS for extraordinary look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    div[data-testid="stSidebar"] {
        background-color: rgba(15, 25, 40, 0.95);
        border-right: 2px solid #3b82f6;
    }
    .stApp > header {
        background-color: transparent !important;
    }
    .custom-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#4b6cb7, #182848);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 1rem 0;
        color: white;
    }
    h1, h2, h3, p, label {
        color: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}

# load json and create model
with open('emotion_model1.json', 'r') as json_file:
    loaded_model_json = json_file.read()

classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.error("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    #image gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        image=img_gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img=img, pt1=(x, y), pt2=(
            x + w, y + h), color=(255, 0, 0), thickness=2)
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi, verbose=0)[0]
            maxindex = int(np.argmax(prediction))
            finalout = emotion_dict[maxindex]
            output = str(finalout)
            label_position = (x, y - 10)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.markdown('<h1 class="custom-title">🎭 EmoSense AI Explorer</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem;'>Advanced Real-Time Facial Emotion Analysis System</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712010.png", width=120)
        st.title("Navigation")
        choice = st.radio("Select Mode", ["🏠 Home", "🎥 Live Camera Feed", "ℹ️ About the Project"])
        st.markdown("---")
        st.markdown('<div style="text-align:center;"><b>Built by Team Backbench Warriors</b><br><br>Harsha | Vishnu | Rakesh | Rohit</div>', unsafe_allow_html=True)

    if choice == "🏠 Home":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
                <div class="card">
                    <h3 style="color:#60a5fa;">About EmoSense AI</h3>
                    <p style="font-size: 1.1rem; line-height: 1.6;">
                    Welcome to our state-of-the-art emotion sensing platform. 
                    Using advanced Convolutional Neural Networks (CNN) and OpenCV, this system detects and analyzes human emotions in real-time with high accuracy.
                    </p>
                    <ul style="font-size: 1.1rem; line-height: 1.6;">
                        <li>🚀 Instant Real-time Processing</li>
                        <li>🧠 Custom Trained Deep Learning Model</li>
                        <li>🔒 Secure WebRTC Streaming Protocol</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class="card">
                    <h3 style="color:#60a5fa;">How to use it?</h3>
                    <p style="font-size: 1.1rem; line-height: 1.6;">
                    1. Navigate to <b>Live Camera Feed</b> from the sidebar.<br>
                    2. Grant camera permissions when prompted.<br>
                    3. Click <b>START</b> and watch the AI analyze your expressions instantly!<br>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
    elif choice == "🎥 Live Camera Feed":
        st.markdown("<h3 style='text-align: center;'>Live Emotion Tracking</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #cbd5e1;'>Enable your camera below. The AI will highlight your face and identify emotions dynamically.</p>", unsafe_allow_html=True)
        
        webrtc_streamer(
            key="emotion-detection", 
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

    elif choice == "ℹ️ About the Project":
        st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <h2 style="color:#60a5fa;">The Science Behind the Screen</h2>
                <br>
                <p style="font-size: 1.2rem; max-width: 800px; margin: auto; line-height: 1.8;">
                This intelligent application leverages a deep Convolutional Neural Network trained on thousands of facial expression images. 
                Coupled with a Haar Cascade Classifier from OpenCV for face detection, it accurately plots facial coordinates and performs real-time classification across 5 emotional states.
                </p>
                <br>
                <h3 style="color:#fcd34d;">Core Technologies</h3>
                <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 15px;">
                    <span style="background: #2563eb; padding: 5px 15px; border-radius: 20px;">Python</span>
                    <span style="background: #ea4335; padding: 5px 15px; border-radius: 20px;">TensorFlow / Keras</span>
                    <span style="background: #34a853; padding: 5px 15px; border-radius: 20px;">OpenCV</span>
                    <span style="background: #fbbc04; color: black; padding: 5px 15px; border-radius: 20px;">Streamlit</span>
                </div>
                <br><br>
                <p>Developed with passion by <b>Team Backbench Warriors</b>.</p>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
