import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tempfile
import shap
import matplotlib.pyplot as plt
import json
import os

# Set page config as the first Streamlit command
st.set_page_config(page_title="Classroom Emotion Insights", page_icon="📚", layout="wide")

# Suppress TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Model path
MODEL_PATH = "D:/QRIOCITY/C556/test/model_weights.h5.keras"  # Updated to full path based on your error

# Load the model with error handling
model = None
model_load_error = None
try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    model_load_error = str(e)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    model = Sequential([Dense(7, input_shape=(56*56,), activation='softmax')])  # Dummy model

# Original emotion labels from the model
ORIGINAL_EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Role-specific emotion mappings
STUDENT_EMOTION_MAP = {
    "Angry": "Frustrated", "Fear": "Frustrated",
    "Disgust": "Boredom", "Sad": "Boredom",
    "Happy": "Confident", "Surprise": "Confident",
    "Neutral": "Neutral"
}

TEACHER_EMOTION_MAP = {
    "Angry": "Angry", "Disgust": "Sad", "Fear": "Sad",
    "Happy": "Happy", "Neutral": "Neutral", "Sad": "Sad",
    "Surprise": "Surprise"
}

# Role-specific responses
STUDENT_EMOTION_RESPONSES = {
    "Frustrated": "Feeling frustrated? Take a moment to breathe—it’ll get better! 🧘‍♂️",
    "Boredom": "Boredom creeping in? Let’s find something exciting to focus on! 🤗",
    "Confident": "You’re shining with confidence—keep it up! 😃",
    "Neutral": "Cool and calm—perfectly balanced! ⚖️"
}

STUDENT_POPUPS = {
    "Frustrated": "Try asking a question or taking a quick break to reset!",
    "Boredom": "Make the class interactive—suggest a group activity!",
    "Confident": "Great job! Share your ideas with the class!",
    "Neutral": "You’re doing fine—keep engaged!"
}

TEACHER_EMOTION_RESPONSES = {
    "Angry": "It’s okay to feel angry sometimes. Take a deep breath! 🧘‍♂️",
    "Sad": "Feeling down? You’re not alone—we’re here! ❤️",
    "Happy": "You’re glowing with happiness! Keep it up! 😃",
    "Neutral": "Cool and calm—perfectly balanced! ⚖️",
    "Surprise": "Wow, what a surprise! Hope it’s a good one! 🎉"
}

# Classroom configuration
CLASSROOM_CONFIG = {
    "roles": ["Student", "Teacher"],
    "welcome_msg": "Welcome to Classroom Emotion Insights!",
    "login_file": "classroom_users.json",
    "icon": "https://cdn-icons-png.flaticon.com/512/167/167756.png"
}

# Initialize user data
def initialize_user_data():
    sample_data = {"Student": {"student1": "pass123"}, "Teacher": {"teacher1": "teach456"}}
    filename = CLASSROOM_CONFIG["login_file"]
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump(sample_data, f)

# Login and Signup functions
def check_login(role, username, password):
    filename = CLASSROOM_CONFIG["login_file"]
    try:
        with open(filename, "r") as f:
            users = json.load(f)
        return username in users[role] and users[role][username] == password
    except:
        return False

def signup_user(role, username, password):
    filename = CLASSROOM_CONFIG["login_file"]
    try:
        with open(filename, "r") as f:
            users = json.load(f)
    except:
        users = {r: {} for r in CLASSROOM_CONFIG["roles"]}
    if username in users[role]:
        return False
    users[role][username] = password
    with open(filename, "w") as f:
        json.dump(users, f)
    return True

# Process frame and detect emotion
def process_frame(frame, role):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (56, 56))
    img_array = img_to_array(resized_frame) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    original_emotion = ORIGINAL_EMOTION_LABELS[predicted_class]
    emotion_map = STUDENT_EMOTION_MAP if role == "Student" else TEACHER_EMOTION_MAP
    predicted_emotion = emotion_map[original_emotion]

    cv2.putText(frame, f'Emotion: {predicted_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame, predicted_emotion

# Enhanced CSS with animations
st.markdown("""
    <style>
    .main {background: linear-gradient(to right, #e0f7fa, #b2ebf2); padding: 20px; border-radius: 15px;}
    .stButton>button {background: #0288d1; color: white; border-radius: 10px; padding: 10px 20px; font-weight: bold; transition: all 0.3s;}
    .stButton>button:hover {background: #0277bd; transform: scale(1.05);}
    .stTextInput>div>input {border-radius: 10px; padding: 10px; border: 2px solid #0288d1;}
    .stSelectbox {background: white; border-radius: 10px; padding: 5px;}
    .sidebar .sidebar-content {background: #ffffff; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    .emotion-popup {background: #ffeb3b; padding: 15px; border-radius: 10px; border: 2px solid #f9a825; font-size: 16px; animation: popup 0.5s ease-in-out; margin-top: 10px;}
    @keyframes popup {0% {transform: scale(0.8); opacity: 0;} 100% {transform: scale(1); opacity: 1;}}
    .welcome-text {font-size: 36px; color: #0288d1; text-align: center; animation: fadeIn 1s;}
    @keyframes fadeIn {0% {opacity: 0;} 100% {opacity: 1;}}
    </style>
""", unsafe_allow_html=True)

# Display model loading status
if model_load_error is None:
    st.write("✅ Model loaded successfully!")
else:
    st.error(f"❌ Failed to load model from {MODEL_PATH}. Error: {model_load_error}")
    st.write("Please ensure the model file exists and is compatible with your TensorFlow version.")
    st.write(f"Current TensorFlow version: {tf.__version__}")
    st.warning("Using a dummy model for testing. Functionality will be limited.")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/201/201614.png", width=100)
    st.title("Emotion Insights")
    initialize_user_data()
    st.image(CLASSROOM_CONFIG["icon"], width=50)
    role = st.selectbox("🎭 Select Your Role", CLASSROOM_CONFIG["roles"], help="Choose if you're a Student or Teacher!")
    st.markdown("---")
    st.info("💡 Analyze emotions in real-time to enhance classroom vibes!")

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if "recording" not in st.session_state:
    st.session_state.recording = False

# Main Content
st.markdown(f"<h1 class='welcome-text'>{CLASSROOM_CONFIG['welcome_msg']}</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #0288d1;'>Unlock the power of emotions with AI—upload or go live!</p>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["🔑 Login", "✨ Sign Up"])
    
    with tab1:
        st.subheader("Login to Your Classroom")
        login_username = st.text_input("Username", key="login_user", help="Enter your username")
        login_password = st.text_input("Password", type="password", key="login_pass", help="Enter your password")
        if st.button("Login", key="login_btn"):
            if check_login(role, login_username, login_password):
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.success(f"Welcome back, {login_username} ({role})! Let’s dive in!")
            else:
                st.error("Oops! Invalid credentials. Try again or sign up!")

    with tab2:
        st.subheader("Join the Classroom")
        signup_username = st.text_input("New Username", key="signup_user", help="Choose a unique username")
        signup_password = st.text_input("New Password", type="password", key="signup_pass", help="Create a strong password")
        if st.button("Sign Up", key="signup_btn"):
            if signup_user(role, signup_username, signup_password):
                st.success(f"Account created for {signup_username} ({role})! Log in to start.")
            else:
                st.error("Username taken! Try something else.")
else:
    st.markdown(f"<h3 style='color: #0288d1;'>Hello, {st.session_state.username} ({role})! 🚀</h3>", unsafe_allow_html=True)
    input_method = st.radio("📡 Choose Your Mode", ["Upload File", "Live Camera"], horizontal=True)
    EMOTION_RESPONSES = STUDENT_EMOTION_RESPONSES if role == "Student" else TEACHER_EMOTION_RESPONSES

    # Placeholder for student popups on the page
    if role == "Student":
        popup_placeholder = st.empty()

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("📸 Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"], key="uploader", help="Supports images and videos!")
        if uploaded_file:
            file_type = uploaded_file.type.split('/')[0]
            if file_type == "image":
                img_size = (56, 56)
                img = load_img(uploaded_file, target_size=img_size, color_mode="grayscale")
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(img, caption="Your Upload", use_column_width=True)
                with col2:
                    with st.spinner("🔍 Detecting Emotion..."):
                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions)
                        original_emotion = ORIGINAL_EMOTION_LABELS[predicted_class]
                        emotion_map = STUDENT_EMOTION_MAP if role == "Student" else TEACHER_EMOTION_MAP
                        predicted_emotion = emotion_map[original_emotion]
                    st.success(f"🎉 Emotion: **{predicted_emotion}**")
                    st.write(EMOTION_RESPONSES[predicted_emotion])
                    if role == "Student":
                        popup_placeholder.markdown(f"<div class='emotion-popup'>💡 {STUDENT_POPUPS[predicted_emotion]}</div>", unsafe_allow_html=True)

                with st.expander("🧠 AI Insights"):
                    masker = shap.maskers.Image("inpaint_telea", img_array[0].shape)
                    explainer = shap.Explainer(model, masker=masker)
                    shap_values = explainer(img_array)
                    plt.figure()
                    shap.image_plot(shap_values)
                    plt.savefig("shap_plot.png")
                    st.image("shap_plot.png", caption="What the AI Sees", use_column_width=True)

            elif file_type == "video":
                st.write("🎥 Streaming Your Video...")
                stframe = st.empty()
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)

                frame_count = 0
                skip_frames = 5
                last_emotion = "Neutral"
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % skip_frames != 0:
                        continue
                    frame, predicted_emotion = process_frame(frame, role)
                    last_emotion = predicted_emotion
                    stframe.image(frame, channels="BGR", use_column_width=True)
                    st.success(f"🎬 Current Emotion: **{last_emotion}**")
                    st.write(EMOTION_RESPONSES[last_emotion])
                    if role == "Student":
                        popup_placeholder.markdown(f"<div class='emotion-popup'>💡 {STUDENT_POPUPS[last_emotion]}</div>", unsafe_allow_html=True)
                cap.release()
                os.unlink(tfile.name)
                st.success(f"🎬 Final Emotion: **{last_emotion}**")
                st.write(EMOTION_RESPONSES[last_emotion])

    elif input_method == "Live Camera":
        st.write("📷 Live Emotion Detection")
        stframe = st.empty()
        start_stop = st.button("🎥 Start" if not st.session_state.recording else "⏹️ Stop", key="start_stop")

        if start_stop:
            st.session_state.recording = not st.session_state.recording

        if st.session_state.recording:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Webcam not accessible! Check your connection.")
                st.session_state.recording = False
            else:
                last_emotion = "Neutral"
                progress_bar = st.progress(0)
                stop_button = st.button("⏹️ Stop Recording", key="stop_recording")
                while st.session_state.recording:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame, predicted_emotion = process_frame(frame, role)
                    last_emotion = predicted_emotion
                    stframe.image(frame, channels="BGR", use_column_width=True)
                    progress_bar.progress(min(100, int(np.random.uniform(0, 100))))
                    st.success(f"🎭 Current Emotion: **{last_emotion}**")
                    st.write(EMOTION_RESPONSES[last_emotion])
                    if role == "Student":
                        popup_placeholder.markdown(f"<div class='emotion-popup'>💡 {STUDENT_POPUPS[last_emotion]}</div>", unsafe_allow_html=True)
                    if stop_button or not st.session_state.recording:
                        st.session_state.recording = False
                        break
                cap.release()
                st.success(f"🛑 Stopped. Last Emotion: **{last_emotion}**")
                st.write(EMOTION_RESPONSES[last_emotion])
        else:
            if "last_emotion" in st.session_state:
                st.success(f"🎭 Last Emotion: **{st.session_state.last_emotion}**")
                st.write(EMOTION_RESPONSES[st.session_state.last_emotion])
                if role == "Student":
                    popup_placeholder.markdown(f"<div class='emotion-popup'>💡 {STUDENT_POPUPS[st.session_state.last_emotion]}</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚪 Logout", key="logout_btn"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

st.markdown("<p style='text-align: center; color: #0288d1;'>💡 <i>Empowering Classrooms with Emotional Intelligence</i> 📚</p>", unsafe_allow_html=True)