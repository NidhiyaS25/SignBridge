"""SignBridge - Breaking Communication Barriers with Indian Sign Language.

Tagline: Sign Beyond Words.
Streamlit Web Application providing real-time ISL recognition,
message construction, and text-to-speech output.
"""

import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from modules.camera import Camera
from modules.hand_tracking import HandTracker
from modules.preprocessing import extract_feature_vector, get_feature_column_names
from modules.recognition import SignClassifier
from modules.sentence_builder import SentenceBuilder
from modules.speech import SpeechEngine
from modules.training import train_sign_model
from utils.config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_CAMERA_INDEX,
    MODEL_PATH,
    SMOOTHING_WINDOW_SIZE,
    SUPPORTED_SIGNS,
    SIGN_LOOKUP,
    DATASET_CSV,
)

# Page Configuration
st.set_page_config(
    page_title="SignBridge - Sign Beyond Words",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: Deep navy & blue accents with a clean, light, accessible theme
st.markdown(
    """
    <style>
    /* Main Background & Font Styling */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: #ffffff;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.15);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .hero-tagline {
        font-size: 1.15rem;
        color: #93c5fd;
        margin: 6px 0 12px 0;
        font-weight: 500;
    }
    .hero-description {
        font-size: 0.95rem;
        color: #e2e8f0;
        max-width: 900px;
        line-height: 1.5;
        margin: 0;
    }

    /* Content Cards */
    .ui-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .ui-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Message Display Box */
    .message-display {
        background-color: #f1f5f9;
        border: 2px dashed #94a3b8;
        border-radius: 10px;
        padding: 18px 24px;
        font-size: 1.4rem;
        font-weight: 600;
        color: #0f172a;
        min-height: 70px;
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    .message-placeholder {
        color: #94a3b8;
        font-weight: 400;
        font-size: 1.05rem;
    }

    /* Badge & Status Indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-active {
        background-color: #dcfce7;
        color: #166534;
    }
    .status-inactive {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .status-analyzing {
        background-color: #e0f2fe;
        color: #075985;
    }

    /* Primary & Action Buttons */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    
    /* Disclaimer Card */
    .disclaimer-card {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #92400e;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "sentence_builder" not in st.session_state:
    st.session_state.sentence_builder = SentenceBuilder()
if "last_spoken_message" not in st.session_state:
    st.session_state.last_spoken_message = ""
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "speech_engine" not in st.session_state:
    st.session_state.speech_engine = SpeechEngine()
if "classifier" not in st.session_state:
    st.session_state.classifier = SignClassifier()
if "auto_add" not in st.session_state:
    st.session_state.auto_add = True


def get_dataset_stats() -> dict:
    """Reads dataset sample counts from CSV."""
    counts = {s["id"]: 0 for s in SUPPORTED_SIGNS}
    if DATASET_CSV.exists():
        try:
            df = pd.read_csv(DATASET_CSV)
            if "label" in df.columns:
                existing = df["label"].value_counts().to_dict()
                for k, v in existing.items():
                    if k in counts:
                        counts[k] = v
        except Exception:
            pass
    return counts


# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/docs/images/mediapipe_logo.png",
        width=140,
    )
    st.title("SignBridge Controls")

    # Mode Navigation
    app_mode = st.radio(
        "Navigation",
        ["🤟 Live Sign Recognition", "📊 Dataset & Model Status", "📖 ISL Guide & Help"],
        index=0,
    )

    st.markdown("---")
    st.subheader("⚙️ Settings")
    cam_index = st.number_input("Camera Index", min_value=0, max_value=5, value=DEFAULT_CAMERA_INDEX, step=1)
    conf_thresh = st.slider("Confidence Threshold", 0.40, 0.95, CONFIDENCE_THRESHOLD, 0.05)
    st.session_state.classifier.confidence_threshold = conf_thresh

    st.markdown("---")
    st.subheader("📌 Vocabulary (8-10 Signs)")
    for sign in SUPPORTED_SIGNS:
        with st.expander(f"{sign['name']} ({sign['id']})"):
            st.write(f"**Meaning:** {sign['meaning']}")
            st.write(f"**Hands Required:** {sign['hands']}")
            st.write(f"*{sign['description']}*")

    st.markdown(
        """
        <div class="disclaimer-card">
            <strong>⚠️ Prototype Notice:</strong><br>
            SignBridge is a limited-vocabulary communication aid developed for demonstration.
            Concatenating English glosses does not constitute full ISL grammatical translation.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- HEADER ---------------- #
st.markdown(
    """
    <div class="hero-card">
        <h1 class="hero-title">SignBridge</h1>
        <div class="hero-tagline">Sign Beyond Words.</div>
        <p class="hero-description">
            An accessible assistive vision platform bridging Indian Sign Language (ISL) signers
            with non-signers in real-time. Translates hand landmarks into readable text and audible speech.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- MODE 1: LIVE RECOGNITION ---------------- #
if app_mode == "🤟 Live Sign Recognition":
    col_cam, col_panel = st.columns([1.1, 0.9], gap="large")

    with col_cam:
        st.markdown(
            '<div class="ui-card-title">📹 Live Camera & Landmark Visualization</div>',
            unsafe_allow_html=True,
        )

        # Camera Start/Stop Controls
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            start_btn = st.button("▶ Start Camera", use_container_width=True, type="primary")
        with btn_col2:
            stop_btn = st.button("⏹ Stop Camera", use_container_width=True)
        with btn_col3:
            st.session_state.auto_add = st.checkbox("Auto-add Words", value=st.session_state.auto_add)

        if start_btn:
            st.session_state.camera_running = True
        if stop_btn:
            st.session_state.camera_running = False

        # Live Feed Placeholder
        image_spot = st.empty()
        status_spot = st.empty()

        if not st.session_state.camera_running:
            image_spot.info(
                "Camera is currently stopped. Click **'Start Camera'** above to begin live hand tracking and sign recognition."
            )
            status_spot.markdown(
                '<span class="status-badge status-inactive">● Camera Offline</span>',
                unsafe_allow_html=True,
            )

    with col_panel:
        # Recognized Sign Display Card
        st.markdown(
            '<div class="ui-card-title">🎯 Recognized ISL Sign</div>',
            unsafe_allow_html=True,
        )
        rec_card = st.container()

        with rec_card:
            rec_sign_spot = st.empty()
            rec_conf_spot = st.empty()
            rec_info_spot = st.empty()

        st.markdown("---")

        # Sentence Builder & TTS Card
        st.markdown(
            '<div class="ui-card-title">💬 Message Builder & Speech Output</div>',
            unsafe_allow_html=True,
        )

        message_box_spot = st.empty()

        # Sentence Controls
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
        with ctrl_col1:
            add_word_btn = st.button("➕ Add Current Sign", use_container_width=True)
        with ctrl_col2:
            del_word_btn = st.button("⌫ Delete Last", use_container_width=True)
        with ctrl_col3:
            clear_btn = st.button("🗑 Clear All", use_container_width=True)

        speech_col1, speech_col2 = st.columns([1.2, 0.8])
        with speech_col1:
            speak_btn = st.button("🔊 Speak Message", use_container_width=True, type="primary")
        with speech_col2:
            repeat_btn = st.button("🔁 Repeat Speech", use_container_width=True)

        audio_playback_spot = st.empty()

        # Handle sentence builder actions
        if del_word_btn:
            st.session_state.sentence_builder.delete_last_word()
        if clear_btn:
            st.session_state.sentence_builder.clear()

        # Handle speech triggers
        msg_to_speak = None
        if speak_btn:
            current_sentence = st.session_state.sentence_builder.get_sentence()
            if current_sentence:
                msg_to_speak = current_sentence
                st.session_state.last_spoken_message = current_sentence
            else:
                st.warning("Message is empty. Sign or add words to the message first.")

        if repeat_btn:
            if st.session_state.last_spoken_message:
                msg_to_speak = st.session_state.last_spoken_message
            else:
                st.warning("No previous message to repeat.")

        if msg_to_speak:
            # 1. Local Python SAPI5 narration
            ok_spk, spk_info = st.session_state.speech_engine.speak(msg_to_speak)
            # 2. Browser Web Speech API injection for web clients
            audio_playback_spot.markdown(
                st.session_state.speech_engine.get_browser_speech_html(msg_to_speak),
                unsafe_allow_html=True,
            )
            st.success(f"Speaking: \"{msg_to_speak}\"")

        # Render message box
        active_sentence = st.session_state.sentence_builder.get_sentence()
        if active_sentence:
            message_box_spot.markdown(
                f'<div class="message-display">{active_sentence}</div>',
                unsafe_allow_html=True,
            )
        else:
            message_box_spot.markdown(
                '<div class="message-display"><span class="message-placeholder">Recognized words will appear here...</span></div>',
                unsafe_allow_html=True,
            )

    # ---------------- LIVE CAMERA FEED LOOP ---------------- #
    if st.session_state.camera_running:
        camera = Camera(camera_index=int(cam_index))
        success, err_msg = camera.start()

        if not success:
            st.session_state.camera_running = False
            image_spot.error(f"⚠️ {err_msg}")
            status_spot.markdown(
                '<span class="status-badge status-inactive">● Connection Error</span>',
                unsafe_allow_html=True,
            )
        else:
            tracker = HandTracker()
            classifier: SignClassifier = st.session_state.classifier

            # Ensure model is reloaded if newly trained
            if not classifier.is_trained():
                classifier.load_model()

            try:
                while st.session_state.camera_running:
                    ret, frame, status_str = camera.read_frame()
                    if not ret or frame is None:
                        time.sleep(0.05)
                        continue

                    # MediaPipe hand processing
                    results = tracker.process_frame(frame)
                    num_hands = tracker.count_hands(results)

                    # Landmark visualization overlay
                    display_frame = tracker.draw_landmarks(frame, results)

                    # Extract features and classify
                    features = extract_feature_vector(results)
                    pred = classifier.predict(features, num_hands)

                    # Update visual elements
                    rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    image_spot.image(
                        rgb_display,
                        channels="RGB",
                        use_container_width=True,
                    )

                    # Status indicator badge
                    if num_hands > 0:
                        status_spot.markdown(
                            f'<span class="status-badge status-active">● Camera Active | {num_hands} Hand(s) Detected</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        status_spot.markdown(
                            '<span class="status-badge status-analyzing">● Camera Active | Waiting for Hands in View...</span>',
                            unsafe_allow_html=True,
                        )

                    # Prediction Card Details
                    if pred["is_confident"] and pred["sign_id"]:
                        rec_sign_spot.markdown(
                            f"<h2 style='color:#1e3a8a; margin:0;'>🤟 {pred['name']}</h2>",
                            unsafe_allow_html=True,
                        )
                        rec_conf_spot.progress(
                            min(1.0, max(0.0, pred["confidence"])),
                            text=f"Confidence: {pred['confidence']:.1%}",
                        )
                        rec_info_spot.markdown(f"**Meaning:** {pred['meaning']}")

                        # Auto-add if enabled
                        if st.session_state.auto_add:
                            added = st.session_state.sentence_builder.add_word(pred["sign_id"])
                            if added:
                                current_s = st.session_state.sentence_builder.get_sentence()
                                message_box_spot.markdown(
                                    f'<div class="message-display">{current_s}</div>',
                                    unsafe_allow_html=True,
                                )
                    else:
                        rec_sign_spot.markdown(
                            f"<h3 style='color:#64748b; margin:0;'>{pred['name']}</h3>",
                            unsafe_allow_html=True,
                        )
                        if pred["confidence"] > 0:
                            rec_conf_spot.progress(
                                min(1.0, max(0.0, pred["confidence"])),
                                text=f"Confidence: {pred['confidence']:.1%} (Below threshold)",
                            )
                        else:
                            rec_conf_spot.empty()
                        rec_info_spot.empty()

                    # Handle manual Add Current Sign button
                    if add_word_btn and pred.get("sign_id"):
                        st.session_state.sentence_builder.add_word(pred["sign_id"], force=True)
                        current_s = st.session_state.sentence_builder.get_sentence()
                        message_box_spot.markdown(
                            f'<div class="message-display">{current_s}</div>',
                            unsafe_allow_html=True,
                        )

                    time.sleep(0.01)

            finally:
                camera.stop()
                tracker.close()


# ---------------- MODE 2: DATASET & MODEL STATUS ---------------- #
elif app_mode == "📊 Dataset & Model Status":
    st.subheader("📊 ISL Dataset & Classifier Training Center")

    stats = get_dataset_stats()
    total_samples = sum(stats.values())

    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Total Collected Samples", total_samples)
    col_metric2.metric("Supported Signs", len(SUPPORTED_SIGNS))
    model_exists = MODEL_PATH.exists()
    col_metric3.metric("Model Status", "Trained & Ready" if model_exists else "Not Trained")

    st.markdown("---")

    col_chart, col_train = st.columns([1.1, 0.9])

    with col_chart:
        st.subheader("Sample Distribution per Sign")
        df_stats = pd.DataFrame(
            [{"Sign": s["name"], "ID": s["id"], "Samples": stats[s["id"]]} for s in SUPPORTED_SIGNS]
        )
        st.dataframe(df_stats, use_container_width=True)

    with col_train:
        st.subheader("Model Training")
        st.write(
            "Train a Random Forest classifier directly on the normalized hand landmark dataset."
        )

        if st.button("🚀 Train Model Now", type="primary", use_container_width=True):
            with st.spinner("Training Random Forest Classifier on landmark features..."):
                ok, res = train_sign_model()
                if ok:
                    st.success(f"🎉 Model Trained Successfully! Test Accuracy: {res['accuracy']:.2%}")
                    st.session_state.classifier.load_model()
                    st.json(res)
                else:
                    st.error(f"Training Failed: {res.get('error')}")

        st.markdown("---")
        st.subheader("Interactive Sample Recorder")
        st.info(
            "To record new samples with visual landmark preview, run the standalone tool in your terminal:\n\n"
            "```powershell\n"
            "py collect_samples.py\n"
            "```\n"
            "Hold each sign steadily and press **[SPACE]** to capture bursts of 30 landmark samples."
        )


# ---------------- MODE 3: ISL GUIDE & HELP ---------------- #
else:
    st.subheader("📖 SignBridge Demonstration Guide & ISL Reference")

    st.markdown(
        """
        ### How to Perform the Live Demonstration:
        1. **Start the Camera**: Navigate to **Live Sign Recognition** and click **Start Camera**.
        2. **Position Your Hands**: Face your laptop webcam in good lighting. Ensure hands and wrists are visible in the camera frame.
        3. **Perform a Supported Sign**: Hold one of the 8–10 supported signs (see table below).
        4. **Watch Landmark Tracking**: Observe the MediaPipe 21-point tracking mesh overlay.
        5. **Confirm Sign Recognition**: Once the confidence bar clears the threshold, the sign name appears.
        6. **Build Message**: Allow words to be added automatically or click **Add Current Sign**.
        7. **Narrate Speech**: Click **Speak Message** to hear the computer voice speak the sentence out loud.

        ---
        ### Supported ISL Vocabulary Guide:
        """
    )

    guide_data = []
    for s in SUPPORTED_SIGNS:
        guide_data.append(
            {
                "Sign": s["name"],
                "ID": s["id"],
                "Hands": f"{s['hands']} Hand{'s' if s['hands'] > 1 else ''}",
                "Meaning": s["meaning"],
                "How to Sign": s["description"],
            }
        )
    st.table(pd.DataFrame(guide_data))
