# 🚀 SignBridge – Sign Beyond Words

> **Tagline:** Sign Beyond Words.

---

### 💡 What's the idea?
Over 18 million Deaf and hard-of-hearing individuals live in India, communicating predominantly through Indian Sign Language (ISL). Because the broader public rarely understands ISL, everyday interactions in healthcare centers, government offices, transit stations, and public spaces often face steep communication barriers.

**SignBridge** is an assistive communication web application designed to bridge this divide. Using standard webcam video and Google MediaPipe hand landmark tracking, SignBridge recognizes a curated set of **10 predefined everyday ISL signs** in real time, normalizes hand geometry to be independent of camera distance and hand size, assembles recognized signs into readable messages, and speaks them aloud via text-to-speech.

> **Prototype Scope & Disclaimer:**  
> SignBridge is an educational hackathon prototype built to demonstrate real-time computer vision feasibility with a limited vocabulary of 10 everyday signs. It is **not** an exhaustive sign language translation engine. Concatenating sign glosses serves as an accessible functional communication aid, but does not represent the full grammatical, spatial, and facial syntax of conversational Indian Sign Language.

---

### 🌐 Check it out!
- **Live Website:** [https://signbridge-sooty-ten.vercel.app/](https://signbridge-sooty-ten.vercel.app/)
- **GitHub Repo:** [https://github.com/NidhiyaS25/SignBridge](https://github.com/NidhiyaS25/SignBridge)
- **Streamlit Cloud Deployment:** [Deploy on Streamlit Community Cloud](https://share.streamlit.io/deploy?repository=NidhiyaS25/SignBridge&branch=stable&mainModule=app.py)

---

### 🛠️ What did you use?

| Category | Technologies & Libraries |
|---|---|
| **Core Languages** | Python 3.12, JavaScript (ES6+), HTML5, CSS3 |
| **Computer Vision** | Google MediaPipe Tasks API (`HandLandmarker`), MediaPipe JavaScript Web SDK, OpenCV (`opencv-python`) |
| **Machine Learning & Data** | Scikit-learn (`RandomForestClassifier`), NumPy, Pandas, Joblib |
| **Web Frameworks & UI** | Streamlit, Responsive Single-Page Web Client |
| **Speech Synthesis (TTS)** | Python `pyttsx3` (Windows SAPI5) & Browser Web Speech API (`window.speechSynthesis`) |
| **Deployment & Hosting** | Vercel (Production Web App), Streamlit Community Cloud (Python Backend) |
| **Version Control & Quality** | Git, GitHub, Automated Python Pipeline Test Suite |

---

### 👥 The Humans Behind It
- **Emi Mariyam Shiju** – [@emimariam2006](https://github.com/emimariam2006) *(Main Developer)*
- **Nidhiya** – [@NidhiyaS25](https://github.com/NidhiyaS25) *(Collaborator)*

---

### ⚙️ Hardware Corner
- **Computer:** Standard laptop / PC running Windows 11 (64-bit)
- **Webcam:** Integrated laptop webcam or standard USB webcam (tested at 640×480 resolution @ 30 FPS)
- **Audio:** Standard built-in laptop speakers / headphones for speech playback
- **Specialized Wearables:** None required (100% software-based camera vision)

---

### 📸 Show Us What You Built!

*Screenshots and demo recordings of the live SignBridge interface:*

```
+-------------------------------------------------------------------------------+
|                       [ SCREENSHOT PLACEHOLDER 1 ]                            |
|                                                                               |
|   Live Webcam Feed showing 21 MediaPipe hand tracking landmark points,        |
|   skeletal connections, and real-time sign recognition (e.g., 'HELLO').       |
|                                                                               |
|   File Path: ./docs/screenshots/live_hand_tracking.png                        |
+-------------------------------------------------------------------------------+
```

```
+-------------------------------------------------------------------------------+
|                       [ SCREENSHOT PLACEHOLDER 2 ]                            |
|                                                                               |
|   Interactive Message Builder assembling recognized words with                |
|   Text-to-Speech audio narration and vocabulary reference grid.              |
|                                                                               |
|   File Path: ./docs/screenshots/message_builder_and_speech.png                |
+-------------------------------------------------------------------------------+
```

```
+-------------------------------------------------------------------------------+
|                       [ DEMO VIDEO / GIF PLACEHOLDER ]                        |
|                                                                               |
|   Short video / GIF walkthrough demonstrating:                                |
|   Camera Start -> Hand Signing -> Text Recognition -> Speech Narration.      |
|                                                                               |
|   File Path: ./docs/screenshots/signbridge_demo.gif                           |
+-------------------------------------------------------------------------------+
```

> *(To attach screenshots: place your PNG/GIF files in the `docs/screenshots/` folder and link them directly above).*

---

### ✨ Cool Things It Does

1. **Real-Time 21-Point Hand Pose Tracking**  
   Tracks 21 3D landmarks per hand on one or both hands simultaneously using MediaPipe, rendering high-contrast skeletal connections and cyan fingertip indicators at interactive frame rates.

2. **Translation & Scale-Invariant Landmark Normalization**  
   Translates landmark coordinates relative to the wrist (landmark 0) and scales them by the hand's bounding span into a normalized 126-dimensional feature vector. Recognition remains robust whether the signer is close to or far from the camera.

3. **Curated ISL Vocabulary Recognition with Anti-Flicker Smoothing**  
   Recognizes 10 genuine ISL everyday communication signs:
   - **1-Hand Signs:** `HELLO`, `THANK_YOU`, `YES`, `NO`, `PLEASE`, `GOOD`, `WATER`, `BYE`
   - **2-Hand Signs:** `NAMASTE`, `HELP`  
   Applies a $\ge 65\%$ confidence threshold and a 7-frame sliding window majority vote to prevent rapid prediction flickering.

4. **Interactive Sentence Builder with Cooldown Debouncing**  
   Assembles isolated recognized signs into readable sentences. Employs a 1.2–1.6 second cooldown debounce to prevent continuous video frames from spamming duplicate words, and provides instant **➕ Add Word**, **⌫ Delete Last**, and **🗑 Clear** controls.

5. **Dual-Engine Audible Speech Output**  
   Converts constructed messages into audible speech using Python `pyttsx3` (Windows SAPI5) on desktop and the native browser Web Speech API (`SpeechSynthesis`) on the live Vercel web portal with Indian English (`en-IN`) pronunciation support.

---

### 📖 Supported Vocabulary Reference

| # | Sign Name | ID | Hands | Intent / Meaning | Visual Technique |
|---|---|---|---|---|---|
| 1 | **Hello** | `HELLO` | 1 | Greeting / Welcome | Open hand raised near temple/head, angled outward in polite salute/wave |
| 2 | **Namaste** | `NAMASTE` | 2 | Respectful Greeting | Both palms pressed flat together at center chest (*Anjali Mudra*) |
| 3 | **Thank You** | `THANK_YOU` | 1 | Expressing Gratitude | Flat hand fingertips touching chin, extending forward toward listener |
| 4 | **Yes** | `YES` | 1 | Affirmation / Agreement | Closed fist held forward, nodding downward from the wrist |
| 5 | **No** | `NO` | 1 | Negation / Disagreement | Index finger extended upright waving gently side-to-side |
| 6 | **Please** | `PLEASE` | 1 | Polite Request | Open flat palm placed gently over chest/heart |
| 7 | **Help** | `HELP` | 2 | Requesting Assistance | Dominant fist with upright thumb resting on flat open palm of other hand |
| 8 | **Good** | `GOOD` | 1 | Quality / Well done | Firm upright thumbs-up gesture at chest level |
| 9 | **Water** | `WATER` | 1 | Thirst / Drink | 'W' handshape (index, middle, ring upright) tapped gently at lips |
| 10 | **Goodbye** | `BYE` | 1 | Parting Farewell | Open palm facing forward, waving side-to-side |

---

### 💻 Quick Start & Local Execution

```powershell
# 1. Clone repository
git clone https://github.com/NidhiyaS25/SignBridge.git
cd SignBridge

# 2. Set up virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run pipeline diagnostics
py tests/test_pipeline.py

# 5. Launch local Streamlit application
streamlit run app.py
```

---

**Built with curiosity at Saturday Hack Night 💜**
