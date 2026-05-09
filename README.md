ASL AirScribble
Offline real-time American Sign Language to text and speech converter with an additional air-drawing mode. The project uses a webcam, MediaPipe hand landmarks, OpenCV, scikit-learn, SQLite, and offline text-to-speech.

This project is designed to run fully offline on a Windows laptop. No cloud API, paid service, GPU, or internet connection is required after setup.

Features
Real-time webcam hand tracking using MediaPipe Hands
ASL alphabet recognition using hand landmark features
Letter-to-word-to-sentence pipeline
Stability filtering to reduce flickering predictions
Word suggestions and basic autocorrection
Offline text-to-speech using pyttsx3
SQLite session history
Data collection tool for custom training
Random Forest classifier training with accuracy report and confusion matrix
Dataset cleaning tool for removing bad samples
Air drawing mode using index fingertip tracking
Fully local data, model, and history storage
Tech Stack
Python 3.9 to 3.11
OpenCV
MediaPipe
NumPy
scikit-learn
joblib
pyttsx3
SQLite
Do not use Python 3.12 for this project because MediaPipe wheels may be unavailable or unreliable.

Project Structure
textconverter/
├── main.py
├── collect_data.py
├── train_classifier.py
├── clean_dataset.py
├── view_history.py
├── config.py
├── requirements.txt
├── test_webcam.py
├── test_hand_landmarks.py
├── test_tts.py
├── modules/
│   ├── air_drawing.py
│   ├── classifier.py
│   ├── database.py
│   ├── display.py
│   ├── feature_extractor.py
│   ├── hand_tracker.py
│   ├── text_assist.py
│   ├── tts_engine.py
│   └── word_builder.py
├── utils/
│   └── helpers.py
├── data/
│   └── .gitkeep
└── assets/
    └── .gitkeep
Generated files such as landmarks.csv, model.pkl, sessions.db, and saved air drawings are stored locally in textconverter/data/ and are ignored by Git.

Installation
Install Python 3.11, then run these commands from the project root:

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
cd textconverter
python -m pip install --upgrade pip
pip install -r requirements.txt
If PowerShell blocks virtual environment activation:

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
For later runs:

cd "D:\capstone project"
.\.venv\Scripts\Activate.ps1
cd textconverter
Quick Test
Test webcam:

python test_webcam.py
Test MediaPipe hand landmarks:

python test_hand_landmarks.py
Test offline speech:

python test_tts.py
Data Collection
Before recognition works, you must collect landmark samples for your own hand, webcam, and lighting.

Run:

python collect_data.py
Controls:

A-Z = collect that letter continuously
0   = SPACE
1   = DELETE
2   = NOTHING
ESC = stop current capture
Q   = quit
Recommended collection process:

Show the sign clearly.
Press the matching key once.
Hold the sign while slightly changing hand position, distance, and angle.
Collect at least 100 samples per class.
Prefer 200 or more samples per class for better accuracy.
Press ESC before switching to another sign.
Start with:

A B C SPACE DELETE NOTHING
Then expand to more letters.

Training
Train the classifier:

python train_classifier.py
The script:

Reads data/landmarks.csv
Trains a Random Forest classifier
Prints sample counts
Prints accuracy
Prints classification report
Prints a confusion matrix
Saves the model to data/model.pkl
If predictions are wrong, collect more samples for confused letters and train again.

Cleaning Bad Data
If you accidentally collected faulty samples:

python clean_dataset.py
You can:

Delete all samples for a label
Delete the last N samples
Save the cleaned dataset
After cleaning, train again:

python train_classifier.py
Running the App
Run:

python main.py
The app asks which mode to run:

1 = ASL Sign-to-Text mode
2 = Air Drawing mode
Only one mode runs at a time to keep performance stable on CPU.

ASL Sign-to-Text Mode
ASL mode shows:

Live webcam feed
Hand landmarks
Current prediction
Stable prediction
Confidence score
Hold progress
Current word
Final sentence
FPS
Model and hand status
Controls:

TAB       = accept top suggestion
SPACE     = commit current word / add space
BACKSPACE = delete last character
ENTER     = finalize and speak sentence
C         = clear text
H         = print recent history in terminal
T         = toggle automatic word speech
Q or ESC  = quit
Air Drawing Mode
Air Drawing mode tracks the index fingertip and draws its path on a virtual canvas.

Gesture controls:

Index finger up          = draw
Index + middle finger up = erase
Open palm                = clear canvas
Closed hand              = pause drawing
Keyboard controls:

A-Z       = add drawn letter to text
TAB       = accept top suggestion
SPACE     = add space
BACKSPACE = delete last character
S         = save drawing image
C         = clear canvas
Q or ESC  = quit
Saved drawings are stored locally in:

textconverter/data/air_drawings/
Session History
View recent sessions:

python view_history.py
Session data is stored locally in:

textconverter/data/sessions.db
How It Works
Webcam frame
→ MediaPipe hand landmark detection
→ 21 hand landmarks
→ Wrist-relative normalization
→ Feature extraction
→ Random Forest prediction
→ Majority vote and hold-time filtering
→ Text generation
→ Offline speech output
→ SQLite session storage
Limitations
The model must be trained with your own collected landmark data.
Accuracy depends on lighting, webcam quality, hand position, and training variety.
Static ASL signs are supported first.
Motion signs such as J and Z need additional sequence tracking.
Air drawing currently uses manual keyboard confirmation for drawn letters.
Future Improvements
Motion recognition for J and Z
Automatic air-drawn character recognition
Multiple user profiles
Confusion matrix image export
Tkinter settings window
Better language model for sentence correction
License
This project is intended for academic and portfolio use .
