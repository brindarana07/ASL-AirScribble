 # ASL AirScribble

 ASL AirScribble is an offline real-time American Sign Language to text converter with an additional air drawing mode. It uses a webcam, MediaPipe hand landmarks, OpenCV, scikit-learn, SQLite, and offline text-to-speech.

 The project is designed for Windows laptops and runs fully offline after setup. It does not use paid APIs, cloud services, GPU-only models, or internet-based speech/OCR.

 ## Features

 - Real-time webcam hand tracking with MediaPipe Hands
 - ASL alphabet recognition using landmark-based features
 - Letter-to-word-to-sentence building
 - Stability filtering to reduce flickering predictions
 - Word suggestions and basic autocorrection
 - Offline text-to-speech using `pyttsx3`
 - SQLite session history
 - Custom data collection tool
 - Dataset cleaning tool for faulty samples
 - Random Forest classifier training
 - Air drawing mode with smoothed white brush strokes
 - Trainable air-drawn letter recognition using a small local classifier
 - Fully local model, dataset, drawing, and session storage
 - can handle the motion alphabets like 'j' and 'z'.

 ## Tech Stack

 - Python 3.9 to 3.11
 - OpenCV
 - MediaPipe
 - NumPy
 - scikit-learn
 - joblib
 - pyttsx3
 - SQLite


 ## Project Structure

 ```text
 textconverter/
 |-- main.py
 |-- collect_data.py
 |-- train_classifier.py
 |-- train_air_draw_classifier.py
 |-- clean_dataset.py
 |-- view_history.py
 |-- config.py
 |-- requirements.txt
 |-- test_webcam.py
 |-- test_hand_landmarks.py
 |-- test_tts.py
 |-- modules/
 |   |-- air_drawing.py
 |   |-- air_draw_recognizer.py
 |   |-- classifier.py
 |   |-- database.py
 |   |-- display.py
 |   |-- feature_extractor.py
 |   |-- hand_tracker.py
 |   |-- text_assist.py
 |   |-- tts_engine.py
 |   `-- word_builder.py
 |-- utils/
 |   `-- helpers.py
 |-- data/
 |   `-- .gitkeep
 `-- assets/
     `-- .gitkeep
 ```

 Generated files are stored locally in `textconverter/data/` and are ignored by Git:

 - `landmarks.csv`
 - `model.pkl`
 - `air_draw_model.pkl`
 - `sessions.db`
 - `air_drawings/`
 - `air_draw_samples/`

 ## Installation

 Run these commands from the repository root.

 ```powershell
 py -3.11 -m venv .venv
 .\.venv\Scripts\Activate.ps1
 cd textconverter
 python -m pip install --upgrade pip
 pip install -r requirements.txt
 ```

 If PowerShell blocks activation:

 ```powershell
 Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
 ```

 For later runs:

 ```powershell
 cd "D:\capstone project"
 .\.venv\Scripts\Activate.ps1
 cd textconverter
 ```

 ## Quick Tests

 ```powershell
 python test_webcam.py
 python test_hand_landmarks.py
 python test_tts.py
 ```

 Use `Q` or `Esc` in the OpenCV window to quit the camera tests.

 ## Collect Training Data

 Before ASL recognition works, collect your own landmark samples:

 ```powershell
 python collect_data.py
 ```

 Controls:

 ```text
 A-Z       collect that letter continuously
 0         collect SPACE
 1         collect DELETE
 2         collect NOTHING / idle
 Esc       stop current label capture
 Q         quit
 ```

 Recommended process:

 1. Show one sign clearly.
 2. Press the matching key once.
 3. Hold the sign while changing hand position, distance, and angle slightly.
 4. Collect at least 100 samples per class.
 5. Prefer 200 or more samples per class.
 6. Press `Esc` before switching to another sign.

 Start with a small set such as `A`, `B`, `C`, `SPACE`, `DELETE`, and `NOTHING`, then add more letters.

 ## Train the Model

 ```powershell
 python train_classifier.py
 ```

 The training script:

 - Reads `data/landmarks.csv`
 - Trains a Random Forest classifier
 - Prints sample counts
 - Prints accuracy and classification report
 - Prints a confusion matrix
 - Saves the model to `data/model.pkl`

 If accuracy is low, collect more samples for the confused letters and train again.

 ## Clean Faulty Data

 If you collected wrong samples:

 ```powershell
 python clean_dataset.py
 ```

 You can delete all samples for a label or delete the most recent samples. After cleaning, train again:

 ```powershell
 python train_classifier.py
 ```

 ## Run the App

 ```powershell
 python main.py
 ```

 Choose a mode:

 ```text
 1 = ASL Sign-to-Text mode
 2 = Air Drawing mode
 ```

 Only one mode runs at a time so the app remains smooth on CPU.

 ## ASL Sign-to-Text Controls

 ```text
 TAB       accept top suggestion
 SPACE     commit current word / add space
 BACKSPACE delete last character
 ENTER     finalize and speak sentence
 C         clear text
 H         print recent history in terminal
 T         toggle automatic word speech
 Q or Esc  quit
 ```

 ## Air Drawing Controls

 Gesture controls:

 ```text
 Index finger up          draw
 Index + middle finger up erase
 Open palm                clear canvas
 Closed hand              pause drawing
 ```

 Keyboard controls:

 ```text
 R         recognize current drawing and add it to text
 G then A-Z save current drawing as a training sample
 A-Z       add drawn character manually
 TAB       accept top suggestion
 SPACE     add space
 BACKSPACE delete last character
 S         save drawing image
 C         clear canvas
 Q or Esc  quit
 ```

 Saved drawings are stored in:

 ```text
 textconverter/data/air_drawings/
 ```

 ## Train Air Drawing Recognition

 Air drawing recognition needs a few examples of your handwriting style.

 1. Run the app:

 ```powershell
 python main.py
 ```

 2. Choose Air Drawing mode.
 3. Draw one letter in the air.
 4. Press `G`, then press the correct letter key, for example `A`.
 5. Repeat 15-30 times per letter for the letters you want to recognize.
 6. Train the air drawing model:

 ```powershell
 python train_air_draw_classifier.py
 ```

 7. Run Air Drawing mode again, draw a letter, and press `R` to recognize it.

 Samples are stored locally in:

 ```text
 textconverter/data/air_draw_samples/
 ```

 ## Session History

 ```powershell
 python view_history.py
 ```

 Session history is stored locally in:

 ```text
 textconverter/data/sessions.db
 ```

 ## How It Works

 ```text
 Webcam frame
 -> MediaPipe hand landmark detection
 -> 21 hand landmarks
 -> Wrist-relative normalization
 -> Feature extraction
 -> Random Forest prediction
 -> Majority vote and hold-time filtering
 -> Text generation
 -> Offline speech output
 -> SQLite session storage
 ```

 ## Limitations

 - The ASL classifier must be trained with your own collected landmark data.
 - Air drawing recognition must be trained with your own drawn samples.
 - Accuracy depends on lighting, webcam quality, hand position, and sample variety.
 - Static signs are supported first.
 - Air drawing accuracy improves when letters are drawn in a consistent size and style.
   

 ## Future Improvements

 - Motion recognition for `J` and `Z`
 - Larger air-drawn character dataset
 - User profiles for different hands/cameras
 - Confusion matrix image export
 - Settings window using Tkinter
 - Better sentence-level correction

