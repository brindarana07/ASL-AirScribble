import cv2

from config import (
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    DATABASE_PATH,
    DELETE_LABELS,
    FEATURE_INCLUDE_ANGLES,
    FEATURE_INCLUDE_DISTANCES,
    FEATURE_INCLUDE_FINGER_FLAGS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LANDMARK_CSV_PATH,
    LETTER_HOLD_SECONDS,
    MEDIAPIPE_DETECTION_CONFIDENCE,
    MEDIAPIPE_MAX_NUM_HANDS,
    MEDIAPIPE_TRACKING_CONFIDENCE,
    MIRROR_CAMERA,
    MODEL_PATH,
    NOTHING_LABELS,
    SPACE_LABELS,
    STABLE_FRAME_WINDOW,
    CONFIRM_COOLDOWN_SECONDS,
    TTS_ENABLED,
    TTS_RATE,
    TTS_SPEAK_WORDS_AUTOMATICALLY,
    TTS_VOICE_INDEX,
    TTS_VOLUME,
    VOCABULARY,
)
from modules.air_drawing import AirDrawingSystem
from modules.air_draw_recognizer import AirDrawRecognizer, save_air_draw_sample
from modules.classifier import SignClassifier
from modules.database import SessionDatabase
from modules.display import Display
from modules.feature_extractor import extract_feature_vector, load_label_counts
from modules.hand_tracker import HandTracker
from modules.text_assist import TextAssist
from modules.tts_engine import TTSEngine
from modules.word_builder import WordBuilder
from utils.helpers import FPSCounter, read_key, should_quit


ASL_WINDOW_NAME = "Offline ASL to Text Converter"
AIR_DRAW_WINDOW_NAME = "Air Drawing Mode"
ASL_ALLOWED_LABELS = {
    "SPACE",
    "DELETE",
    "DEL",
    "BACKSPACE",
    "NOTHING",
    "IDLE",
    "_",
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
}


def normalize_asl_prediction(label: str) -> str:
    label = (label or "IDLE").upper()
    return label if label in ASL_ALLOWED_LABELS else "IDLE"


def print_startup_help(model_ready: bool):
    print("Offline ASL to Text Converter")
    print(f"Model path: {MODEL_PATH}")
    print(f"Dataset path: {LANDMARK_CSV_PATH}")
    if not model_ready:
        print("No trained model found yet. Run collect_data.py, then train_classifier.py.")
    print("Shortcuts: TAB=suggestion, SPACE=space, BACKSPACE=delete, ENTER=finalize+speak, C=clear, H=history, T=auto speech, Q=quit")


def commit_current_word(builder, text_assist, database, session_id):
    correction = text_assist.best_correction(builder.current_word) if builder.current_word else None
    event = builder.add_space(correction=correction)
    database.save_event(
        session_id,
        word=event.value if event.kind == "word" else "",
        sentence=builder.full_text(),
    )
    return event


def choose_startup_mode():
    print("\nChoose operating mode:")
    print("1. ASL Sign-to-Text mode")
    print("2. Air Drawing mode")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "ASL"
        if choice == "2":
            return "AIR_DRAW"
        print("Please enter 1 or 2.")


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Close other camera apps and try again.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def run_asl_mode():
    classifier = SignClassifier(MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)
    print_startup_help(classifier.is_ready)

    cap = open_camera()
    tracker = HandTracker(
        max_num_hands=MEDIAPIPE_MAX_NUM_HANDS,
        detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
        tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
    )
    builder = WordBuilder(
        stable_frame_window=STABLE_FRAME_WINDOW,
        hold_seconds=LETTER_HOLD_SECONDS,
        cooldown_seconds=CONFIRM_COOLDOWN_SECONDS,
        space_labels=SPACE_LABELS,
        delete_labels=DELETE_LABELS,
        nothing_labels=NOTHING_LABELS,
    )
    text_assist = TextAssist(VOCABULARY)
    display = Display(FRAME_WIDTH, FRAME_HEIGHT)
    fps_counter = FPSCounter()
    tts = TTSEngine(TTS_ENABLED, TTS_RATE, TTS_VOLUME, TTS_VOICE_INDEX)
    database = SessionDatabase(DATABASE_PATH)
    session_id = database.start_session()
    auto_speak_words = TTS_SPEAK_WORDS_AUTOMATICALLY
    sample_count = sum(load_label_counts(LANDMARK_CSV_PATH).values())

    prediction_label = "IDLE"
    confidence = 0.0
    mode = "idle"
    event_text = ""

    try:
        cv2.namedWindow(ASL_WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            hands = tracker.process_frame(frame)
            tracker.draw_landmarks(frame)

            if not classifier.is_ready:
                prediction_label = "NO MODEL"
                confidence = 0.0
                mode = "collect/train first"
            elif not hands:
                prediction_label = "IDLE"
                confidence = 0.0
                mode = "no hand"
                builder.update("IDLE")
            else:
                features = extract_feature_vector(
                    hands[0].landmarks,
                    FEATURE_INCLUDE_DISTANCES,
                    FEATURE_INCLUDE_ANGLES,
                    FEATURE_INCLUDE_FINGER_FLAGS,
                )
                prediction = classifier.predict(features)
                prediction_label = normalize_asl_prediction(prediction.label if prediction else "IDLE")
                confidence = prediction.confidence if prediction else 0.0
                mode = "detecting"
                event = builder.update(prediction_label)
                if event:
                    if event.kind == "letter":
                        event_text = f"Typed {event.value}"
                        database.save_event(session_id, letter=event.value, sentence=builder.full_text())
                    elif event.kind == "word":
                        event_text = f"Word {event.value}"
                        database.save_event(session_id, word=event.value, sentence=builder.full_text())
                        if auto_speak_words:
                            tts.speak(event.value)
                    elif event.kind == "delete":
                        event_text = "Deleted"
                        database.save_event(session_id, sentence=builder.full_text())
                    elif event.kind == "space":
                        event_text = "Space"
                        database.save_event(session_id, sentence=builder.full_text())

            fps = fps_counter.update()
            suggestions = text_assist.suggestions(builder.current_word)
            display.draw(
                frame,
                current_prediction=prediction_label,
                stable_label=builder.stable_label,
                confidence=confidence,
                current_word=builder.current_word,
                sentence=builder.full_text(),
                mode=mode,
                fps=fps,
                model_ready=classifier.is_ready,
                event_text=event_text,
                stability_progress=builder.hold_progress(),
                auto_speak_words=auto_speak_words,
                hand_count=len(hands),
                sample_count=sample_count,
                suggestions=suggestions,
            )

            cv2.imshow(ASL_WINDOW_NAME, frame)
            key = read_key(1)

            if should_quit(key, ASL_WINDOW_NAME):
                break
            if key == ord(" "):
                event = commit_current_word(builder, text_assist, database, session_id)
                event_text = f"Word {event.value}" if event.kind == "word" else "Manual space"
            elif key == 9:
                suggestions = text_assist.suggestions(builder.current_word)
                if suggestions:
                    event = builder.accept_suggestion(suggestions[0])
                    event_text = f"Accepted {event.value}"
            elif key in (8, 127):
                builder.backspace()
                database.save_event(session_id, sentence=builder.full_text())
                event_text = "Manual delete"
            elif key in (10, 13):
                if builder.current_word:
                    commit_current_word(builder, text_assist, database, session_id)
                text = builder.full_text()
                database.save_event(session_id, sentence=text)
                tts.speak(text)
                event_text = "Final sentence spoken"
            elif key in (ord("c"), ord("C")):
                builder.clear()
                database.save_event(session_id, sentence="")
                event_text = "Cleared"
            elif key in (ord("h"), ord("H")):
                print("\nRecent sessions:")
                for row in database.recent_sessions():
                    print(row)
                event_text = "History printed"
            elif key in (ord("t"), ord("T")):
                auto_speak_words = not auto_speak_words
                event_text = f"Auto speech {'on' if auto_speak_words else 'off'}"
    finally:
        final_sentence = builder.full_text()
        database.end_session(session_id, final_sentence)
        database.close()
        tts.stop()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print(f"Session saved. Final sentence: {final_sentence}")


def run_air_drawing_mode():
    print("Air Drawing Mode")
    print("Index finger up draws. Index + middle erases. Open palm clears.")
    print("R recognizes the drawing. G then A-Z saves a training sample. S saves drawing. Q quits.")

    cap = open_camera()
    tracker = HandTracker(
        max_num_hands=1,
        detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
        tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
    )
    fps_counter = FPSCounter()
    air_drawing = AirDrawingSystem(FRAME_WIDTH, FRAME_HEIGHT)
    air_recognizer = AirDrawRecognizer()
    text_assist = TextAssist(VOCABULARY)
    sample_label_mode = False

    try:
        cv2.namedWindow(AIR_DRAW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            hands = tracker.process_frame(frame)
            tracker.draw_landmarks(frame)
            air_drawing.update(frame, hands)
            fps = fps_counter.update()
            current_word = air_drawing.text.rstrip().split(" ")[-1] if air_drawing.text.strip() else ""
            suggestions = text_assist.suggestions(current_word)
            output = air_drawing.render(frame, fps, suggestions=suggestions)

            cv2.imshow(AIR_DRAW_WINDOW_NAME, output)
            key = read_key(10)

            if should_quit(key, AIR_DRAW_WINDOW_NAME):
                break
            elif key == 9:
                if suggestions:
                    air_drawing.accept_suggestion(suggestions[0])
            elif key in (ord("r"), ord("R")):
                prediction = air_recognizer.predict(air_drawing.canvas)
                if prediction is None:
                    air_drawing.message = "Train air model first: python train_air_draw_classifier.py"
                elif prediction.label == "?":
                    air_drawing.message = f"Unclear drawing ({prediction.confidence:.0%})"
                else:
                    air_drawing.add_character(prediction.label)
                    air_drawing.message = f"Recognized {prediction.label} ({prediction.confidence:.0%})"
            elif key in (ord("g"), ord("G")):
                sample_label_mode = True
                air_drawing.message = "Press A-Z to save this drawing as a sample"
            elif key in (ord("c"), ord("C")):
                air_drawing.clear()
                air_drawing.message = "Canvas cleared"
                sample_label_mode = False
            elif key in (ord("s"), ord("S")):
                air_drawing.save_canvas()
            elif ord("a") <= key <= ord("z"):
                label = chr(key).upper()
                if sample_label_mode:
                    try:
                        save_air_draw_sample(air_drawing.canvas, label)
                        air_drawing.clear()
                        air_drawing.message = f"Saved air sample for {label}"
                    except ValueError as exc:
                        air_drawing.message = str(exc)
                    sample_label_mode = False
                else:
                    air_drawing.add_character(label)
            elif ord("A") <= key <= ord("Z"):
                label = chr(key).upper()
                if sample_label_mode:
                    try:
                        save_air_draw_sample(air_drawing.canvas, label)
                        air_drawing.clear()
                        air_drawing.message = f"Saved air sample for {label}"
                    except ValueError as exc:
                        air_drawing.message = str(exc)
                    sample_label_mode = False
                else:
                    air_drawing.add_character(label)
            elif key == ord(" "):
                air_drawing.add_space()
            elif key in (8, 127):
                air_drawing.backspace()
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print(f"Air drawing text: {air_drawing.text}")


def main():
    selected_mode = choose_startup_mode()
    if selected_mode == "AIR_DRAW":
        run_air_drawing_mode()
    else:
        run_asl_mode()


if __name__ == "__main__":
    main()
