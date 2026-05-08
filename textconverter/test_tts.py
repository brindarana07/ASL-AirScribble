try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from config import TTS_RATE, TTS_VOICE_INDEX, TTS_VOLUME


def main():
    if pyttsx3 is None:
        print("Error: pyttsx3 is not installed. Install it with 'pip install pyttsx3'.")
        return
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")

    if voices:
        voice_index = min(TTS_VOICE_INDEX, len(voices) - 1)
        engine.setProperty("voice", voices[voice_index].id)

    engine.setProperty("rate", TTS_RATE)
    engine.setProperty("volume", TTS_VOLUME)

    print("Available offline voices:")
    for index, voice in enumerate(voices):
        print(f"{index}: {voice.name}")

    text = "Offline text to speech is working."
    print(f"Speaking: {text}")
    engine.say(text)
    engine.runAndWait()
    engine.stop()


if __name__ == "__main__":
    main()
