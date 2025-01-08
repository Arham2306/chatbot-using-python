import speech_recognition as sr

recognizer = sr.Recognizer()

try:
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... Please speak.")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You said:", text)
except sr.UnknownValueError:
    print("Sorry, could not understand the audio.")
except sr.RequestError as e:
    print(f"Could not request results from the speech recognition service: {e}")
except sr.WaitTimeoutError:
    print("Listening timed out.")