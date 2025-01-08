import random
import json
import pickle
import numpy as np
import nltk
from textblob import TextBlob
import tensorflow as tf
import tkinter as tk
from tkinter import scrolledtext, messagebox
from googletrans import Translator  # Import Translator for language translation
import speech_recognition as sr  # Import for speech recognition

# Load necessary NLTK resources
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and load model data
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

# Initialize translator for multi-language support
translator = Translator()

# Speech recognizer setup
recognizer = sr.Recognizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def analyze_sentiment(sentence):
    blob = TextBlob(sentence)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0.1:
        return 'positive'
    elif sentiment_score < -0.1:
        return 'negative'
    else:
        return 'neutral'

def get_response(ints, intents, sentiment):
    result = None
    for i in intents['intents']:
        if i['tag'] == ints[0]['intent']:
            result = random.choice(i['responses'])
            break

    if result is None:
        result = "I'm sorry, I didn't understand that. Can you try again?"

    if sentiment != 'neutral':
        if sentiment == 'positive':
            result = f"That's great! {result}"
        elif sentiment == 'negative':
            result = f"I'm sorry to hear that. {result}"
    return result

def display_help():
    help_text = (
        "Welcome! Here's how to interact with me:\n"
        "1. Ask for a joke: 'Tell me a joke!'\n"
        "2. Ask a question: 'What is your name?' or 'How are you?'\n"
        "3. Share how you're feeling: 'I'm feeling great!' or 'I'm sad.'\n"
        "4. Get help with commands: Type 'help'.\n"
        "5. Exit: Close the window.\n"
        "6. Change language: Use the dropdown menu to select your preferred language.\n"
        "7. Use speech input: Click the 'Speak' button and start speaking.\n"
    )
    return help_text

def translate_response(response, target_language):
    try:
        translated = translator.translate(response, dest=target_language)
        return translated.text
    except Exception as e:
        return "I'm sorry, I couldn't translate the response."

def recognize_speech():
    global selected_language
    try:
        speak_dialog = tk.Toplevel(root)
        speak_dialog.title("Speak Now")
        speak_dialog.geometry("250x100")
        tk.Label(speak_dialog, text="Please speak something...", font=("Arial", 12)).pack(pady=20)
        speak_dialog.update()

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
            chat_area.config(state=tk.NORMAL)
            chat_area.insert(tk.END, "Bot: Listening... Please speak now.\n", 'bot')
            chat_area.config(state=tk.DISABLED)
            chat_area.yview(tk.END)

            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            speak_dialog.destroy()

            speech_text = recognizer.recognize_google(audio)
            chat_area.config(state=tk.NORMAL)
            chat_area.insert(tk.END, f"User (spoken): ", 'user')
            chat_area.insert(tk.END, speech_text + "\n")
            chat_area.config(state=tk.DISABLED)
            chat_area.yview(tk.END)

            send_message_from_speech(speech_text)

    except sr.WaitTimeoutError:
        speak_dialog.destroy()
        chat_area.config(state=tk.NORMAL)
        chat_area.insert(tk.END, "Bot: Listening timed out. No speech detected. Please try again.\n", 'bot')
        chat_area.config(state=tk.DISABLED)
        chat_area.yview(tk.END)

    except sr.UnknownValueError:
        speak_dialog.destroy()
        chat_area.config(state=tk.NORMAL)
        chat_area.insert(tk.END, "Bot: Sorry, I couldn't understand your speech. Please speak clearly and try again.\n", 'bot')
        chat_area.config(state=tk.DISABLED)
        chat_area.yview(tk.END)

    except sr.RequestError as e:
        speak_dialog.destroy()
        chat_area.config(state=tk.NORMAL)
        chat_area.insert(tk.END, f"Bot: Error with the speech recognition service: {e}\n", 'bot')
        chat_area.config(state=tk.DISABLED)
        chat_area.yview(tk.END)

def send_message_from_speech(speech_text):
    user_message = speech_text
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "User (spoken): ", 'user')
    chat_area.insert(tk.END, user_message + "\n")
    if user_message.lower() == 'help':
        response = display_help()
    else:
        ints = predict_class(user_message)
        sentiment = analyze_sentiment(user_message)
        response = get_response(ints, intents, sentiment)
        target_language = selected_language.get()
        if target_language != 'en':
            response = translate_response(response, target_language)
    chat_area.insert(tk.END, "Bot: ", 'bot')
    chat_area.insert(tk.END, response + "\n")
    chat_area.config(state=tk.DISABLED)
    chat_area.yview(tk.END)

# Create the main window
root = tk.Tk()
root.title("Chatbot")
root.geometry("400x700")

# Create a scrolled text area
chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD)
chat_area.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
chat_area.config(state=tk.DISABLED)

# Configure tags for different text colors
chat_area.tag_config('user', foreground='blue')
chat_area.tag_config('bot', foreground='green')

# Display default help text when the application starts
chat_area.config(state=tk.NORMAL)
chat_area.insert(tk.END, "Bot: ", 'bot')
chat_area.insert(tk.END, display_help() + "\n")
chat_area.config(state=tk.DISABLED)

# Add a dropdown for language selection
language_options = {'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Chinese': 'zh-cn'}
selected_language = tk.StringVar(value='en')

def set_language(lang_code):
    global selected_language
    selected_language.set(lang_code)

language_label = tk.Label(root, text="Select Language:")
language_label.pack(pady=5)

language_menu = tk.OptionMenu(root, selected_language, *language_options.keys(), command=lambda selection: set_language(language_options[selection]))
language_menu.pack()

# Create an entry box for user input
user_input = tk.Entry(root, width=50)
user_input.pack(pady=10, padx=10)

def send_message(event=None):
    user_message = user_input.get()
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "User: ", 'user')
    chat_area.insert(tk.END, user_message + "\n")
    user_input.delete(0, tk.END)

    if user_message.lower() == 'help':
        response = display_help()
    else:
        ints = predict_class(user_message)
        sentiment = analyze_sentiment(user_message)
        response = get_response(ints, intents, sentiment)
        target_language = selected_language.get()
        if target_language != 'en':
            response = translate_response(response, target_language)

    chat_area.insert(tk.END, "Bot: ", 'bot')
    chat_area.insert(tk.END, response + "\n")
    chat_area.config(state=tk.DISABLED)
    chat_area.yview(tk.END)

user_input.bind("<Return>", send_message)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(pady=10)

speak_button = tk.Button(root, text="Speak", command=recognize_speech)
speak_button.pack(pady=10)

# Run the application
root.mainloop()
