# 🎙️ Convo Bot - Voice-Activated AI Chatbot

**Convo Bot** is a voice-enabled AI chatbot built using Python. It listens to your voice, understands what you're saying using natural language processing (NLP), and responds using text-to-speech (TTS). It's a fun, interactive assistant you can talk to hands-free!

---

## 🧠 Features

- 🎧 **Speech Recognition** using `speech_recognition`
- 🗣️ **Text-to-Speech** using `pyttsx3`
- 🧾 **Intent Classification** with `scikit-learn` and Naive Bayes
- 🔁 Continuous chat loop until the user says "bye" or "exit"
- 🔊 Fully voice-driven interaction

---

## 🛠️ Tech Stack

- Python 3.x
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
- [PyAudio](https://pypi.org/project/PyAudio/) (or suitable audio backend)
- [pyttsx3](https://pypi.org/project/pyttsx3/)
- [scikit-learn](https://scikit-learn.org/)
- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- Naive Bayes Classifier

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/voice-chatbot.git
cd voice-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Chatbot
```bash
python chatbot.py
```



