"""
College Help Desk Voice Assistant - Backend & CLI Application
Provides speech-to-text (Whisper + Google Speech Fallback), NLP intent classification (TF-IDF + Linear SVM),
knowledge-base retrieval in Marathi, and text-to-speech (gTTS).
"""

import os
import sys

# ==============================================================================
# 1. WINDOWS DLL & OPENMP CONFIGURATION (PREVENTS WINERROR 1114)
# ==============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure Windows terminal supports UTF-8 Devanagari / Marathi characters
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Register torch/lib into Windows DLL directory search path
try:
    import site
    for site_pkg in site.getsitepackages():
        torch_lib = os.path.join(site_pkg, "torch", "lib")
        if os.path.isdir(torch_lib):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(torch_lib)
                except Exception:
                    pass
            if torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

# ==============================================================================
# 2. FFMPEG PATH CONFIGURATION
# ==============================================================================
COMMON_FFMPEG_PATHS = [
    r"C:\ffmpeg\bin",
    r"C:\ffmpeg",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\tools\ffmpeg\bin",
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links"),
]

for candidate_path in COMMON_FFMPEG_PATHS:
    if os.path.exists(candidate_path) and candidate_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = candidate_path + os.pathsep + os.environ.get("PATH", "")

# CRITICAL FIX: Load torch and whisper BEFORE sklearn or joblib to avoid OpenMP lockup
_WHISPER_MODULE = None
try:
    import torch
    import whisper
    _WHISPER_MODULE = whisper
except Exception as _e:
    pass

import tempfile
import subprocess
import joblib
import numpy as np

# ==============================================================================
# KNOWLEDGE BASE DICTIONARY (EDITABLE PLACEHOLDERS)
# You can customize these answers to match your college's specific rules,
# contact details, office timings, portal URLs, and fees.
# ==============================================================================
KNOWLEDGE_BASE = {
    "admission_query": (
        "प्रथम वर्ष व थेट द्वितीय वर्ष (DSE) प्रवेश प्रक्रिया CET Cell च्या नियमांनुसार सुरू आहे. "
        "अधिकृत माहितीसाठी आणि ARC सेंटर कागदपत्र पडताळणीसाठी कॉलेजच्या ॲडमिशन सेलमध्ये सकाळी १० ते ५ या वेळेत भेट द्या."
    ),
    "fee_query": (
        "कॉलेजची शैक्षणिक फी ऑनलाईन ईआरपी (ERP) पोर्टलद्वारे अथवा बँक चलनाद्वारे भरता येते. "
        "फी हप्त्यांमध्ये (Installments) भरण्यासाठी अकाउंट्स सेक्शनमध्ये रीतसर अर्ज सादर करावा लागतो."
    ),
    "timetable_query": (
        "चालू सेमिस्टरचे लेक्चर्स व प्रॅक्टिकल्सचे वेळापत्रक संबंधित विभागाच्या नोटीस बोर्डवर "
        "आणि कॉलेजच्या पोर्टलवर प्रसिद्ध केले आहे. नियमित वर्ग सकाळी ९:०० वाजता सुरू होतात."
    ),
    "scholarship_query": (
        "महाडीबीटी (MahaDBT), ईबीसी (EBC) व टीएफडब्ल्यूएस (TFWS) शिष्यवृत्तीचे अर्ज पोर्टलवर सुरू आहेत. "
        "कागदपत्रांची पडताळणी कॉलेजच्या स्कॉलरशिप डेस्कवर त्वरित करून घ्यावी."
    ),
    "document_query": (
        "बोनाफाईड प्रमाणपत्र, ट्रान्सक्रिप्ट व लीव्हिंग सर्टिफिकेट (LC) साठी स्टुडंट सेक्शनमध्ये "
        "विहित नमुन्यात अर्ज करावा. अर्ज केल्यानंतर साधारणपणे २ ते ३ कार्यालयीन दिवसांत प्रमाणपत्र मिळते."
    ),
    "contact_query": (
        "कॉलेज कार्यालयाचा अधिकृत दूरध्वनी क्रमांक ०२०-२४५६७८९० असा आहे. "
        "अधिक माहितीसाठी आपण contact@collegehelpdesk.edu.in वर ईमेल करू शकता किंवा रिसेप्शनवर संपर्क साधावा."
    ),
    "hostel_query": (
        "मुले व मुलींसाठी वसतिगृह सुविधा मेरिटनुसार उपलब्ध आहे. "
        "हॉस्टेल मेस, वाय-फाय आणि रात्रीच्या गेट टाइमिंगच्या (९:०० PM) माहितीसाठी वसतिगृह रेक्टर कार्यालयात संपर्क करा."
    ),
    "exam_query": (
        "सत्र परीक्षांचे हॉल तिकीट व अंतिम वेळापत्रक विद्यापीठाच्या परीक्षा पोर्टलवरून डाऊनलोड करता येईल. "
        "केटी (KT) परीक्षा फॉर्म आणि रीव्हॅल्युएशनसाठी कॉलेज परीक्षा सेलशी संपर्क साधा."
    ),
    "library_query": (
        "मध्यवर्ती ग्रंथालय व वाचन कक्ष सोमवार ते शनिवार सकाळी ८:०० ते रात्री ८:०० पर्यंत सुरू असते. "
        "ग्रंथालय कार्डावर २ पुस्तके १५ दिवसांसाठी मिळतात व डिजिटल लायब्ररीमध्ये IEEE जर्नल्स उपलब्ध आहेत."
    ),
    "placement_query": (
        "ट्रेनिंग अँड प्लेसमेंट सेल (TPO) द्वारे १००+ नामांकित कंपन्यांचे कॅम्पस ड्राइव्ह घेतले जातात. "
        "कॉलेजचे सरासरी पॅकेज ५.५ लाख व कमाल पॅकेज १८ लाख आहे. मॉक टेस्ट व नोंदणीसाठी TPO पोर्टल तपासा."
    ),
}

# Human-readable English labels for UI displays
INTENT_DISPLAY_NAMES = {
    "admission_query": "Admission Inquiry (प्रवेश प्रक्रिया)",
    "fee_query": "Fees & Payments (फी व शुल्क)",
    "timetable_query": "Timetable & Classes (वेळापत्रक)",
    "scholarship_query": "Scholarship & Freeship (शिष्यवृत्ती)",
    "document_query": "Certificates & Documents (कागदपत्रे)",
    "contact_query": "Contact & Helpdesk (संपर्क व पत्ता)",
    "hostel_query": "Hostel & Accommodation (वसतिगृह)",
    "exam_query": "Exams & Results (परीक्षा व निकाल)",
    "library_query": "Library & Reading Hall (ग्रंथालय)",
    "placement_query": "Placements & Internships (नोकरी व प्लेसमेंट)",
}

# Global cached singletons
_CLASSIFIER = None
_VECTORIZER = None
_WHISPER_MODEL = None

def load_models(model_path="models/intent_classifier.pkl", vectorizer_path="models/tfidf_vectorizer.pkl"):
    """Load and cache the trained intent classifier and TF-IDF vectorizer."""
    global _CLASSIFIER, _VECTORIZER
    if _CLASSIFIER is None or _VECTORIZER is None:
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Model or vectorizer not found at '{model_path}' / '{vectorizer_path}'. "
                "Please run train_model.py first."
            )
        _CLASSIFIER = joblib.load(model_path)
        _VECTORIZER = joblib.load(vectorizer_path)
    return _CLASSIFIER, _VECTORIZER

def get_whisper_model(model_name="base"):
    """Load and cache the OpenAI Whisper STT model safely."""
    global _WHISPER_MODEL, _WHISPER_MODULE
    if _WHISPER_MODEL is None:
        if _WHISPER_MODULE is None:
            try:
                import torch
                import whisper
                _WHISPER_MODULE = whisper
            except Exception:
                return None
        try:
            print(f"Loading Whisper '{model_name}' model...")
            _WHISPER_MODEL = _WHISPER_MODULE.load_model(model_name)
        except Exception as e:
            print(f"Could not load Whisper model: {e}")
            _WHISPER_MODEL = None
    return _WHISPER_MODEL

def transcribe_audio(audio_path, model_name="base"):
    """
    Transcribe speech from an audio file.
    Primary Engine: OpenAI Whisper.
    Fallback Engine: Google Web Speech Recognition (no torch/c10 DLL dependencies).
    """
    text = ""
    # 1. Primary Engine: OpenAI Whisper
    try:
        model = get_whisper_model(model_name=model_name)
        if model is not None:
            result = model.transcribe(audio_path, fp16=False)
            text = result.get("text", "").strip()
            if text:
                return text
    except Exception as whisper_err:
        print(f"[Whisper Notice]: {whisper_err}. Using Google Speech Fallback...")

    # 2. Fallback Engine: SpeechRecognition (Google API - 100% reliable)
    try:
        import speech_recognition as sr
        from pydub import AudioSegment
        AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"

        r = sr.Recognizer()
        temp_wav = audio_path + "_converted.wav"
        
        # Normalize audio format to 16kHz Mono WAV
        sound = AudioSegment.from_file(audio_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(temp_wav, format="wav")

        with sr.AudioFile(temp_wav) as source:
            r.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = r.record(source)
            try:
                text = r.recognize_google(audio_data, language="en-IN")
            except Exception:
                try:
                    text = r.recognize_google(audio_data, language="hi-IN")
                except Exception:
                    try:
                        text = r.recognize_google(audio_data, language="mr-IN")
                    except Exception:
                        text = ""

        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass

        return text.strip()
    except Exception as fallback_err:
        print(f"[Fallback Speech Error]: {fallback_err}")
        return text.strip()

def predict_intent(text):
    """
    Predict intent for a given user query using the trained TF-IDF + Classifier.
    Returns:
        predicted_intent (str): The predicted intent class name.
        confidence (float): Confidence score between 0.0 and 1.0.
    """
    classifier, vectorizer = load_models()
    X = vectorizer.transform([text])
    
    # Calculate decision scores or probabilities
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X)[0]
        max_idx = np.argmax(probs)
        intent = classifier.classes_[max_idx]
        confidence = float(probs[max_idx])
    elif hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(X)[0]
        if scores.ndim == 1:
            # Softmax over decision values for calibrated confidence
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)
            max_idx = np.argmax(probs)
            intent = classifier.classes_[max_idx]
            confidence = float(probs[max_idx])
        else:
            intent = classifier.predict(X)[0]
            confidence = 1.0
    else:
        intent = classifier.predict(X)[0]
        confidence = 1.0

    return intent, confidence

def get_answer(intent):
    """Retrieve answer string from knowledge base dictionary."""
    return KNOWLEDGE_BASE.get(
        intent,
        "माफ करा, या विषयासंबंधी माहिती सध्या उपलब्ध नाही. कृपया कॉलेजच्या मुख्य कार्यालयाशी संपर्क साधा."
    )

def text_to_speech(text, output_path=None, lang="mr"):
    """
    Convert text to speech using gTTS and save to an MP3 file.
    Returns the path to the saved MP3 file.
    """
    from gtts import gTTS
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"college_helpdesk_{os.getpid()}_{id(text)%10000}.mp3")
    
    # Generate speech
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)
    return output_path

def play_audio(audio_path):
    """Play an audio file on Windows using playsound or PowerShell fallback."""
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    try:
        from playsound import playsound
        playsound(audio_path)
    except Exception:
        # Robust Windows COM MediaPlayer fallback
        try:
            abs_path = os.path.abspath(audio_path).replace("'", "''")
            ps_cmd = (
                f'$p = New-Object -ComObject WMPlayer.OCX; '
                f'$p.URL = "{abs_path}"; '
                f'$p.controls.play(); '
                f'Start-Sleep -Milliseconds 500; '
                f'while ($p.playState -eq 3) {{ Start-Sleep -Milliseconds 250 }}'
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=False)
        except Exception as e:
            print(f"(Note: Could not play audio automatically: {e})")

def record_microphone(duration=5, sample_rate=16000, output_path=None):
    """Record microphone audio for specified duration in seconds."""
    import sounddevice as sd
    import wave

    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"mic_record_{os.getpid()}.wav")

    print(f"\n[*] Listening... Speaking now for {duration} seconds!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("[+] Recording complete.")

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return output_path

def handle_query(query_text, speak=True):
    """Process a typed or transcribed query with Advanced RAG & MemAC pipeline, print details, and speak back."""
    print(f"\n[Question]: {query_text}")
    intent, confidence = predict_intent(query_text)
    display_name = INTENT_DISPLAY_NAMES.get(intent, intent)
    base_answer = get_answer(intent)
    
    # Run Advanced RAG + MemAC Pipeline
    try:
        from advanced_nlp import run_advanced_nlp_pipeline
        nlp_res = run_advanced_nlp_pipeline(
            query=query_text,
            predicted_intent=intent,
            intent_confidence=confidence,
            base_marathi_answer=base_answer
        )
        final_answer = nlp_res["synthesized_response"]["marathi_answer"]
        citation = nlp_res["synthesized_response"]["top_citation"]
        urgency = nlp_res["urgency_info"]["level"]
        slots = nlp_res.get("slots", {})
        cmi = nlp_res.get("code_switch_metrics", {}).get("cmi_score", 0.0)
        
        print(f"[Detected Intent]:    {intent} ({display_name})")
        print(f"[Hybrid Confidence]:  {nlp_res['hybrid_confidence'] * 100:.1f}%")
        print(f"[Query Urgency]:      {urgency}")
        print(f"[Code-Mixing Index]:  CMI {cmi}")
        if slots:
            print(f"[Extracted Slots]:    {slots}")
        print(f"[RAG Grounded Source]: {citation}")
        answer = final_answer
    except Exception as e:
        print(f"[Detected Intent]: {intent} ({display_name})")
        print(f"[Confidence]:     {confidence * 100:.1f}%")
        answer = base_answer
    
    print(f"\n[Answer (Marathi)]:\n{answer}\n")

    if speak:
        print("[Speaking answer back via gTTS...]")
        mp3_path = text_to_speech(answer, lang="mr")
        play_audio(mp3_path)
        try:
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
        except Exception:
            pass

    return {
        "query": query_text,
        "intent": intent,
        "display_name": display_name,
        "confidence": confidence,
        "answer": answer
    }

def main():
    """Command-line interface menu loop."""
    print("=" * 60)
    print("     COLLEGE HELP DESK - VOICE & TEXT ASSISTANT (CLI)     ")
    print("=" * 60)
    print("Pre-loading models...")
    load_models()
    print("Ready!")

    while True:
        print("\n" + "-" * 40)
        print("1. Ask by Voice (Record 5s from microphone)")
        print("2. Ask by Typing")
        print("3. Exit")
        print("-" * 40)
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            try:
                wav_file = record_microphone(duration=5)
                print("[*] Transcribing audio...")
                transcribed_text = transcribe_audio(wav_file)
                try:
                    if os.path.exists(wav_file):
                        os.remove(wav_file)
                except Exception:
                    pass

                if not transcribed_text:
                    print("[!] No speech detected. Please try speaking louder or closer to the mic.")
                    continue

                print(f"[Transcribed]: '{transcribed_text}'")
                confirm = input("Confirm question? (Press Enter to proceed, or type correction): ").strip()
                final_text = confirm if confirm else transcribed_text
                handle_query(final_text, speak=True)
            except Exception as e:
                print(f"[Error recording/transcribing]: {e}")

        elif choice == "2":
            user_text = input("\nEnter your question: ").strip()
            if not user_text:
                print("[!] Empty question entered.")
                continue
            handle_query(user_text, speak=True)

        elif choice == "3" or choice.lower() in ["quit", "exit", "q"]:
            print("\nDhanyavaad! Exiting College Help Desk Assistant. Bye!\n")
            break
        else:
            print("[!] Invalid option. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
