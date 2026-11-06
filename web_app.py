"""
College Help Desk - Flask Web Server & API Backend
Connects the HTML/CSS/JS frontend to the trained NLP classifier,
Advanced Multi-Document RAG, MemAC Conversational Memory, and Marathi TTS.
"""

import os
import sys
import uuid
import tempfile
import hashlib
from flask import Flask, render_template, request, jsonify, send_file

# Windows & OpenMP Configuration
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import core backend functions and models
from app import (
    load_models,
    predict_intent,
    get_answer,
    transcribe_audio,
    text_to_speech,
    INTENT_DISPLAY_NAMES,
    KNOWLEDGE_BASE
)
from advanced_nlp import (
    run_advanced_nlp_pipeline,
    get_memac_engine,
    INSTITUTIONAL_KNOWLEDGE_CORPUS
)

app = Flask(__name__)

# Ensure models are loaded at server start
try:
    load_models()
    print("[+] Model and TF-IDF vectorizer loaded successfully.")
except Exception as e:
    print(f"[!] Warning during model load: {e}")

AUDIO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "college_helpdesk_audio")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)


@app.route("/")
def index():
    """Render the main HTML/CSS frontend."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat_api():
    """
    Process student text query through the NLP Pipeline:
    - Intent Classification
    - Dense Vector RAG Retrieval
    - MemAC State Update
    - BIO Slot Tagging & Code-Switching Index
    """
    try:
        data = request.get_json(force=True)
        query = data.get("query", "").strip()
        mode = data.get("mode", "advanced")  # 'advanced' (RAG+MemAC) or 'standard'

        if not query:
            return jsonify({"error": "Empty query provided"}), 400

        # 1. Intent Classification
        intent, confidence = predict_intent(query)
        display_name = INTENT_DISPLAY_NAMES.get(intent, intent)
        base_marathi = get_answer(intent)

        # 2. Process according to selected engine mode
        mem_engine = get_memac_engine()

        if mode == "advanced":
            nlp_res = run_advanced_nlp_pipeline(
                query=query,
                predicted_intent=intent,
                intent_confidence=confidence,
                base_marathi_answer=base_marathi,
                mem_engine=mem_engine
            )
            marathi_answer = nlp_res["synthesized_response"]["marathi_answer"]
            english_answer = nlp_res["synthesized_response"]["english_answer"]
            citation = nlp_res["synthesized_response"]["top_citation"]
            source_category = nlp_res["synthesized_response"]["source_category"]
            hybrid_confidence = nlp_res["hybrid_confidence"]
            urgency_info = nlp_res["urgency_info"]
            code_switch = nlp_res["code_switch_metrics"]
            slots = nlp_res["slots"]
            iob_tags = nlp_res["iob_tags"]
            retrieved_chunks = nlp_res["retrieved_chunks"]
            working_memory = nlp_res["working_memory"]
            episodic_turns = nlp_res["episodic_turns"]
        else:
            # Standard fast mode
            marathi_answer = base_marathi
            english_answer = ""
            citation = "Institutional Standard FAQ"
            source_category = "General"
            hybrid_confidence = confidence
            urgency_info = {"level": "STANDARD / ROUTINE", "score": 0.2, "color": "#10B981", "action": "Standard FAQ"}
            code_switch = {"cmi_score": 0.0, "dominant_language": "Standard", "classification": "Monolingual", "indic_ratio": 0, "english_ratio": 0}
            slots = {}
            iob_tags = []
            retrieved_chunks = []
            working_memory = mem_engine.get_working_memory_summary()
            episodic_turns = len(mem_engine.episodic_history)

        # 3. Dynamic On-Demand Audio Hash Identifier
        text_hash = hashlib.md5(marathi_answer.encode("utf-8")).hexdigest()
        audio_url = f"/api/tts?hash={text_hash}"

        # Save mapping of hash to text for lazy generation
        hash_file = os.path.join(AUDIO_CACHE_DIR, f"{text_hash}.txt")
        with open(hash_file, "w", encoding="utf-8") as f:
            f.write(marathi_answer)

        response_payload = {
            "query": query,
            "intent": intent,
            "display_name": display_name,
            "confidence": round(confidence * 100, 1),
            "hybrid_confidence": round(hybrid_confidence * 100, 1),
            "marathi_answer": marathi_answer,
            "english_answer": english_answer,
            "citation": citation,
            "source_category": source_category,
            "urgency_info": urgency_info,
            "code_switch": code_switch,
            "slots": slots,
            "iob_tags": iob_tags,
            "retrieved_chunks": retrieved_chunks,
            "working_memory": working_memory,
            "episodic_turns": episodic_turns,
            "audio_url": audio_url
        }

        return jsonify(response_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts", methods=["GET"])
def tts_api():
    """Generate or serve cached Marathi speech audio on demand."""
    text_hash = request.args.get("hash", "")
    text = request.args.get("text", "")

    hash_file = os.path.join(AUDIO_CACHE_DIR, f"{text_hash}.txt")
    if not text and text_hash and os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as f:
            text = f.read()

    if not text:
        return jsonify({"error": "No text supplied for TTS"}), 400

    file_hash = text_hash or hashlib.md5(text.encode("utf-8")).hexdigest()
    audio_path = os.path.join(AUDIO_CACHE_DIR, f"{file_hash}.mp3")

    if not os.path.exists(audio_path):
        try:
            # Clean markdown symbols for cleaner speech
            clean_text = text.replace("📌", "").replace("💡", "").replace("💳", "").replace("📄", "").replace("**", "")
            text_to_speech(clean_text, output_path=audio_path, lang="mr")
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return send_file(audio_path, mimetype="audio/mpeg")


@app.route("/api/voice_upload", methods=["POST"])
def voice_upload_api():
    """Handle audio blob uploaded from browser microphone and transcribe with Whisper/Google Speech."""
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file attached"}), 400

        audio_file = request.files["audio"]
        temp_input = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}.webm")
        audio_file.save(temp_input)

        # Transcribe
        transcribed_text = transcribe_audio(temp_input)

        try:
            if os.path.exists(temp_input):
                os.remove(temp_input)
        except Exception:
            pass

        return jsonify({
            "transcribed_text": transcribed_text,
            "success": bool(transcribed_text)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/clear", methods=["POST"])
def clear_memory_api():
    """Reset the MemAC Conversational Memory State."""
    mem_engine = get_memac_engine()
    mem_engine.clear()
    return jsonify({
        "status": "success",
        "message": "Conversational memory reset successfully.",
        "working_memory": mem_engine.get_working_memory_summary()
    })


@app.route("/api/stats", methods=["GET"])
def get_stats_api():
    """Return project, model and dataset statistics."""
    mem_engine = get_memac_engine()
    return jsonify({
        "dataset_size": 13500,
        "intent_count": len(INTENT_DISPLAY_NAMES),
        "intents": INTENT_DISPLAY_NAMES,
        "accuracy": "100.0%",
        "model_architecture": "Linear Support Vector Machine (LinearSVC) + Sublinear TF-IDF (12k features)",
        "rag_chunks_count": len(INSTITUTIONAL_KNOWLEDGE_CORPUS),
        "working_memory": mem_engine.get_working_memory_summary()
    })


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" 🎓 COLLEGE HELP DESK - MODERN WEB APPLICATION ")
    print(" Server running on: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
