"""
College Help Desk - Advanced Voice & RAG Streamlit Web Application
=================================================================
A state-of-the-art multilingual assistant for college helpdesk queries in
romanized Marathi/Hindi code-mixed with English (Maranglish / Hinglish).

Key NLP Capabilities:
1. Multi-Document Dense Vector RAG (Retrieval-Augmented Generation)
2. MemAC-LLM: Memory-Augmented Conversational State & Entity Working Memory
3. Joint Intent Classification + Token-Level BIO Sequence Tagging (NER)
4. Multilingual Code-Switching Index (CMI) Profiling
5. Student Query Urgency & Administrative Priority Triage
6. Bilingual (Marathi/English) Answer Synthesis with gTTS Audio Playback
"""

import os
import sys
import re

# Prevent Intel OpenMP runtime collision on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Register torch/lib into Windows DLL search paths
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

# Pre-load torch and whisper safely
try:
    import torch
    import whisper
except Exception:
    pass

import io
import time
from datetime import datetime
import streamlit as st
from gtts import gTTS

# Import core backend functions
from app import (
    load_models,
    predict_intent,
    get_answer,
    transcribe_audio,
    INTENT_DISPLAY_NAMES,
    KNOWLEDGE_BASE
)

# Import 4th-Year Advanced NLP & MemAC RAG Engine
from advanced_nlp import (
    run_advanced_nlp_pipeline,
    get_memac_engine,
    MemACMemoryEngine
)

# ==============================================================================
# EDITABLE INSTITUTION & PROJECT CONFIGURATION
# ==============================================================================
COLLEGE_NAME = "Vidyalankar Institute of Technology"
COLLEGE_TAGLINE = "Autonomous Engineering & Technology Institute"
CAMPUS_LOCATION = "Wadala (E), Mumbai, Maharashtra, India"
STUDENT_HELPLINE = "+91 (022) 2416-1126"
OFFICIAL_EMAIL = "helpdesk@vit.edu.in"
OFFICE_HOURS = "Mon - Fri: 9:30 AM to 5:30 PM | Sat: 9:30 AM to 1:30 PM"
PROJECT_TITLE = "College Help Desk AI Voice Assistant"
PROJECT_SUBTITLE = "Advanced NLP & Generative AI • Dense Vector RAG • MemAC Conversational Memory"

# Set page configuration
st.set_page_config(
    page_title=f"{PROJECT_TITLE} | {COLLEGE_NAME}",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Warm Educational Palette with Plus Jakarta Sans & Playfair Display)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #1F2937;
}

code, pre, .mono-font {
    font-family: 'JetBrains Mono', monospace !important;
}

/* App Header Card */
.header-card {
    background: linear-gradient(135deg, #FFFDF9 0%, #FFF7ED 100%);
    border: 1px solid #FED7AA;
    border-radius: 18px;
    padding: 24px 28px;
    text-align: center;
    box-shadow: 0 10px 25px -5px rgba(234, 88, 12, 0.08), 0 8px 10px -6px rgba(234, 88, 12, 0.04);
    margin-bottom: 20px;
}

.bell-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, #FFEDD5 0%, #FED7AA 100%);
    border: 2px solid #FB923C;
    border-radius: 50%;
    font-size: 26px;
    margin-bottom: 10px;
    box-shadow: 0 4px 12px rgba(234, 88, 12, 0.15);
}

.main-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #9A3412;
    margin: 0;
    letter-spacing: -0.5px;
}

.subtitle {
    font-size: 0.95rem;
    color: #7C2D12;
    font-weight: 600;
    margin-top: 4px;
    margin-bottom: 10px;
}

.badge-tag {
    display: inline-block;
    background: #FEF3C7;
    color: #92400E;
    font-size: 0.76rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 9999px;
    border: 1px solid #FCD34D;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 2px;
}

/* Chat Bubbles */
.user-bubble {
    background: #FFF7ED;
    border-left: 4px solid #EA580C;
    border-top: 1px solid #FED7AA;
    border-right: 1px solid #FED7AA;
    border-bottom: 1px solid #FED7AA;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 2px 6px rgba(234, 88, 12, 0.05);
}

.assistant-bubble {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
}

.intent-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #EEF2FF;
    color: #4338CA;
    border: 1px solid #C7D2FE;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
    margin-right: 6px;
}

.conf-pill {
    display: inline-flex;
    align-items: center;
    background: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    margin-right: 6px;
}

.rag-pill {
    display: inline-flex;
    align-items: center;
    background: #FDF4FF;
    color: #86198F;
    border: 1px solid #F0ABFC;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    margin-right: 6px;
}

.urgency-pill {
    display: inline-flex;
    align-items: center;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
}

.answer-marathi {
    font-size: 1.06rem;
    line-height: 1.68;
    color: #111827;
    margin: 10px 0 14px 0;
    font-weight: 500;
}

/* Inspection Cards */
.inspect-card {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 14px;
    margin-top: 8px;
    margin-bottom: 8px;
}

.bio-token {
    display: inline-block;
    padding: 2px 6px;
    margin: 2px;
    border-radius: 4px;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
}

.bio-O { background: #F3F4F6; color: #4B5563; border: 1px solid #E5E7EB; }
.bio-B { background: #DBEAFE; color: #1E40AF; border: 1px solid #BFDBFE; font-weight: 600; }
.bio-I { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; font-weight: 600; }

.memory-box {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 10px;
    padding: 12px;
    font-size: 0.86rem;
}

.verify-card {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 12px;
    padding: 16px;
    margin: 16px 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "transcribed_draft" not in st.session_state:
    st.session_state["transcribed_draft"] = None

if "last_recorded_audio" not in st.session_state:
    st.session_state["last_recorded_audio"] = None

if "memac_engine" not in st.session_state:
    st.session_state["memac_engine"] = MemACMemoryEngine()

@st.cache_resource(show_spinner=False)
def init_assistant():
    """Pre-load classification models once per session."""
    load_models()
    return True

init_assistant()

def synthesize_marathi_audio(text: str) -> bytes:
    """Generate audio bytes for Marathi text using gTTS."""
    fp = io.BytesIO()
    # Clean markdown formatting before sending to TTS engine
    clean_speech_text = re.sub(r"[*#_`\[\]()]", "", text)
    tts = gTTS(text=clean_speech_text, lang="mr", slow=False)
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

def process_and_add_query(query_text: str, engine_mode: str = "Neural RAG + MemAC-LLM (Advanced)"):
    """
    Executes NLP pipeline based on chosen mode:
    - Neural RAG + MemAC-LLM: 6-stage RAG, Working Memory, Slot Filling & CMI Analysis.
    - Standard Fast Classifier: Linear SVM baseline.
    """
    cleaned = query_text.strip()
    if not cleaned:
        return

    with st.spinner("🧠 Executing Dense Vector RAG & Conversational Memory Pipeline..."):
        # 1. Base intent prediction via Linear SVM
        intent, intent_conf = predict_intent(cleaned)
        base_answer = get_answer(intent)
        display_name = INTENT_DISPLAY_NAMES.get(intent, intent)

        # 2. Run Advanced NLP & MemAC RAG Pipeline
        mem_engine = st.session_state["memac_engine"]
        nlp_result = run_advanced_nlp_pipeline(
            query=cleaned,
            predicted_intent=intent,
            intent_confidence=intent_conf,
            base_marathi_answer=base_answer,
            mem_engine=mem_engine
        )

        final_answer = nlp_result["synthesized_response"]["marathi_answer"]
        final_english = nlp_result["synthesized_response"]["english_answer"]
        top_citation = nlp_result["synthesized_response"]["top_citation"]
        confidence = nlp_result["hybrid_confidence"]

        # 3. Text-to-Speech audio synthesis
        audio_bytes = synthesize_marathi_audio(final_answer)

        # 4. Insert into session state
        st.session_state["chat_history"].insert(0, {
            "query": cleaned,
            "intent": intent,
            "display_name": display_name,
            "confidence": confidence,
            "intent_confidence": intent_conf,
            "answer": final_answer,
            "english_answer": final_english,
            "citation": top_citation,
            "audio_bytes": audio_bytes,
            "nlp_result": nlp_result,
            "engine_mode": engine_mode,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown(f"### 🏫 {COLLEGE_NAME}")
    st.caption(COLLEGE_TAGLINE)
    st.markdown("---")

    st.markdown("#### ⚙️ NLP Engine Settings")
    engine_selection = st.radio(
        "Active Intelligence Engine:",
        [
            "Neural RAG + MemAC-LLM (Advanced)",
            "Standard Fast Classifier (Linear SVM)"
        ],
        index=0,
        help="Neural RAG + MemAC-LLM incorporates multi-document vector retrieval, entity working memory, NER sequence slot tagging, and code-switching index profiling."
    )

    st.markdown("---")
    st.markdown("#### 🧠 MemAC Live Working Memory")
    current_mem = st.session_state["memac_engine"].get_working_memory_summary()
    st.markdown(f"""
    <div class="memory-box">
        <strong>🎓 Active Student Context:</strong><br/>
        • <b>Branch:</b> <span style="color:#047857;">{current_mem['STUDENT_BRANCH']}</span><br/>
        • <b>Academic Year:</b> <span style="color:#047857;">{current_mem['ACADEMIC_YEAR']}</span><br/>
        • <b>Admission Quota:</b> <span style="color:#047857;">{current_mem['ADMISSION_CATEGORY']}</span><br/>
        • <b>Last Topic:</b> <span style="color:#047857;">{current_mem['LAST_ACTIVE_TOPIC']}</span><br/>
        • <b>Memory Turns:</b> <code>{len(st.session_state['memac_engine'].episodic_history)}</code>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Reset Working Memory", use_container_width=True):
        st.session_state["memac_engine"].clear()
        st.rerun()

    st.markdown("---")
    with st.expander("🔬 Advanced NLP Architecture Breakdown", expanded=False):
        st.markdown("""
        **1. Dense Semantic Vector RAG:**
        - Institutional corpus chunked into 30+ domain sub-documents.
        - Sublinear Cosine Similarity vector retrieval with $k=3$ top chunks.
        
        **2. MemAC Conversational Memory:**
        - **Working Memory:** Maintains stateful student entity slots across turns.
        - **Episodic Buffer:** Contextual query enrichment for follow-up questions.
        
        **3. Joint Slot Filling & NER (BIO Tagging):**
        - Sequence tagging identifying `B-BRANCH`, `B-CATEGORY`, `B-DOCUMENT`, `B-PAYMENT_MODE`.
        
        **4. Multilingual Linguistic Profiling:**
        - Calculates Code-Mixing Index ($CMI$) for Indic-English code-switching.
        
        **5. Urgency & Administrative Triage:**
        - Flags Critical / Time-Sensitive / Routine priority queries.
        """)

    with st.expander("📌 College Contact & Hours", expanded=False):
        st.markdown(f"**📍 Campus:** {CAMPUS_LOCATION}")
        st.markdown(f"**📞 Helpline:** `{STUDENT_HELPLINE}`")
        st.markdown(f"**✉️ Email:** `{OFFICIAL_EMAIL}`")
        st.markdown(f"**⏰ Office Hours:** {OFFICE_HOURS}")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation History", use_container_width=True):
        st.session_state["chat_history"] = []
        st.session_state["transcribed_draft"] = None
        st.session_state["last_recorded_audio"] = None
        st.rerun()

# ==============================================================================
# MAIN PAGE HEADER
# ==============================================================================
st.markdown(f"""
<div class="header-card">
    <div class="bell-badge">🎓</div>
    <h1 class="main-title">{PROJECT_TITLE}</h1>
    <p class="subtitle">{COLLEGE_NAME} • AI Student Help Desk</p>
    <div>
        <span class="badge-tag" style="background:#EEF2FF; color:#4338CA; border-color:#C7D2FE;">⚡ Multi-Document RAG</span>
        <span class="badge-tag" style="background:#FDF4FF; color:#86198F; border-color:#F0ABFC;">🧠 MemAC Working Memory</span>
        <span class="badge-tag" style="background:#ECFDF5; color:#047857; border-color:#A7F3D0;">🏷️ BIO Sequence Slot Tagging</span>
        <span class="badge-tag" style="background:#FFFBEB; color:#92400E; border-color:#FCD34D;">🌐 Code-Switching Index (CMI)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# QUICK SAMPLE CHIPS
# ==============================================================================
st.markdown("##### 💡 Try Sample Student Queries (Code-Mixed Maranglish & English)")
chips = [
    ("🎓 CS Admission", "first year computer science engineering admission kuthun apply karu?"),
    ("💳 Fee Installment", "college fee installment madhe pay karta yete ka?"),
    ("📅 Timetable PDF", "semester class timetable kuthun download karu?"),
    ("📜 Bonafide Certificate", "bonafide certificate milayla kiti divas lagtat student section madhe?"),
    ("💼 Highest Placements", "campus placement highest package kiti gela ya year la?"),
    ("🏢 Hostel Gate Rules", "hostel mess food quality kashi ahe and gate timing kay ahe?"),
    ("📝 Hall Ticket Admit Card", "exam hall ticket admit card download kasa karaycha?"),
    ("📚 Library Book Issue", "library madhun reference book issue kashi karaychi?"),
    ("💰 TFWS Scholarship", "TFWS seat sathi family income limit kiti pahije?"),
    ("📞 Office Helpline", "college administration office contact number milel ka?"),
]

cols_r1 = st.columns(5)
for i in range(5):
    label, query = chips[i]
    if cols_r1[i].button(label, key=f"chip_{i}", use_container_width=True):
        process_and_add_query(query, engine_selection)
        st.rerun()

cols_r2 = st.columns(5)
for i in range(5):
    label, query = chips[i + 5]
    if cols_r2[i].button(label, key=f"chip_{i+5}", use_container_width=True):
        process_and_add_query(query, engine_selection)
        st.rerun()

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# INPUT SECTION: VOICE & TYPING TABS
# ==============================================================================
tab_voice, tab_type = st.tabs(["🎙️ Ask by Voice (Whisper STT)", "⌨️ Ask by Typing"])

# --- TAB 1: ASK BY VOICE ---
with tab_voice:
    st.info("🎙️ Speak your question in conversational Marathi/Hindi/English (e.g., *'bhai CS branch cha cut off kiti ahe?'*). Click below to record.")
    
    audio_val = st.audio_input("Record your voice question:")
    
    if audio_val is not None:
        audio_bytes_raw = audio_val.getvalue()
        if audio_bytes_raw != st.session_state.get("last_recorded_audio"):
            st.session_state["last_recorded_audio"] = audio_bytes_raw
            with st.spinner("🎧 Transcribing spoken audio with OpenAI Whisper..."):
                temp_audio_path = "temp_voice_input.wav"
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_bytes_raw)
                
                try:
                    transcription = transcribe_audio(temp_audio_path)
                    if transcription:
                        st.session_state["transcribed_draft"] = transcription
                    else:
                        st.warning("⚠️ Speech not detected. Please speak louder or type below.")
                except Exception as e:
                    st.warning("⚠️ Transcription error. Please type your query below.")
                finally:
                    if os.path.exists(temp_audio_path):
                        try:
                            os.remove(temp_audio_path)
                        except Exception:
                            pass

    # Human-in-the-Loop Review
    if st.session_state.get("transcribed_draft"):
        st.markdown("""
        <div class="verify-card">
            <strong style="color: #92400E;">📝 Human-in-the-Loop Review (Transcribed Question):</strong>
            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #78350F;">
                Verify or edit the transcribed speech before passing it to the RAG & MemAC inference engine:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        edited_text = st.text_input(
            "Transcribed query:",
            value=st.session_state["transcribed_draft"],
            key="voice_draft_input"
        )
        
        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            if st.button("✅ Confirm & Run RAG Inference", type="primary", use_container_width=True):
                if edited_text.strip():
                    process_and_add_query(edited_text, engine_selection)
                    st.session_state["transcribed_draft"] = None
                    st.session_state["last_recorded_audio"] = None
                    st.rerun()
                else:
                    st.warning("Query cannot be empty.")
                    
        with btn_c2:
            if st.button("🔄 Discard & Record Again", use_container_width=True):
                st.session_state["transcribed_draft"] = None
                st.session_state["last_recorded_audio"] = None
                st.rerun()

# --- TAB 2: ASK BY TYPING ---
with tab_type:
    st.markdown("Type your question in Romanized Marathi, Hindi, or English:")
    with st.form("typed_form", clear_on_submit=True):
        typed_query = st.text_input(
            "Question:",
            placeholder="e.g., first year CS admission cutoff kiti ahe and fee installment madhe pay karta yeil ka?",
            label_visibility="collapsed"
        )
        submit_btn = st.form_submit_button("🚀 Submit Question to RAG Engine", type="primary", use_container_width=True)
        
        if submit_btn:
            if typed_query.strip():
                process_and_add_query(typed_query, engine_selection)
                st.rerun()
            else:
                st.warning("Please type a valid question.")

# ==============================================================================
# RUNNING CONVERSATION HISTORY & DEEP NLP INSPECTION
# ==============================================================================
st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
st.markdown("### 💬 Conversational History & Institutional Answers")

if not st.session_state["chat_history"]:
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; background: #FFFFFF; border: 1px dashed #D1D5DB; border-radius: 14px; color: #6B7280;">
        <span style="font-size: 2.2rem;">🎙️</span>
        <p style="margin-top: 8px; font-weight: 600; font-size: 1.05rem;">No questions asked yet.</p>
        <p style="font-size: 0.9rem; color: #9CA3AF;">Use the microphone, type a query, or tap any quick sample chip above to start the RAG conversation!</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for idx, item in enumerate(st.session_state["chat_history"]):
        nlp_res = item.get("nlp_result", {})
        urgency = nlp_res.get("urgency_info", {"level": "STANDARD / ROUTINE", "color": "#059669"})

        # 1. User Message Card
        st.markdown(f"""
        <div class="user-bubble">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="font-size: 0.76rem; color: #EA580C; font-weight: 700; text-transform: uppercase;">
                    Student Query #{len(st.session_state['chat_history']) - idx} • {item['timestamp']}
                </span>
                <span class="urgency-pill" style="background:{urgency['color']}15; color:{urgency['color']}; border:1px solid {urgency['color']}50;">
                    🚨 Priority: {urgency['level']}
                </span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #1F2937;">
                "{item['query']}"
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Grounded Assistant Response Card
        st.markdown(f"""
        <div class="assistant-bubble">
            <div style="margin-bottom: 8px;">
                <span class="intent-pill">🎯 Intent: {item['display_name']}</span>
                <span class="rag-pill">📚 Source: {item.get('citation', 'Academic Policy')}</span>
                <span class="conf-pill">Hybrid Confidence: {item['confidence']*100:.1f}%</span>
            </div>
            <div class="answer-marathi">
                {item['answer']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Spoken Audio Response Player
        st.caption(f"🔊 Listen to response in natural Marathi (TTS Voice Audio):")
        st.audio(item["audio_bytes"], format="audio/mp3")

        # 3. ADVANCED NLP & RAG DEEP INSPECTOR
        with st.expander("🔬 Inspect Deep NLP & RAG Pipeline Breakdown (For Viva / Demo)", expanded=False):
            col_rag, col_mem = st.columns(2)
            
            with col_rag:
                st.markdown("#### 📚 Multi-Document RAG Retrieval")
                retrieved = nlp_res.get("retrieved_chunks", [])
                for r_idx, chunk in enumerate(retrieved):
                    st.markdown(f"""
                    <div class="inspect-card">
                        <strong>#{r_idx+1}: {chunk['title']}</strong><br/>
                        <span style="font-size:0.78rem; color:#6B7280;">Department: {chunk['category']} | Match: <b>{chunk['similarity_percentage']}%</b></span>
                        <p style="margin:4px 0 0 0; font-size:0.84rem; color:#374151;"><i>"{chunk['content_english']}"</i></p>
                        <span style="font-size:0.75rem; color:#9333EA;">📌 Citation: {chunk['citation']}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with col_mem:
                st.markdown("#### 🧠 MemAC Working Memory State")
                mem_snap = nlp_res.get("working_memory", {})
                st.markdown(f"""
                <div class="inspect-card">
                    <b>Student Branch:</b> {mem_snap.get('STUDENT_BRANCH', 'N/A')}<br/>
                    <b>Academic Year:</b> {mem_snap.get('ACADEMIC_YEAR', 'N/A')}<br/>
                    <b>Admission Quota:</b> {mem_snap.get('ADMISSION_CATEGORY', 'N/A')}<br/>
                    <b>Pending Document:</b> {mem_snap.get('PENDING_DOCUMENT', 'None')}<br/>
                    <b>Payment Preference:</b> {mem_snap.get('PAYMENT_PREFERENCE', 'Standard')}<br/>
                    <b>Last Active Topic:</b> {mem_snap.get('LAST_ACTIVE_TOPIC', 'General')}<br/>
                    <b>Session Turns Logged:</b> {nlp_res.get('episodic_turns', 1)}
                </div>
                """, unsafe_allow_html=True)

            col_ner, col_cmi = st.columns(2)
            
            with col_ner:
                st.markdown("#### 🏷️ Token-Level BIO Sequence Tagging (NER)")
                iob_tags = nlp_res.get("iob_tags", [])
                tokens_html = []
                for tok, tag in iob_tags:
                    if tag.startswith("B-"):
                        cls_name = "bio-B"
                    elif tag.startswith("I-"):
                        cls_name = "bio-I"
                    else:
                        cls_name = "bio-O"
                    tokens_html.append(f"<span class='bio-token {cls_name}'>{tok} <sub>[{tag}]</sub></span>")
                st.markdown(f"<div class='inspect-card'>{' '.join(tokens_html)}</div>", unsafe_allow_html=True)

                slots = nlp_res.get("slots", {})
                if slots:
                    st.markdown("**Extracted Named Slots:**")
                    for k, v in slots.items():
                        st.markdown(f"- **{k}:** `{', '.join(v)}`")
                else:
                    st.caption("No specific domain named slots extracted.")

            with col_cmi:
                st.markdown("#### 🌐 Code-Switching Index (CMI) Profiling")
                cmi_data = nlp_res.get("code_switch_metrics", {})
                st.markdown(f"""
                <div class="inspect-card">
                    • <b>Total Words:</b> {cmi_data.get('total_words', 0)}<br/>
                    • <b>Marathi/Hindi Words:</b> {cmi_data.get('indic_words', 0)} ({cmi_data.get('indic_ratio', 0)}%)<br/>
                    • <b>English Words:</b> {cmi_data.get('english_words', 0)} ({cmi_data.get('english_ratio', 0)}%)<br/>
                    • <b>Code-Mixing Index (CMI):</b> <b>{cmi_data.get('cmi_score', 0.0)}</b><br/>
                    • <b>Linguistic Profile:</b> <span style="color:#C2410C; font-weight:600;">{cmi_data.get('dominant_language', 'English')}</span><br/>
                    • <b>Classification:</b> {cmi_data.get('classification', 'Single Script')}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr style='margin: 24px 0; border: none; border-top: 1px solid #F3F4F6;' />", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #9CA3AF; font-size: 0.84rem; margin-top: 40px; padding-bottom: 24px;">
    Autonomous College AI Help Desk • Dense Vector RAG • MemAC-LLM Conversational State • Multilingual Speech Engine
</div>
""", unsafe_allow_html=True)
