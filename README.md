# 🎓 College Help Desk - Voice & Advanced NLP Assistant

An advanced, multilingual, voice-enabled college help desk assistant designed for bilingual and code-mixed student queries in **Marathi and Hindi written in Roman script, mixed naturally with English** (*Maranglish / Hinglish*). 

The system leverages **6 Advanced NLP & Conversational AI Pillars**:
1. **Multi-Document Dense Vector RAG (Retrieval-Augmented Generation)** over an institutional knowledge corpus with sublinear vector space cosine retrieval.
2. **MemAC-LLM (Memory-Augmented Conversational State Engine)** maintaining working entity memory (Branch, Academic Year, Quota, Documents) and episodic session history for multi-turn follow-up queries.
3. **Joint Intent Classification & Token-Level Sequence Tagging** (IOB / BIO Named Entity Recognition & Slot Filling).
4. **Multilingual Linguistic Profiling & Code-Switching Index (CMI / CSI)** for evaluating intra-sentential language mixing.
5. **Query Urgency & Administrative Sentiment Triage** for helpdesk ticket prioritization.
6. **OpenAI Whisper ASR & gTTS Speech Synthesis** providing natural spoken Marathi audio responses.

---

## 🌟 Advanced NLP Architecture Breakdown

```text
                                 [ Student Spoken / Typed Input ]
                                                │
                                    (Whisper STT / Text Input)
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 │                                                             │
                 ▼                                                             ▼
       [ 🏷️ Sequence Slot Filler ]                               [ 🌐 Code-Switching Index ]
    BIO Sequence Tagging (B-BRANCH, etc.)                      CMI Score & Indic/English Ratio
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │
                                                ▼
                                   [ 🧠 MemAC Memory Engine ]
                           Working Entity Slots + Episodic History
                                (Contextual Query Enrichment)
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 │                                                             │
                 ▼                                                             ▼
     [ ⚡ Multi-Document RAG ]                                  [ 🎯 Intent Classifier ]
 Dense Cosine Semantic Retrieval                               Sublinear TF-IDF + Linear SVM
(30+ Institutional Corpus Chunks)                               (13,500 Trained Samples)
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │
                                                ▼
                               [ 💡 Generative RAG Synthesizer ]
                        Dynamic Slot Injection + Institutional Citation
                                                │
                                                ▼
                         [ 🔊 Natural Marathi Voice Output (gTTS) ]
```

---

## 🔬 6 Advanced NLP Concepts Explained (For Project Viva / Presentation)

| Pillar | Advanced NLP Concept | Technical Implementation | Purpose in Project |
| :--- | :--- | :--- | :--- |
| **1** | **Dense Vector RAG (Retrieval-Augmented Generation)** | Multi-document chunking, Sublinear TF-IDF / Neural dense cosine similarity retrieval, dynamic context assembly | Eliminates hallucinations; retrieves exact institutional rules, fees, and procedures with citations. |
| **2** | **MemAC Conversational Memory** | Stateful working memory for entity slots + episodic memory buffer with anaphora resolution | Remembers student profile (Branch, Year, Quota) across turns to answer follow-up queries. |
| **3** | **Joint Sequence Tagging (NER / Slot Filling)** | Token-level IOB / BIO sequence classification (`B-BRANCH`, `B-CATEGORY`, `B-DOCUMENT`, `B-PAYMENT_MODE`) | Extracts critical parameters from informal student slang for personalized responses. |
| **4** | **Linguistic Code-Switching Index ($CMI$)** | Morpho-lexical token categorization ($CMI = 100 \times [1 - \frac{\max(w_1, w_2)}{N}]$) | Measures the degree of Marathi/Hindi/English mixing in Roman script. |
| **5** | **Administrative Urgency Triage** | Multi-class sentiment & emergency keyword classification | Identifies urgent student queries (e.g. payment failure, lost hall ticket, deadline today). |
| **6** | **Hybrid Confidence Scoring** | Reciprocal Rank / Linear Classifier probability + Dense Vector Similarity fusion | Produces calibrated confidence scores for student queries. |

---

## 📂 Project Structure

```text
college_help_desk/
│
├── advanced_nlp.py             # 🧠 Advanced RAG, MemAC Memory, NER & CMI Engine
├── app.py                      # 💻 Backend engine & interactive CLI application
├── streamlit_app.py            # 🌐 Streamlit web app with Deep NLP Inspector
├── generate_dataset.py         # 📊 Combinatorial dataset generator (13,500 rows)
├── train_model.py              # 🏋️ ML training, cross-validation & evaluation script
├── results_summary.md          # 📈 Comprehensive NLP model benchmark report
├── requirements.txt            # 📦 Pinned project dependencies
├── README.md                   # 📖 Complete documentation and setup guide
│
├── data/
│   └── college_faq_dataset.csv # 13,500 balanced student queries across 10 intents
│
├── models/
│   ├── intent_classifier.pkl   # Serialized Linear SVM classifier
│   ├── tfidf_vectorizer.pkl    # Serialized TF-IDF vectorizer (12k features)
│   ├── confusion_matrix.png    # Primary confusion matrix plot
│   └── models_comparison_cm.png# Comparative confusion matrix (LR vs Linear SVM)
│
└── .streamlit/
    └── config.toml             # Streamlit theme configuration (warm palette)
```

---

## 🎯 10 Supported Institutional Domains

1. **Admissions & Seats (`admission_query`):** CAP rounds, freeze/float, cutoffs by branch, TFWS quota, spot rounds.
2. **Fees & Accounts (`fee_query`):** Category fees, installment permissions, ERP online transaction issues.
3. **Timetable & Classes (`timetable_query`):** Lecture schedules, practical lab batches, academic calendar PDF.
4. **Scholarships & Freeships (`scholarship_query`):** MahaDBT post-matric, EBC, income limits, biometric desk.
5. **Student Documents (`document_query`):** Bonafide certificates, LC/TC, transcripts, loan quotations.
6. **Administration Helpdesk (`contact_query`):** Central helpline, department emails, Principal meeting hours.
7. **Campus Hostel (`hostel_query`):** Boys/girls hostel allotment, mess food, night curfew (9:00 PM).
8. **Examinations (`exam_query`):** University hall ticket download, KT backlog forms, revaluation photocopies.
9. **Central Library (`library_query`):** Timings (8 AM - 9 PM), book issue limits, IEEE digital library.
10. **Placements & TPO (`placement_query`):** Campus drives, top recruiters (TCS, NVIDIA, etc.), salary packages.

---

## 🚀 How to Run the Project

### 1. Run the HTML / CSS / JavaScript Web Application (Modern Custom UI)

```bash
python web_app.py
```

- Opens in your browser at: **`http://localhost:5000`** or **`http://127.0.0.1:5000`**
- **Key Features:**
  - **Full Custom HTML5 & CSS3 Design System:** Sleek, responsive, modern glassmorphic dashboard.
  - **Interactive Voice & Text Chat:** One-click microphone recording (Web Speech API / server Whisper fallback) and instant Maranglish/Hinglish text queries.
  - **Live MemAC Working Memory Sidebar:** Real-time state cards tracking Student Branch, Academic Year, Quota, Documents, and session turns.
  - **Deep NLP Inspector:** Expandable visual breakdowns showing token-level BIO tags (`B-BRANCH`, etc.), Code-Switching Index (CMI), and RAG institutional document citations.
  - **Marathi Voice Speech Player:** Built-in audio playback using gTTS.

---

### 2. Run the Streamlit Web Application

```bash
streamlit run streamlit_app.py
```

- Opens in your browser at `http://localhost:8501`.

---

### 3. Run the Command-Line Interface (CLI)

```bash
python app.py
```

- Provides an interactive terminal menu:
  - `1. Ask by Voice` (Microphone recording + Whisper ASR)
  - `2. Ask by Typing` (Direct typed query)
  - `3. Exit`

---

## 🛠️ Customizing the Knowledge Base

To update institutional answers, dates, contact numbers, or fee figures:
1. Open [`advanced_nlp.py`](advanced_nlp.py) and customize the `INSTITUTIONAL_KNOWLEDGE_CORPUS` list.
2. Open [`app.py`](app.py) and update the `KNOWLEDGE_BASE` dictionary.
Both the web app and CLI will automatically reflect your customized college details.
