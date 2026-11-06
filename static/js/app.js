/**
 * College Help Desk Voice & Advanced NLP Assistant
 * Interactive Frontend JavaScript Engine
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatContainer = document.getElementById("chatContainer");
    const welcomeCard = document.getElementById("welcomeCard");
    const chatForm = document.getElementById("chatForm");
    const queryInput = document.getElementById("queryInput");
    const sendBtn = document.getElementById("sendBtn");
    const micBtn = document.getElementById("micBtn");
    const engineModeSelect = document.getElementById("engineModeSelect");
    const clearMemoryBtn = document.getElementById("clearMemoryBtn");
    const infoModalBtn = document.getElementById("infoModalBtn");
    const specsModal = document.getElementById("specsModal");
    const closeModalBtn = document.getElementById("closeModalBtn");
    const modalOkBtn = document.getElementById("modalOkBtn");
    const voiceBanner = document.getElementById("voiceRecordingBanner");
    const voiceStatusText = document.getElementById("voiceStatusText");
    const cancelVoiceBtn = document.getElementById("cancelVoiceBtn");
    const globalAudioPlayer = document.getElementById("globalAudioPlayer");

    // MemAC Sidebar Elements
    const memBranch = document.getElementById("memBranch");
    const memYear = document.getElementById("memYear");
    const memCategory = document.getElementById("memCategory");
    const memDocument = document.getElementById("memDocument");
    const memPayment = document.getElementById("memPayment");
    const memTopic = document.getElementById("memTopic");
    const memTurns = document.getElementById("memTurns");

    // Audio & Speech Recognition State
    let isRecording = false;
    let recognition = null;
    let mediaRecorder = null;
    let audioChunks = [];

    // Initialize Web Speech API if supported in browser
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = "en-IN"; // Handles Indian English / Hinglish well

        recognition.onstart = () => {
            isRecording = true;
            micBtn.classList.add("recording");
            voiceBanner.style.display = "flex";
            voiceStatusText.textContent = "Listening to your voice... Speak now in Marathi/English!";
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            queryInput.value = transcript;
            stopVoiceRecording();
            handleSendQuery(transcript);
        };

        recognition.onerror = (event) => {
            console.warn("SpeechRecognition error:", event.error);
            stopVoiceRecording();
        };

        recognition.onend = () => {
            stopVoiceRecording();
        };
    }

    // Toggle Voice Recording
    micBtn.addEventListener("click", () => {
        if (isRecording) {
            stopVoiceRecording();
        } else {
            startVoiceRecording();
        }
    });

    cancelVoiceBtn.addEventListener("click", () => {
        stopVoiceRecording();
    });

    function startVoiceRecording() {
        if (recognition) {
            try {
                recognition.start();
            } catch (e) {
                console.error("Recognition start error:", e);
                startMediaRecorderFallback();
            }
        } else {
            startMediaRecorderFallback();
        }
    }

    function stopVoiceRecording() {
        isRecording = false;
        micBtn.classList.remove("recording");
        voiceBanner.style.display = "none";
        if (recognition) {
            try { recognition.stop(); } catch (e) {}
        }
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
    }

    // Fallback MediaRecorder for audio upload
    async function startMediaRecorderFallback() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Microphone access is not supported in this browser.");
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
                stream.getTracks().forEach(track => track.stop());
                
                // Upload audio blob to backend
                voiceStatusText.textContent = "Transcribing speech on server...";
                const formData = new FormData();
                formData.append("audio", audioBlob, "voice.webm");

                try {
                    const res = await fetch("/api/voice_upload", { method: "POST", body: formData });
                    const data = await res.json();
                    if (data.transcribed_text) {
                        queryInput.value = data.transcribed_text;
                        handleSendQuery(data.transcribed_text);
                    } else {
                        alert("Could not detect clear speech. Please try speaking louder.");
                    }
                } catch (err) {
                    console.error("Voice upload error:", err);
                } finally {
                    stopVoiceRecording();
                }
            };

            mediaRecorder.start();
            isRecording = true;
            micBtn.classList.add("recording");
            voiceBanner.style.display = "flex";
            voiceStatusText.textContent = "Recording microphone... (Speak for 4-5 seconds)";
            
            // Auto stop after 5s
            setTimeout(() => {
                if (isRecording) stopVoiceRecording();
            }, 5000);

        } catch (err) {
            console.error("Microphone permission denied:", err);
            alert("Microphone access was denied or is unavailable.");
            stopVoiceRecording();
        }
    }

    // Form Submit Handler
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = queryInput.value.trim();
        if (text) {
            handleSendQuery(text);
        }
    });

    // Auto-resize textarea and submit on Enter (without Shift)
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Suggestion Chips Click Handlers
    document.addEventListener("click", (e) => {
        const chip = e.target.closest(".chip-btn, .pill");
        if (chip && chip.dataset.query) {
            const q = chip.dataset.query;
            queryInput.value = q;
            handleSendQuery(q);
        }
    });

    // Clear Memory Handler
    clearMemoryBtn.addEventListener("click", async () => {
        if (confirm("Reset MemAC Conversational Working Memory and entity context?")) {
            try {
                const res = await fetch("/api/memory/clear", { method: "POST" });
                const data = await res.json();
                updateMemACSidebar(data.working_memory, 0);
                
                // Add system notification in chat
                appendSystemMessage("🔄 Conversational working memory has been reset to defaults.");
            } catch (err) {
                console.error("Error clearing memory:", err);
            }
        }
    });

    // Modal Handlers
    infoModalBtn.addEventListener("click", () => { specsModal.style.display = "flex"; });
    closeModalBtn.addEventListener("click", () => { specsModal.style.display = "none"; });
    modalOkBtn.addEventListener("click", () => { specsModal.style.display = "none"; });
    specsModal.addEventListener("click", (e) => {
        if (e.target === specsModal) specsModal.style.display = "none";
    });

    // Handle Sending Query to Backend
    async function handleSendQuery(query) {
        if (!query.trim()) return;

        // Hide welcome card if still visible
        if (welcomeCard) {
            welcomeCard.style.display = "none";
        }

        // 1. Render User Message
        appendUserMessage(query);
        queryInput.value = "";

        // 2. Render Bot Typing Indicator
        const typingId = appendTypingIndicator();
        scrollToBottom();

        const mode = engineModeSelect.value;

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query, mode: mode })
            });

            const data = await response.json();
            removeTypingIndicator(typingId);

            if (data.error) {
                appendErrorMessage(data.error);
            } else {
                appendBotResponse(data);
                updateMemACSidebar(data.working_memory, data.episodic_turns);
            }
        } catch (err) {
            removeTypingIndicator(typingId);
            appendErrorMessage("Network error: Could not connect to the College Help Desk backend server.");
            console.error(err);
        }

        scrollToBottom();
    }

    // Append User Message to Stream
    function appendUserMessage(text) {
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const row = document.createElement("div");
        row.className = "message-row user-row";
        row.innerHTML = `
            <div class="avatar user-avatar">
                <i class="fa-solid fa-user"></i>
            </div>
            <div class="message-bubble">
                <div class="user-text">${escapeHtml(text)}</div>
                <div class="msg-meta">${timeStr}</div>
            </div>
        `;
        chatContainer.appendChild(row);
    }

    // Append Typing Indicator
    function appendTypingIndicator() {
        const id = "typing-" + Date.now();
        const row = document.createElement("div");
        row.id = id;
        row.className = "message-row bot-row";
        row.innerHTML = `
            <div class="avatar bot-avatar">
                <i class="fa-solid fa-graduation-cap"></i>
            </div>
            <div class="message-bubble typing-bubble">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        `;
        chatContainer.appendChild(row);
        return id;
    }

    function removeTypingIndicator(id) {
        const elem = document.getElementById(id);
        if (elem) elem.remove();
    }

    // Append Bot Structured NLP Response
    function appendBotResponse(data) {
        const row = document.createElement("div");
        row.className = "message-row bot-row";

        const uniqueId = "nlp-exp-" + Date.now();
        const urgency = data.urgency_info || { level: "STANDARD / ROUTINE", color: "#10B981" };
        const codeSwitch = data.code_switch || { cmi_score: 0, indic_ratio: 0, english_ratio: 0 };
        const slots = data.slots || {};
        const iobTags = data.iob_tags || [];
        const chunks = data.retrieved_chunks || [];

        // Build BIO Slot HTML tokens
        let bioTokensHtml = "";
        if (iobTags.length > 0) {
            bioTokensHtml = iobTags.map(([tok, tag]) => {
                const isSlot = tag !== "O";
                return `<span class="bio-token ${isSlot ? 'has-slot' : ''}">${escapeHtml(tok)}<span class="token-tag">${tag}</span></span>`;
            }).join(" ");
        } else {
            bioTokensHtml = `<span style="font-size: 12px; color: #94A3B8;">No named entity slots detected.</span>`;
        }

        // Build RAG Chunks HTML
        let ragChunksHtml = "";
        if (chunks.length > 0) {
            ragChunksHtml = chunks.map((c, i) => `
                <div class="rag-chunk-card">
                    <div class="rag-chunk-header">
                        <span>#${i+1} ${escapeHtml(c.title || c.id)}</span>
                        <span class="rag-score-badge"><i class="fa-solid fa-bullseye"></i> ${c.similarity_percentage}% match</span>
                    </div>
                    <div style="color: #475569; font-size: 11.5px;">${escapeHtml(c.citation || '')}</div>
                </div>
            `).join("");
        }

        // Format Marathi text (convert line breaks and markdown bold)
        let formattedMarathi = escapeHtml(data.marathi_answer)
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');

        row.innerHTML = `
            <div class="avatar bot-avatar">
                <i class="fa-solid fa-graduation-cap"></i>
            </div>
            <div class="message-bubble">
                
                <!-- Header Meta -->
                <div class="bot-header-meta">
                    <div class="meta-left">
                        <span class="intent-pill">
                            <i class="fa-solid fa-tag"></i> ${escapeHtml(data.display_name)}
                        </span>
                        <span class="confidence-pill" title="Hybrid SVM + Semantic Cosine Confidence">
                            ${data.hybrid_confidence}% Match
                        </span>
                    </div>
                    <span class="urgency-pill" style="background: ${urgency.color}18; color: ${urgency.color};">
                        <i class="fa-solid fa-bolt"></i> ${urgency.level}
                    </span>
                </div>

                <!-- Grounded Marathi Spoken Answer -->
                <div class="marathi-response-box">
                    <div class="marathi-label">
                        <i class="fa-solid fa-bullhorn"></i> अधिकृत महाविद्यालयीन माहिती (Official Answer):
                    </div>
                    <div>${formattedMarathi}</div>
                </div>

                <!-- English Contextual Summary (if available) -->
                ${data.english_answer ? `
                <div class="english-summary-box">
                    <strong><i class="fa-solid fa-language"></i> English Context:</strong> ${escapeHtml(data.english_answer)}
                </div>
                ` : ''}

                <!-- Audio Player & Source Citation Bar -->
                <div class="bot-audio-action">
                    ${data.audio_url ? `
                    <button class="play-voice-btn" onclick="playMarathiAudio('${data.audio_url}', this)">
                        <i class="fa-solid fa-volume-high"></i> Listen (मराठी आवाज)
                    </button>
                    ` : `<span></span>`}
                    <div class="citation-tag">
                        <i class="fa-solid fa-bookmark"></i> <em>${escapeHtml(data.citation)}</em>
                    </div>
                </div>

                <!-- Deep NLP Inspector Accordion -->
                <button class="nlp-inspector-toggle" onclick="toggleInspector('${uniqueId}', this)">
                    <span><i class="fa-solid fa-microscope"></i> Deep NLP Inspector & RAG Retrieval</span>
                    <i class="fa-solid fa-chevron-down toggle-arrow"></i>
                </button>

                <div class="nlp-inspector-content" id="${uniqueId}" style="display: none;">
                    
                    <!-- 1. Slot Filling & BIO Tagging -->
                    <div class="inspector-section">
                        <h4><i class="fa-solid fa-tags"></i> 1. Token-Level BIO Named Entity Recognition (NER):</h4>
                        <div class="bio-token-row">${bioTokensHtml}</div>
                    </div>

                    <!-- 2. Code-Switching Index (CMI) -->
                    <div class="inspector-section">
                        <h4><i class="fa-solid fa-globe"></i> 2. Code-Switching Index (CMI) & Language Mixing:</h4>
                        <div class="cmi-meter-wrap">
                            <div class="cmi-progress">
                                <div class="cmi-bar" style="width: ${Math.min(100, codeSwitch.cmi_score * 2.5)}%;"></div>
                            </div>
                            <div class="cmi-labels">
                                <span>CMI Score: <strong>${codeSwitch.cmi_score}</strong> (${codeSwitch.classification || 'Mixed'})</span>
                                <span>Indic: <strong>${codeSwitch.indic_ratio}%</strong> | English: <strong>${codeSwitch.english_ratio}%</strong></span>
                            </div>
                        </div>
                    </div>

                    <!-- 3. Dense RAG Chunks -->
                    ${chunks.length > 0 ? `
                    <div class="inspector-section">
                        <h4><i class="fa-solid fa-database"></i> 3. Multi-Document RAG Semantic Matches:</h4>
                        <div>${ragChunksHtml}</div>
                    </div>
                    ` : ''}

                </div>

            </div>
        `;

        chatContainer.appendChild(row);
    }

    // Play Marathi Speech Audio
    window.playMarathiAudio = function(audioUrl, btnElement) {
        if (!audioUrl) return;
        globalAudioPlayer.src = audioUrl;
        globalAudioPlayer.play();
        
        const originalHtml = btnElement.innerHTML;
        btnElement.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Playing...`;
        btnElement.disabled = true;

        globalAudioPlayer.onended = () => {
            btnElement.innerHTML = originalHtml;
            btnElement.disabled = false;
        };

        globalAudioPlayer.onerror = () => {
            btnElement.innerHTML = originalHtml;
            btnElement.disabled = false;
        };
    };

    // Toggle Deep NLP Inspector
    window.toggleInspector = function(id, btnElement) {
        const content = document.getElementById(id);
        const arrow = btnElement.querySelector(".toggle-arrow");
        if (content.style.display === "none" || !content.style.display) {
            content.style.display = "flex";
            arrow.style.transform = "rotate(180deg)";
        } else {
            content.style.display = "none";
            arrow.style.transform = "rotate(0deg)";
        }
    };

    // Update Right Sidebar MemAC Working Memory
    function updateMemACSidebar(memory, turns) {
        if (!memory) return;
        if (memBranch) memBranch.textContent = memory.STUDENT_BRANCH || "Not Specified";
        if (memYear) memYear.textContent = memory.ACADEMIC_YEAR || "Not Specified";
        if (memCategory) memCategory.textContent = memory.ADMISSION_CATEGORY || "Open / General";
        if (memDocument) memDocument.textContent = memory.PENDING_DOCUMENT || "None";
        if (memPayment) memPayment.textContent = memory.PAYMENT_PREFERENCE || "Standard";
        if (memTopic) memTopic.textContent = memory.LAST_ACTIVE_TOPIC || "General Information";
        if (memTurns && turns !== undefined) memTurns.textContent = turns;
    }

    function appendSystemMessage(msg) {
        const div = document.createElement("div");
        div.style.textAlign = "center";
        div.style.fontSize = "12px";
        div.style.color = "#64748B";
        div.style.padding = "6px 12px";
        div.style.background = "#F1F5F9";
        div.style.borderRadius = "20px";
        div.style.margin = "8px auto";
        div.style.width = "fit-content";
        div.textContent = msg;
        chatContainer.appendChild(div);
    }

    function appendErrorMessage(msg) {
        const div = document.createElement("div");
        div.style.textAlign = "center";
        div.style.fontSize = "13px";
        div.style.color = "#DC2626";
        div.style.padding = "8px 14px";
        div.style.background = "#FEE2E2";
        div.style.borderRadius = "8px";
        div.style.margin = "8px 0";
        div.textContent = `⚠️ ${msg}`;
        chatContainer.appendChild(div);
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
