# low-latency-voice-rag-pipeline
This project builds a low-latency voice-based QA system using RAG and TTS. A filler audio response is played instantly while retrieval runs in parallel, ensuring the spoken response starts within the required limit of 800 ms latency.

## System architecture
readme_content = """
# 🔊 Zero-Latency Voice Knowledge Base (RAG + Voice AI)

## 📌 Overview
This project implements a low-latency voice-based Retrieval-Augmented Generation (RAG) system for a CCaaS Voice AI agent.
The system answers complex technical queries from a hardware manual and responds via speech while ensuring the
Time To First Byte (TTFB) of audio remains under 800 ms.

---

## 🧠 Problem Statement
The goal is to design a Voice AI agent that:
- Accepts spoken user queries
- Retrieves relevant information from a large technical manual
- Responds using speech (TTS)
- Maintains sub-800 ms audio latency for a natural user experience

---

## 🏗️ System Architecture
User Voice (.wav)  
→ ASR (Speech-to-Text)  
→ Parallel RAG Prefetch  
→ Hybrid Search (Vector + BM25)  
→ Cross-Encoder Reranking  
→ Voice-Optimized Answer  
→ TTS Audio Output  

---

## ✅ Task 1: Parallelized RAG Pipeline

### Goal
Reduce end-to-end latency by avoiding a strictly linear pipeline.

### Implementation
- Partial ASR transcripts are used to prefetch RAG results early
- Query rewriting resolves contextual references (e.g., “the second one”)
- Conversation history is incorporated before retrieval

### Notes
- ASR Model Used: **ASR Model Used: OpenAI Whisper**
- Vector Database: **FAISS (Facebook AI Similarity Search)**
- Query Rewrite Strategy: **Rule-based query rewriting**

## 🧠 Contextual Queries Handling
**“what about the second one?” are handled using lightweight, rule-based query rewriting. The system uses prior conversation context to resolve ambiguous references and rewrites the query before retrieval, ensuring that the results are accurate.**


---

## ✅ Task 2: Complex Querying & Reranking

### Goal
Improve retrieval accuracy for technical queries without increasing perceived latency.

### Implementation
- Hybrid retrieval using:
  - Dense vector embeddings
  - BM25 keyword-based search
- Cross-encoder reranker for final ranking
- A filler audio response is played immediately while reranking runs in parallel

### Notes
- Embedding Model: **Sentence-Transformers (all-MiniLM-L6-v2)**
- Reranker Model: **[ADD HERE]**
- Avg Reranker Latency: **[ADD HERE]**

---

## ✅ Task 3: Voice-Optimized Chunking & Speech Output

### Goal
Ensure responses sound natural when spoken.

### Implementation
- Removed PDF artifacts such as:
  - Bullet symbols
  - Page numbers
  - Headers and indexes
- Converted text into short, spoken-friendly sentences
- Generated final output using Text-to-Speech (TTS)

### Notes
- TTS Engine Used: **[ADD HERE]**
- Audio Format: **mp3 / wav**
- Avg Audio Duration: **[ADD HERE]**

---

## ⏱️ Latency Measurement (TTFB)

TTFB is measured as the time between receiving the user query and the start of audio playback.

```python

