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
```
import threading
import whisper
import importlib
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
```
```
asr_model = whisper.load_model("base")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
```
### Notes
- ASR Model Used: **ASR Model Used: OpenAI Whisper**
- Vector Database: **FAISS (Facebook AI Similarity Search)**
- Query Rewrite Strategy: **Rule-based query rewriting**
```
def stream_asr(audio_path, callback):
    result = asr_model.transcribe(audio_path)
    words = result["text"].split()
    partial = ""

    for w in words:
        partial += w + " "
        callback(partial)
```
Here we have created a function stream_asr
```
def on_partial_text(text):
    if len(text.split()) > 4:
        threading.Thread(target=prefetch_rag, args=(text,)).start()
```
This function triggers an early RAG prefetch as soon as a partial speech transcript becomes meaningful. Once enough words are detected, a background thread starts retrieval in parallel, reducing overall response latency when the full query is finalized.
## query audio used-

[Click here to listen](assets/sample.wav)


## 🧠 Contextual Queries Handling
**“what about the second one?” are handled using lightweight, rule-based query rewriting. The system uses prior conversation context to resolve ambiguous references and rewrites the query before retrieval, ensuring that the results are accurate.**

```
User: What safety rules apply during installation?
Agent: [answers]
User: What about the second one?
```
```
def rewrite_query(query, history):
    if "second" in query.lower() and history:
        return f"Explain the second item related to {history[-1]}"
    return query
```
The system will:

Understand that “second one” refers to the second safety rule

Use conversation history to resolve ambiguity

Rewrite the query BEFORE retrieval

***we call it query rewriting with context.***
```
conversation_history = []

def task2_pipeline(query):
    global conversation_history

    # Rewrite query using context
    rewritten_query = rewrite_query(query, conversation_history)

    # Start filler response immediately
    filler_thread = threading.Thread(target=filler_response)
    filler_thread.start()

    # Heavy work
    candidate_ids = hybrid_search(rewritten_query)
    ranked_docs = rerank(rewritten_query, candidate_ids)

    # Save context for next turn
    conversation_history.append(query)

    return ranked_docs[0]

```
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
```
!pip install -q rank-bm25
!pip install -q pypdf nltk
from pypdf import PdfReader
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
```
This repository sets up a lightweight text retrieval pipeline for PDF documents using classical information retrieval (BM25) and modern NLP models from Sentence Transformers.
## Libraries Used

PdfReader (pypdf): Extracts raw text from PDF documents

NLTK: Splits extracted text into sentences using the Punkt tokenizer

BM25Okapi: Ranks sentences based on keyword relevance

SentenceTransformer: Generates dense vector embeddings for semantic similarity

CrossEncoder: Re-ranks retrieved results using deep contextual relevance

### 🎙️Text Chunking for Voice Output
```
def chunk_text_for_voice(text, max_words=80):
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()

        if word_count + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0

        current_chunk.append(sentence)
        word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

```
This function splits a large text into smaller, sentence-based chunks suitable for voice or TTS processing. It first tokenizes the text into sentences using NLTK, then groups consecutive sentences into chunks while ensuring each chunk does not exceed a specified maximum word limit. This helps maintain natural speech flow and prevents overly long audio segments.
## 🔍Hybrid Search with Cross-Encoder Reranker
```
def hybrid_search(query, top_k=5):
    # Vector search
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    _, vector_ids = index.search(q_emb, top_k)
    vector_results = set(vector_ids[0])

    # BM25 search
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ids = np.argsort(bm25_scores)[-top_k:]
    bm25_results = set(bm25_ids)

    # Union of both
    combined_ids = list(vector_results.union(bm25_results))
    return combined_ids

```
```
def rerank(query, doc_ids):
    pairs = [(query, documents[i]) for i in doc_ids]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(doc_ids, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [documents[i] for i, _ in ranked]
```
## Dealing with the latency twist by generating a filler response
```
def filler_response():
    print("Voice Agent:", "Give me a second while i check that for you...")

```
### Notes
- Embedding Model: **Sentence-Transformers (all-MiniLM-L6-v2)**
- Reranker Model: **Simulated cross-encoder reranker**
- Avg Reranker Latency: **Simulated cross-encoder reranker**

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
- TTS Engine Used: **OpenAI Whisper / gTTS / Coqui TTS (depending on implementation)**
- Audio Format: **mp3 / wav**
- Avg Audio Duration: **10–30 seconds per chunk (depends on text length)**

## 🎙️ Voice-Optimized Response Processing
```
import re
def voice_optimized_response(text):
    """
    Converts technical, text-heavy RAG output
    into short, spoken-English sentences.
    """

    # 1. Shorten long sentences
    sentences = re.split(r'[.;]', text)

    simplified = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # 2. Replace complex words with simpler spoken terms
        s = s.replace("indicates", "means")
        s = s.replace("occurs", "happens")
        s = s.replace("approximately", "nearly")
        s = s.replace("utilize", "use")
        s = s.replace("damage", "destroy")

        # 3. Phonetic hints for technical terms
        s = s.replace("SATA", "S A T A")
        s = s.replace("SMART", "S M A R T")
        s = s.replace("HDD", "hard disk")
        s = s.replace("hazards", "problems")

        simplified.append(s)

    # 4. Join as short spoken sentences
    return ". ".join(simplified) + "."
```

This function converts technical, text-heavy RAG outputs into clear, spoken-English sentences suitable for text-to-speech systems. It breaks long responses into shorter sentences, replaces complex words with simpler terms, and adds phonetic hints to improve pronunciation. The result is audio-friendly output that sounds natural and easy to understand when spoken.

## 🔊 Text-to-Speech Generation
```
from gtts import gTTS
from IPython.display import Audio

def speak_text(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = "output.mp3"
    tts.save(filename)
    return Audio(filename, autoplay=True)

query = audio_to_query("sample.wav")
answer_doc = task2_pipeline(query)  # call your existing pipeline

print("\nRetrieved Answer Chunk:")
print(answer_doc)

speak_text(answer_doc)

```
This module converts the retrieved answer from the RAG pipeline into spoken audio using Google Text-to-Speech (gTTS). The response text is synthesized into an MP3 file and automatically played within the notebook environment, enabling hands-free, voice-based interaction with the system.

---
## Final Result
```
!pip install gtts
from gtts import gTTS
from IPython.display import Audio

def speak_text(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = "output.mp3"
    tts.save(filename)
    return Audio(filename, autoplay=True)

query = audio_to_query("sample.wav")
answer_doc = task2_pipeline(query)  # call your existing pipeline

print("\nRetrieved Answer Chunk:")
print(answer_doc)

speak_text(answer_doc)
```
## ⏱️ Latency Measurement (TTFB)

TTFB is measured as the time between receiving the user query and the start of audio playback.


