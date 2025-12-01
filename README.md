# 🚀 RAG + Fine-Tuning on State-of-the-Art LLM Research Papers

This project implements a Retrieval-Augmented Generation (RAG) pipeline combined with a fine-tuned language model specialized in modern large language model (LLM) research.

The system is trained and evaluated on a curated corpus of high-impact AI papers, including:

Attention Is All You Need (Transformers)

GPT-3 technical report (massive autoregressive models)

InstructGPT & Constitutional AI (RLHF & alignment)

Scaling Laws & Chinchilla (compute/data efficiency)

Mixture-of-Experts architectures (Switch Transformers, GShard)

Llama 2 / PaLM / contemporary LLM architecture reports

# 🧠 Purpose

Modern LLM research is large, technical, and rapidly evolving. This project builds a domain-specialized AI research assistant capable of:

answering deep technical questions

retrieving precise passages from complex papers

synthesizing concepts across multiple research directions

explaining advanced architectures (MoE, RLHF, attention, scaling laws)

supporting model-training and architecture-design workflows

# 🛠️ Architecture

Embeddings: OpenAI text-embedding-3-*

Vector Store: Chroma (persisted locally)

LLM: Local Mistral via Ollama

UI: Lightweight, transparent Streamlit interface

Indexing: Section-aware, semantically chunked academic PDFs

Fine-Tuning: Instruction-style dataset generated over the research corpus

# 🎯 Why This Project Matters

This system demonstrates the ability to combine:

advanced NLP architectures

retrieval systems

domain-specific fine-tuning

modern research comprehension

full-stack ML engineering (indexing, storage, UI, evaluation)
