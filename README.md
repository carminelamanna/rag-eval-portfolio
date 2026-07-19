# RAG Eval Portfolio

Sistema RAG (Retrieval-Augmented Generation) costruito da zero, con eval
harness dedicato — la versione "portfolio dimostrabile" del lavoro che
faccio in produzione come AI Engineer freelance.

## Obiettivo

Dimostrare in modo concreto e documentato la capacità di progettare,
costruire e valutare un sistema RAG end-to-end, dalle scelte
architetturali (chunking, retrieval, reranking) alla misurazione
della qualità delle risposte (LLM-as-judge).

## Dataset

Normativa pubblica italiana (es. testi di legge, regolamenti).

## Architettura

ingestion → chunking → embedding → storage (vector DB) →
retrieval (+ reranking) → generazione risposta → eval (incl.
LLM-as-judge) → instrumentazione costi/latenza/token

## Stack

Python, embedding via OpenAI API o `sentence-transformers`,
`chromadb`/`faiss-cpu` per il vector store, `pandas` per l'eval harness.
Deploy con Docker + CI minima.

## Stato del progetto

🚧 In costruzione — un componente a settimana. Vedi i commit per il
progresso.
