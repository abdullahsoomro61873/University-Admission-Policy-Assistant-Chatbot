# University Admission Policy Assistant Chatbot

## Introduction
The **University Admission Policy Assistant Chatbot** is a domain-specific virtual assistant designed to answer frequently asked admission questions **24/7**. Unlike generic chatbots, this system focuses exclusively on university admission information to ensure reliability and precision.

## Problem Statement
University policy documents are often lengthy and difficult for students to search manually. Students frequently ask repetitive questions, and traditional chatbots either provide generic answers or hallucinate information.

## Solution
This project implements a **Retrieval-Augmented Generation (RAG)** system that:
1. Retrieves relevant policy sections first.
2. Generates answers using a **Small Language Model (SLM)**.

**Key Benefits:**
* ✅ Answers are fact-based
* ✅ No guessing or hallucination
* ✅ Fast and reliable responses

## Dataset

The dataset `policy.txt` contains the official admission policies of Mohammad Ali Jinnah University (MAJU), Karachi, in text format. It serves as the knowledge base for the University Admission Policy Assistant Chatbot.

Contents include:
- General Admission Information
- Programs and Departments
- Fee Structure
- Scholarships and Financial Aid
- Important Deadlines

Purpose: This dataset is used to build a Retrieval-Augmented Generation (RAG) system, enabling the chatbot to provide accurate, fact-based answers to student queries about admission at MAJU.

## Methodology
1. Collect official admission policy data
2. Preprocess and chunk the data
3. Convert text into embeddings
4. Store embeddings in **FAISS**
5. Retrieve relevant context
6. Generate response using **phi3:mini SLM**
7. Stream output to user interface

## Technology Stack
* **Backend:** Python & Flask
* **RAG Framework:** LangChain
* **Vector Search:** FAISS
* **Language Model:** Ollama (phi3:mini)
* **Frontend:** HTML, CSS, JavaScript

## Summary
The University Admission Policy Assistant Chatbot is a **real-world NLP project** that leverages RAG to provide **accurate, reliable, and grounded answers** to admission-related queries. The system retrieves policy information from a FAISS vector database and uses a locally hosted SLM (phi3:mini) via Ollama to generate responses. The web interface built with Flask ensures an interactive and deployable application.
