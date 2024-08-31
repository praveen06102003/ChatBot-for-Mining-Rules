# ChatBot for Mining Rules

A chatbot designed to answer questions based on the Mining Rules document. It leverages PDF document processing, FAISS for vector search, and Groq's large language model (LLM) for generating accurate responses.

## Description

This project implements a Streamlit-based chatbot that interacts with a Mining Rules PDF. The chatbot uses FAISS for document retrieval and integrates Groq's LLM to generate context-aware answers.

## Installation

To set up and run the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/chatbot-for-mining-rules.git
2. **Navigate to the project directory:**
   ```bash
   cd chatbot-for-mining-rules
3. **Create a virtual environment (optional but recommended) & Install the required dependencies & Run the application**
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate.
  pip install -r requirements.txt
  streamlit run app.py



