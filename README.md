
### AIShree Portfolio Chatbot
---
AIShree is a personalized, portfolio-driven AI chatbot built using Streamlit, LangChain, FAISS, and Groq's LLaMA-3 model. This application is designed to act as a smart assistant that answers questions strictly related to an individual's professional background, skills, experience, and achievements by referencing a structured portfolio file.

Features
---
🔹 Interactive Chatbot UI powered by Streamlit with custom styling.

🔹 Contextual Q&A using vector-based retrieval with FAISS and HuggingFace embeddings.

🔹 Natural Language Understanding with Groq's LLaMA-3-70B Versatile model.

🔹 Two-column layout for professional profile display and real-time chat interaction.

🔹 Strict relevance enforcement: Only answers portfolio-related queries; gracefully declines unrelated requests.

Tech Stack
---
Python

Streamlit

LangChain

HuggingFace Transformers

FAISS (Facebook AI Similarity Search)

Groq (LLaMA-3 model API)

How It Works
---
The user uploads a portfolio text file.

The file is split into chunks and embedded using HuggingFace sentence transformers.

The chunks are stored in a FAISS vector store for similarity-based retrieval.

Upon a user query, relevant portfolio snippets are retrieved and used to craft a context-aware prompt.

Groq's LLaMA-3 model generates an informative and concise response.

Use Case
---
Perfect for showcasing individual skills and experiences in a conversational manner, suitable for portfolio presentations, personal websites, or career-oriented chatbot applications.
