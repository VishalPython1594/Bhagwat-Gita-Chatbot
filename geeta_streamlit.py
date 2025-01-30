import streamlit as st
import google.generativeai as genai
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # You can change this to Google embeddings
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract the content of Bhagavad Gita from PDF
pdf_path = "Geeta.pdf"
geeta_text = extract_text_from_pdf(pdf_path)

# Create embeddings for the content (using OpenAI embeddings)
embeddings = HuggingFaceEmbeddings()
vector_db = FAISS.from_texts([geeta_text], embeddings)

# Streamlit UI
st.title("📖 Bhagavad Gita Chatbot")
st.write("Ask any question related to Bhagavad Gita!")

user_input = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_input:
        # Retrieve relevant documents using FAISS
        retrieved_docs = vector_db.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in retrieved_docs])  # Concatenate docs
        st.write("### Context:")
        st.write(context)
    else:
        st.warning("Please enter a question.")