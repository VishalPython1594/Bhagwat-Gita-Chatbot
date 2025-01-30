import os
import streamlit as st
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# 📌 Load API Key Safely
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found. Make sure your .env file is set correctly.")

# 📌 Initialize NLTK resources
# nltk.download('stopwords')
# nltk.download('punkt')

# 📌 Preprocessing Function (Cached)
@st.cache_data
def preprocess_text(_data):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean(doc):
        doc = re.sub(r'https?://\S+|www\.\S+', '', doc)  # Remove URLs
        doc = re.sub(r'[^A-Za-z\s]', ' ', doc)  # Keep only letters and spaces
        doc = ''.join([char for char in doc if char not in string.punctuation])
        tokens = nltk.word_tokenize(doc.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    return [clean(doc.page_content) for doc in _data]

# 📌 Load PDF and Process Data (Cached)
@st.cache_data
def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    data_content = preprocess_text(data)
    
    # Convert to LangChain Document format
    from langchain.schema import Document
    docs = [Document(page_content=content, metadata=doc.metadata) for doc, content in zip(data, data_content)]
    
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=65)
    return text_splitter.split_documents(docs)

# 📌 Load or Create ChromaDB (Cached)
@st.cache_resource
def initialize_chroma_db(_chunks):
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(documents=chunks, embedding=hf_embeddings, persist_directory='chroma_db')
    return db

# 📌 Load Data & Initialize DB
chunks = load_and_process_pdf('Geeta.pdf')
db = initialize_chroma_db(chunks)

# 📌 Set up Retriever
retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': 5})

# 📌 Chat Prompt Template
prompt_template = '''
Answer the question based on the following context:
{context}

Answer the question based on the above context: {question}

Provide a detailed answer.
Do not justify your answers or add extra context.
Do not say "according to the context" or similar phrases.
'''

prompt_temp = ChatPromptTemplate.from_template(prompt_template)

# 📌 Chat Model
chat_model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model='gemini-1.5-flash')

# 📌 Output Parser
output_parser = StrOutputParser()

# 📌 Format Retrieved Documents
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# 📌 RAG Chain
chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt_temp
    | chat_model
    | output_parser
)

# 📌 Streamlit UI
st.title("📖 Bhagavad Gita Chatbot")
st.write("Ask any question related to Bhagavad Gita!")

user_input = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_input:
        response = chain.invoke(user_input)
        st.write("### Teaching:")
        st.write(response)
    else:
        st.warning("Please enter a question.")
