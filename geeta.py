# - Step1: Initialize the chroma db connection
# - Step2: Create a Retriever Object
# - Step3: Initialize a Chat Prompt Template
# - Step4: Initialize a Generator(Chat Model)
# - Step5: Initialize an Output Parser
# - Step6: Define a RAG Chain
# - Step7: Invoke the chain

### Before we initialize the chroma db connection, lets load the data first
import os
import streamlit as st
import tensorflow as tf
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('Geeta.pdf')
data = loader.load()
from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy # type: ignore


data_content = [doc.page_content for doc in data]

### After that we'll do cleaning, chunking, embedding

import re
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = list(set(stopwords.words('english')))
lemmatizer = WordNetLemmatizer()

def clean(doc):
    doc = re.sub(r'https?://\S+|www\.\S+', '', doc)  # Remove URLs
    doc = re.sub(r'[^A-Za-z\s]', ' ', doc)  # Keep only letters and spaces
    doc = ''.join([char for char in doc if char not in string.punctuation])
    tokens = nltk.word_tokenize(doc.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data_content = [clean(doc) for doc in data_content]

meta_data = [doc.metadata for doc in data]

data_fin = []
for i in range(len(meta_data)):
    data_fin.append({'metadata' : meta_data[i],
                'content' : data_content[i]})
    
from langchain.schema import Document
documents = [Document(page_content=item['content'], metadata=item['metadata']) for item in data_fin]

### Chunking

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(separators = ['\n\n', '\n', ' ', ''],
                                               chunk_size = 350,
                                               chunk_overlap = 65)
chunks = text_splitter.split_documents(documents)                                          

### Initializing the embedding model

from langchain_huggingface import HuggingFaceEmbeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 1: Initialize the Chroma DB connection

### Lets initalize our vector db

from langchain_chroma import Chroma

db = Chroma(collection_name = 'vect-db',
            embedding_function = hf_embeddings,
            persist_directory = 'chroma_db')

db.add_documents(chunks)

# Step 2: Create a retriever object

retriever = db.as_retriever(search_type = 'similarity', search_kwargs = {'k':5})

# Step 3: Initialize a Chat Prompt Template

from langchain_core.prompts import ChatPromptTemplate

prompt_template = '''
Answer the question based on the following context:
{context}
Answer the question based on the above context: {question}
Provide a detailed answer.
Don't justify your answers.
Don't give information not mentioned in the CONTEXT INFORMATION.
please also dont mention that the provided text is saying that.
just give the response.
Do not say "according to the context" or "mentioned in the context" or similar.
'''
prompt_temp = ChatPromptTemplate.from_template(prompt_template)

# Step 4: Initialize a Generator (Chat Model)

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

chat_model = ChatGoogleGenerativeAI(api_key = GOOGLE_API_KEY, model = 'gemini-1.5-flash')

# Step 5: Initialize a Output Parser

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

# Step 6: Build a RAG chain

from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)
    
chain = ({'context' : retriever | format_docs, 'question' : RunnablePassthrough()} | prompt_temp | chat_model | output_parser)

# Step 7: Invoke the chain

# Invoke the Chain

query = 'What did krishan said to arjun?'

chain.invoke(query)

