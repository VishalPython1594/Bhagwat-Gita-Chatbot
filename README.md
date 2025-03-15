# 🕉️ Bhagavad Gita Chatbot - AI-Powered Spiritual Guide  

## 📌 Project Overview  
The **Bhagavad Gita Chatbot** is an **AI-powered question-answering system** that uses **Retrieval-Augmented Generation (RAG)** to provide **accurate and meaningful answers** from the Bhagavad Gita. It extracts teachings from a **PDF version of the Bhagavad Gita** and delivers insightful responses using **Google Gemini AI**.

✅ **Key Features:**  
- 🔍 **Retrieves relevant verses & teachings** from the Bhagavad Gita.  
- 💡 **Provides AI-generated responses** based on spiritual context.  
- 📜 **Uses a RAG pipeline** (ChromaDB + Gemini AI) for **enhanced accuracy.**  
- 🖥️ **User-friendly chatbot interface** powered by Streamlit.  

This tool helps **spiritual seekers, students, and researchers** explore the Bhagavad Gita's wisdom interactively.

---

## 📊 Technologies Used  
- **Python** (Backend)  
- **Streamlit** (Web UI)  
- **LangChain** (LLM-based RAG architecture)  
- **Google Gemini AI** (Generative model for answering questions)  
- **Hugging Face Embeddings** (Sentence transformers for text similarity)  
- **ChromaDB** (Vector database for efficient retrieval)  
- **PyPDFLoader** (Extracting Bhagavad Gita text from PDF)  

---

## ⚙️ Installation & Setup  

### **1️⃣ Clone the repository**  
```bash
git clone https://github.com/VishalPython1594/Bhagwat-Gita-Chatbot.git
cd Bhagavad-Gita-Chatbot
```

### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Set up API Keys**
* Obtain a Google Gemini API Key from Google AI Studio
* Add the API key to a .env file in the project directory:
```bash
GOOGLE_API_KEY=your_google_api_key
```

### **4️⃣ Add the Bhagavad Gita PDF**
Place the Geeta.pdf file in the project directory.

### **5️⃣ Run the Streamlit Chatbot**
```bash
streamlit run geeta_test.py
```

## **🏗️ Project Workflow**
1️⃣ User enters a spiritual or philosophical question.
2️⃣ RAG pipeline retrieves the most relevant sections from the Bhagavad Gita.
3️⃣ Google Gemini AI generates a structured response based on the retrieved context.
4️⃣ The response is displayed interactively in the chatbot UI.

## **🖥️ Usage**
1. Run the chatbot app:
```bash
streamlit run geeta_test.py
```
2. Enter your spiritual question (e.g., "What is the meaning of duty in the Bhagavad Gita?").
3. Click "Get Answer" to receive an AI-generated response.
4. Read and reflect on the teachings from the Bhagavad Gita.

## **📊 Sample Output**
📖 **Teachings from the Bhagavad Gita on Karma**
---------------------------------------------------
🔹 **Perform your duty selflessly** without attachment to results. (Chapter 2, Verse 47)  
🔹 **Actions should be performed with devotion** to the divine. (Chapter 3, Verse 19)  
🔹 **Karma Yoga (Path of Action)** leads to liberation when done without selfish desires. (Chapter 5, Verse 12)  
🔹 **Even the wisest act in accordance with their nature**—understanding this helps one act righteously. (Chapter 3, Verse 33)

## **🎯 Future Improvements**
🔹 Improve response accuracy with fine-tuned embeddings.
🔹 Expand dataset to include commentaries on Bhagavad Gita.
🔹 Integrate voice-based question input for accessibility.
🔹 Deploy as a chatbot API for mobile & web integration.

## 🤝 Contributing
Contributions are welcome! Feel free to submit a pull request if you have improvements or new features.

## 📜 License
This project is licensed under the MIT License.

## 📩 Contact & Support
📧 Email: vishal1594@outlook.com
🔗 LinkedIn: https://www.linkedin.com/in/vishal-shivnani-87487110a/
