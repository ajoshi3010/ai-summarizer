# AI Summarizer & Q&A Assistant

[![Streamlit Deployment](https://img.shields.io/badge/Live%20Demo-Streamlit-orange)](https://ai-summarizer-multimodal.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/ajoshi3010/ai-summarizer)

## 📌 Project Overview
AI Summarizer & Q&A Assistant is a Streamlit-based web application that extracts YouTube video transcripts, summarizes them using Google's Gemini API, and allows users to ask questions about the video content using an AI-powered conversational agent.

## ✨ Features
- 🔍 **Extracts** YouTube video transcripts
- 📝 **Summarizes** transcripts in a detailed manner
- 🤖 **Q&A System** for users to ask questions about the video content
- ⚡ **FAISS-based Vector Search** for improved context retrieval
- 🎨 **User-friendly Interface** powered by Streamlit

## 🚀 Live Demo
Try out the app here: [AI Summarizer & Q&A Assistant](https://ai-summarizer-multimodal.streamlit.app/)

## 📂 GitHub Repository
Explore the source code: [GitHub Repository](https://github.com/ajoshi3010/ai-summarizer)

---

## ⚙️ Installation & Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip

### Clone the Repository
```sh
 git clone https://github.com/ajoshi3010/ai-summarizer.git
 cd ai-summarizer
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Set Up Environment Variables
Create a `.env` file in the project root and add your Google API key:
```ini
GOOGLE_API_KEY=your_google_api_key_here
```

### Run the Application
```sh
streamlit run app.py
```

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **AI Models:** Google Gemini Pro, Google Generative AI Embeddings
- **Vector Database:** FAISS
- **APIs:** YouTube Transcript API

---

## 🎯 How It Works
1. **Enter a YouTube Video URL**: The app fetches the transcript automatically.
2. **Transcript Processing**: The text is split into manageable chunks.
3. **Summarization**: Google Gemini AI generates a detailed summary.
4. **Vector Storage**: FAISS stores transcript chunks for efficient retrieval.
5. **Q&A System**: Users ask questions, and AI provides contextual answers.

---

## 📢 Contributing
Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact
For any questions or support, contact:
- **GitHub Issues**: [Report an issue](https://github.com/ajoshi3010/ai-summarizer/issues)

