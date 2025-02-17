import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def app2_main():
    # Load environment variables
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    summary_prompt = """You are a YouTube transcript summarizer. Summarize the following text in a very detailed way, providing all essential points and explanations."""

    def get_yt_transcript(youtube_video_url):
        try:
            video_id = youtube_video_url.split("=")[1]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([entry['text'] for entry in transcript])
            return text
        except Exception as e:
            st.error(f"Failed to retrieve transcript: {str(e)}")
            return None

    def generate_gemini_content(text, prompt):
        model = genai.GenerativeModel("gemini-pro")
        try:
            response = model.generate_content(prompt + text)
            if response and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                return "No valid response generated. The content may have been blocked or no text was produced."
        except Exception as e:
            st.error(f"An error occurred during content generation: {str(e)}")
            return None

    def get_text_chunks(text):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            chunks = text_splitter.split_text(text)
            return chunks
        except Exception as e:
            st.error(f"Error splitting text into chunks: {str(e)}")
            return []

    def get_vector_store(text_chunks):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")

    def get_conversational_chain():
        try:
            prompt_template = """
            Answer the question as detailed as possible from the provided context.
            If the answer is not in the provided context, just say, "Answer not available in the context."
            Context:\n {context}?\n
            Question: \n{question}\n
            Answer:
            """
            model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            return chain
        except Exception as e:
            st.error(f"Error setting up conversational chain: {str(e)}")
            return None

    def user_input(user_question):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question, k=4)
            chain = get_conversational_chain()
            if chain:
                response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                st.write("Reply: ", response["output_text"])
        except Exception as e:
            st.error(f"Error processing user input: {str(e)}")

    # Streamlit Interface
    st.title("YouTube Transcript Summarizer & Q&A Assistant")

    if 'show_thumbnail' not in st.session_state:
        st.session_state.show_thumbnail = False
    if 'video_summary' not in st.session_state:
        st.session_state.video_summary = None
    if 'video_transcript' not in st.session_state:
        st.session_state.video_transcript = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    youtube_link = st.text_input("Enter YouTube video URL:")

    if youtube_link:
        st.session_state.show_thumbnail = True
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

        if st.button("Process Video"):
            transcript_text = get_yt_transcript(youtube_link)
            if transcript_text:
                text_chunks = get_text_chunks(transcript_text)
                if text_chunks:
                    get_vector_store(text_chunks)
                    st.session_state.video_transcript = transcript_text
                    st.session_state.video_summary = generate_gemini_content(transcript_text, summary_prompt)
                    st.session_state.show_thumbnail = False

    if st.session_state.video_summary:
        st.markdown("## Video Summary:")
        st.write(st.session_state.video_summary)

        st.markdown("## Ask Questions About the Video:")
        user_question = st.text_input("Ask a question about the video:")

        if user_question:
            user_input(user_question)

    if st.session_state.chat_history:
        st.markdown("## Conversation History:")
        for chat in st.session_state.chat_history:
            st.write(f"*You:* {chat['question']}")
            st.write(f"*Assistant:* {chat['answer']}")
