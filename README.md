
```markdown
# Multi-Function Summarizer App

This Streamlit app provides two main functionalities: summarizing YouTube video transcripts and summarizing PDF documents. It uses Google Generative AI for content generation and Langchain for text processing and question answering.

## Features

- **YouTube Video Summarizer**: 
  - Fetches and processes transcripts from YouTube videos.
  - Provides detailed summaries of video content.
  - Interactive Q&A based on the video transcript.

- **PDF Summarizer**:
  - Upload PDF documents to generate concise summaries.
  - Extracts key points and information from the document.

## Installation

To run this app locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory.
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_google_api_key
     ```

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **YouTube Video Summarizer**:
   - Enter a YouTube video URL in the provided input field.
   - Click "Process Video" to generate a summary and enable Q&A.

3. **PDF Summarizer**:
   - Upload a PDF document using the file uploader.
   - The app will process the document and display a summary.

## Deployment

This app can be deployed for free using Streamlit Community Cloud. Follow these steps:

1. Push your code to a GitHub repository.
2. Sign up or log in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Deploy your app by selecting your GitHub repository and specifying the app file.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for providing an easy-to-use framework for building web apps.
- [Langchain](https://langchain.com/) for text processing and question answering capabilities.
- [Google Generative AI](https://ai.google/) for content generation.

```

### Key Updates:

- **Features Section**: Clearly distinguishes between the YouTube Video Summarizer and the PDF Summarizer, explaining what each section does.
- **Usage Section**: Provides specific instructions for using both the YouTube and PDF summarization features.
- **General Structure**: Maintains a clear and organized structure, making it easy for users to understand and navigate the app's functionalities.

Feel free to adjust the content to better fit the specifics of your app, such as adding more detailed instructions or additional features.