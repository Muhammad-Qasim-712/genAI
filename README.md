# Multi-Mode AI Assistant

A comprehensive Streamlit-powered AI assistant that combines conversational AI, document analysis, and computer vision capabilities. This application integrates Google's Gemini AI for natural language processing, YOLO for object detection, and RAG (Retrieval-Augmented Generation) for document-based question answering.

## Features

- 🤖 **Generic Chat Mode** - Interactive conversations with Google Gemini AI
- 📄 **Document Chat (RAG)** - Upload and chat with PDF, DOCX, and TXT files using vector embeddings
- 👁️ **Object Detection** - Real-time object detection in images using YOLO v8
- 📧 **Email Integration** - Send AI responses and detected images via email
- 💾 **Vector Storage** - Efficient document embedding using FAISS and HuggingFace
- 🔄 **Multi-Session Support** - Separate conversation histories for each mode

## Installation

### Prerequisites

- Python 3.8 or higher
- Gmail account with App Password (for email functionality)
- Google AI API key

### System Dependencies

Install required system packages (Linux/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### Python Dependencies

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-mode-ai-assistant.git
cd multi-mode-ai-assistant
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

1. Set up your Google AI API key in Streamlit secrets:
```toml
# .streamlit/secrets.toml
api_key = "your-google-ai-api-key-here"
```

2. Configure email settings in the sidebar when running the app

## Usage

### Running the Application

```bash
streamlit run gempro.py
```

The application will open in your default web browser at `http://localhost:8501`

### Generic Chat Mode

```python
# Simply select "Generic Chat" mode and start typing
# Example conversations:
"What is machine learning?"
"Explain quantum computing in simple terms"
"Write a Python function to sort a list"
```

### Document Chat (RAG) Mode

1. Select "Chat with Documents (RAG)" mode
2. Upload documents (PDF, DOCX, TXT) in the sidebar
3. Click "Process Documents"
4. Ask questions about your documents:

```python
# Example queries after uploading documents:
"What are the main points in this document?"
"Summarize the key findings"
"What does the document say about [specific topic]?"
```

### Object Detection Mode

1. Select "Object Detection" mode
2. Upload an image (JPG, JPEG, PNG)
3. Click "Detect Objects"
4. Optionally email the results

## Dependencies

### Core Framework
- `streamlit` - Web application framework

### AI & Machine Learning
- `google-generativeai` - Google Gemini AI integration
- `ultralytics` - YOLO object detection
- `torch` & `torchvision` - PyTorch deep learning framework
- `sentence-transformers` - Text embeddings
- `scikit-learn` - Machine learning utilities

### Document Processing & Vector Storage
- `langchain-community` - LangChain community components
- `faiss-cpu` - Efficient similarity search
- `PyPDF2` - PDF text extraction
- `python-docx` - DOCX file handling

### Computer Vision
- `opencv-python-headless` - OpenCV without GUI dependencies
- `PIL` (Pillow) - Image processing

### Utilities
- `pandas` - Data manipulation
- `python-dotenv` - Environment variable management
- `smtplib` - Email functionality (built-in)

## Configuration

### API Keys Required

1. **Google AI API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Gmail App Password**: Generate from [Google Account Security](https://myaccount.google.com/security)

### Environment Setup

Create a `.streamlit/secrets.toml` file:

```toml
api_key = "your-google-ai-api-key"
```

## File Structure

```
multi-mode-ai-assistant/
├── gempro.py              # Main application file
├── requirements.txt       # Python dependencies
├── packages.txt          # System dependencies
├── .gitignore           # Git ignore rules
├── .streamlit/
│   └── secrets.toml     # API keys and secrets
└── README.md            # This file
```

## Troubleshooting

### Common Issues

1. **OpenCV Import Error**: Ensure `opencv-python-headless` is installed before `ultralytics`
2. **YOLO Model Download**: The first object detection will download the YOLOv8 model (~6MB)
3. **Email Authentication**: Use Gmail App Password, not your regular password
4. **Vector Store Error**: Ensure documents are successfully uploaded and processed

### Performance Tips

- Use smaller images for faster object detection
- Process documents in smaller batches for better memory usage
- Clear conversation history periodically to free up memory

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 coding standards
- Add error handling for new features
- Update requirements.txt for new dependencies
- Test all modes before submitting PRs

## Support

If you encounter issues or have questions:
- 🐛 [Create an Issue](https://github.com/yourusername/multi-mode-ai-assistant/issues)
- 📧 Contact: your-email@example.com
- 💬 [Discussions](https://github.com/yourusername/multi-mode-ai-assistant/discussions)

## Acknowledgments

- Google AI for Gemini API
- Ultralytics for YOLO implementation
- Streamlit community for the amazing framework
- HuggingFace for embedding models

---

Made with ❤️ using Streamlit and Google AI
