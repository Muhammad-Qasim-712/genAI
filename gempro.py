<<<<<<< HEAD
import streamlit as st
import google.generativeai as genai
import os
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
from PIL import Image
import numpy as np
import sys

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Mode AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Helper Functions ---

def send_email(recipient_email, subject, body, sender_email, sender_password):
    """Sends an email using SMTP."""
    try:
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def get_file_text(uploaded_files):
    """Extracts text from a list of uploaded files (PDF, DOCX, TXT)."""
    text = ""
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            try:
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Error reading PDF {uploaded_file.name}: {e}")
        elif uploaded_file.name.endswith('.docx'):
            try:
                doc = Document(uploaded_file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                st.error(f"Error reading DOCX {uploaded_file.name}: {e}")
        elif uploaded_file.name.endswith('.txt'):
            try:
                text += uploaded_file.read().decode("utf-8") + "\n"
            except Exception as e:
                st.error(f"Error reading TXT {uploaded_file.name}: {e}")
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    if not text_chunks: return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# --- Main App ---

st.title("🤖 Multi-Mode AI Assistant (Gemini Edition)")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    # IMPORTANT: Replaced hardcoded key with a secure input method
    api_key = "AIzaSyCu6VCLAiznE0wHNtQN-1TWEaSDh0lL7sY"

    st.header("Email Configuration")
    st.text_input("Your Email Address", key="sender_email")
    st.text_input("Your Email App Password", type="password", key="sender_password")
    st.info("For Gmail, you need to generate an 'App Password'.")

    st.header("Mode")
    app_mode = st.radio(
        "Choose the assistant mode:",
        ("Generic Chat", "Chat with Documents (RAG)", "Object Detection")
    )

# --- API Key Check and Model Initialization ---
model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")

if not model and app_mode != "Object Detection":
    st.warning("Please enter a valid Google API key in the sidebar to use the chat features.")
    st.stop()

# --- Session State Initialization ---
if "messages" not in st.session_state: st.session_state.messages = {}
if app_mode not in st.session_state.messages: st.session_state.messages[app_mode] = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None

# --- RAG Mode Logic ---
if app_mode == "Chat with Documents (RAG)":
    st.header("Chat with Your Documents")
    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader("Upload files and click 'Process'", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    raw_text = get_file_text(uploaded_files)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        st.session_state.vector_store = get_vector_store(text_chunks)
                        st.success("Documents processed! You can now ask questions.")
                    else: st.error("Could not extract text.")
            else: st.warning("Please upload at least one document.")

    for i, message in enumerate(st.session_state.messages[app_mode]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                with st.expander("✉️ Email this response"):
                    with st.form(key=f"email_form_rag_{i}"):
                        recipient = st.text_input("Recipient's Email Address")
                        if st.form_submit_button("Send"):
                            if st.session_state.sender_email and st.session_state.sender_password:
                                with st.spinner("Sending email..."):
                                    if send_email(recipient, "Chatbot Response", message["content"], st.session_state.sender_email, st.session_state.sender_password):
                                        st.success(f"Email sent to {recipient}!")
                            else:
                                st.warning("Please configure your email credentials in the sidebar.")

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages[app_mode].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            if st.session_state.vector_store is not None:
                with st.spinner("Thinking..."):
                    docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                    prompt_template = f"Use the following context to answer the question. If you don't know, say you don't know.\n\nContext: {context}\n\nQuestion: {prompt}\n\nAnswer:"
                    response = model.generate_content(prompt_template)
                    st.markdown(response.text)
                    st.session_state.messages[app_mode].append({"role": "assistant", "content": response.text})
            else:
                st.warning("Please upload and process documents first.")

# --- Generic Chat Mode Logic ---
elif app_mode == "Generic Chat":
    st.header("Generic Chat")
    
    for i, message in enumerate(st.session_state.messages[app_mode]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                with st.expander("✉️ Email this response"):
                    with st.form(key=f"email_form_generic_{i}"):
                        recipient = st.text_input("Recipient's Email Address")
                        if st.form_submit_button("Send"):
                            if st.session_state.sender_email and st.session_state.sender_password:
                                with st.spinner("Sending email..."):
                                    if send_email(recipient, "Chatbot Response", message["content"], st.session_state.sender_email, st.session_state.sender_password):
                                        st.success(f"Email sent to {recipient}!")
                            else:
                                st.warning("Please configure your email credentials in the sidebar.")

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages[app_mode].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            # Gemini uses a different format for chat history
            chat = model.start_chat(history=[])
            response = chat.send_message(prompt)
            st.markdown(response.text)
            st.session_state.messages[app_mode].append({"role": "assistant", "content": response.text})

# --- Object Detection Mode Logic ---
elif app_mode == "Object Detection":
    st.header("Object Detection with YOLO")
    st.write("Upload an image to detect objects within it.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting objects..."):
            try:
                model_yolo = YOLO('yolov8n.pt') 
                results = model_yolo(image)
                result_image_bgr = results[0].plot()
                result_image_rgb = Image.fromarray(result_image_bgr[..., ::-1])
                with col2:
                    st.image(result_image_rgb, caption="Image with Detections", use_column_width=True)
            except Exception as e:
                st.error(f"An error occurred during object detection: {e}")

# --- Common UI Elements ---
with st.sidebar:
    st.markdown("---")
    if st.button("Clear Current Mode History"):
        st.session_state.messages[app_mode] = []
        if app_mode == "Chat with Documents (RAG)": st.session_state.vector_store = None
        st.rerun()
=======
import streamlit as st
import google.generativeai as genai
import os
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
from PIL import Image
import numpy as np
import sys

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Mode AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Helper Functions ---

def send_email(recipient_email, subject, body, sender_email, sender_password):
    """Sends an email using SMTP."""
    try:
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def get_file_text(uploaded_files):
    """Extracts text from a list of uploaded files (PDF, DOCX, TXT)."""
    text = ""
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            try:
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Error reading PDF {uploaded_file.name}: {e}")
        elif uploaded_file.name.endswith('.docx'):
            try:
                doc = Document(uploaded_file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                st.error(f"Error reading DOCX {uploaded_file.name}: {e}")
        elif uploaded_file.name.endswith('.txt'):
            try:
                text += uploaded_file.read().decode("utf-8") + "\n"
            except Exception as e:
                st.error(f"Error reading TXT {uploaded_file.name}: {e}")
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    if not text_chunks: return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# --- Main App ---

st.title("🤖 Multi-Mode AI Assistant (Gemini Edition)")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    # IMPORTANT: Replaced hardcoded key with a secure input method
    api_key = "AIzaSyCu6VCLAiznE0wHNtQN-1TWEaSDh0lL7sY"

    st.header("Email Configuration")
    st.text_input("Your Email Address", key="sender_email")
    st.text_input("Your Email App Password", type="password", key="sender_password")
    st.info("For Gmail, you need to generate an 'App Password'.")

    st.header("Mode")
    app_mode = st.radio(
        "Choose the assistant mode:",
        ("Generic Chat", "Chat with Documents (RAG)", "Object Detection")
    )

# --- API Key Check and Model Initialization ---
model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")

if not model and app_mode != "Object Detection":
    st.warning("Please enter a valid Google API key in the sidebar to use the chat features.")
    st.stop()

# --- Session State Initialization ---
if "messages" not in st.session_state: st.session_state.messages = {}
if app_mode not in st.session_state.messages: st.session_state.messages[app_mode] = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None

# --- RAG Mode Logic ---
if app_mode == "Chat with Documents (RAG)":
    st.header("Chat with Your Documents")
    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader("Upload files and click 'Process'", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    raw_text = get_file_text(uploaded_files)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        st.session_state.vector_store = get_vector_store(text_chunks)
                        st.success("Documents processed! You can now ask questions.")
                    else: st.error("Could not extract text.")
            else: st.warning("Please upload at least one document.")

    for i, message in enumerate(st.session_state.messages[app_mode]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                with st.expander("✉️ Email this response"):
                    with st.form(key=f"email_form_rag_{i}"):
                        recipient = st.text_input("Recipient's Email Address")
                        if st.form_submit_button("Send"):
                            if st.session_state.sender_email and st.session_state.sender_password:
                                with st.spinner("Sending email..."):
                                    if send_email(recipient, "Chatbot Response", message["content"], st.session_state.sender_email, st.session_state.sender_password):
                                        st.success(f"Email sent to {recipient}!")
                            else:
                                st.warning("Please configure your email credentials in the sidebar.")

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages[app_mode].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            if st.session_state.vector_store is not None:
                with st.spinner("Thinking..."):
                    docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                    prompt_template = f"Use the following context to answer the question. If you don't know, say you don't know.\n\nContext: {context}\n\nQuestion: {prompt}\n\nAnswer:"
                    response = model.generate_content(prompt_template)
                    st.markdown(response.text)
                    st.session_state.messages[app_mode].append({"role": "assistant", "content": response.text})
            else:
                st.warning("Please upload and process documents first.")

# --- Generic Chat Mode Logic ---
elif app_mode == "Generic Chat":
    st.header("Generic Chat")
    
    for i, message in enumerate(st.session_state.messages[app_mode]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                with st.expander("✉️ Email this response"):
                    with st.form(key=f"email_form_generic_{i}"):
                        recipient = st.text_input("Recipient's Email Address")
                        if st.form_submit_button("Send"):
                            if st.session_state.sender_email and st.session_state.sender_password:
                                with st.spinner("Sending email..."):
                                    if send_email(recipient, "Chatbot Response", message["content"], st.session_state.sender_email, st.session_state.sender_password):
                                        st.success(f"Email sent to {recipient}!")
                            else:
                                st.warning("Please configure your email credentials in the sidebar.")

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages[app_mode].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            # Gemini uses a different format for chat history
            chat = model.start_chat(history=[])
            response = chat.send_message(prompt)
            st.markdown(response.text)
            st.session_state.messages[app_mode].append({"role": "assistant", "content": response.text})

# --- Object Detection Mode Logic ---
elif app_mode == "Object Detection":
    st.header("Object Detection with YOLO")
    st.write("Upload an image to detect objects within it.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting objects..."):
            try:
                model_yolo = YOLO('yolov8n.pt') 
                results = model_yolo(image)
                result_image_bgr = results[0].plot()
                result_image_rgb = Image.fromarray(result_image_bgr[..., ::-1])
                with col2:
                    st.image(result_image_rgb, caption="Image with Detections", use_column_width=True)
            except Exception as e:
                st.error(f"An error occurred during object detection: {e}")

# --- Common UI Elements ---
with st.sidebar:
    st.markdown("---")
    if st.button("Clear Current Mode History"):
        st.session_state.messages[app_mode] = []
        if app_mode == "Chat with Documents (RAG)": st.session_state.vector_store = None
        st.rerun()
>>>>>>> 973860402448c8720011c07d5b9755c2c160688f
