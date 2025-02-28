from importlib.resources import open_text
import os
import faiss
import requests
import shutil
import json
import fitz  # PyMuPDF for extracting text
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv
from bs4 import BeautifulSoup  # For web scraping general URLs
import PyPDF2  # For PDF processing
from sentence_transformers import SentenceTransformer
import asyncio
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timezone
import pyttsx3  # Text-to-speech
import speech_recognition as sr  # Speech recognition



SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
SERVICE_ACCOUNT_FILE = os.path.abspath("D:/Chat Bot API/credentials.json")  # Update the path

# Load API keys from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SECRET = os.getenv("GOOGLE_SECRET")
CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE")

if not GROQ_API_KEY:
    raise ValueError("Error: GROQ_API_KEY not found in .env file!")

if not YOUTUBE_API_KEY:
    raise ValueError("Error: YOUTUBE_API_KEY not found in .env file!")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

video_data_cache = {}  # Caches transcripts & metadata for faster responses

# Load sentence embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS vector store
vector_db = None

def store_text_in_faiss(text):
    """Splits and stores text in FAISS vector storage."""
    global vector_db
    try:
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = text_splitter.create_documents([text])
        vector_db = FAISS.from_documents(documents, embedding_model)
        print("Text stored in FAISS successfully.")
    except Exception as e:
        print(f"Error storing text in FAISS: {str(e)}")

def retrieve_relevant_info(query):
    """Retrieves relevant context from stored FAISS vectors."""
    global vector_db
    if vector_db:
        try:
            docs = vector_db.similarity_search(query, k=2)
            context = " ".join([doc.page_content for doc in docs])
            print(f"Retrieved context: {context}")
            return context
        except Exception as e:
            print(f"Error retrieving info from FAISS: {str(e)}")
    return ""

def extract_video_id(video_url):
    """Extracts the video ID from a YouTube URL"""
    if "watch?v=" in video_url:
        return video_url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in video_url:
        return video_url.split("youtu.be/")[-1].split("?")[0]
    return None

def fetch_youtube_transcript(video_id):
    """Fetches transcript of a YouTube video using YouTubeTranscriptApi"""
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript_data])
        return transcript_text[:5000]  # Limit characters for efficiency
    except (TranscriptsDisabled, NoTranscriptFound):
        return None  # No subtitles available
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def fetch_youtube_metadata(video_id):
    """Fetches video title & description using YouTube API"""
    youtube_api_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={YOUTUBE_API_KEY}"
    response = requests.get(youtube_api_url)

    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            snippet = data["items"][0]["snippet"]
            title = snippet["title"]
            description = snippet["description"][:1000]  # Limit description size
            return title, description
        else:
            return None, "No metadata found."
    else:
        return None, f"Error fetching metadata: {response.text}"

def fetch_webpage_content(url):
    """Fetches the main content of a webpage and returns clean text."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return f"Error fetching webpage: {response.status_code}"

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")  # Extract paragraphs
        text_content = "\n".join([p.get_text() for p in paragraphs])

        return text_content[:5000]  # Limit characters for efficiency
    except Exception as e:
        return f"Error fetching webpage content: {str(e)}"

# ✅ Google Calendar Function - Fetch Upcoming Meetings
def get_calendar_events():
    """Fetch upcoming meetings from Google Calendar."""
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        service = build("calendar", "v3", credentials=creds)

        now = datetime.utcnow().isoformat() + "Z"  # Get current time in UTC format

        events_result = service.events().list(
            calendarId="sachinbfrnd@gmail.com",  # Ensure the correct email is used
            maxResults=10,  # Fetch up to 10 upcoming events
            singleEvents=True,
            orderBy="startTime",
            timeMin=now  # Ensures only upcoming events are fetched
        ).execute()

        events = events_result.get("items", [])

        if not events:
            return {"meetings": [], "voice": "No upcoming meetings found."}

        meetings = []
        voice_output = []

        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            summary = event.get("summary", "No Title")

            formatted_event = f"📅 {summary} on {start}"
            meetings.append(formatted_event)
            voice_output.append(f"{summary} on {start}")

        return {"meetings": meetings, "voice": ". ".join(voice_output)}

    except Exception as e:
        return {"meetings": [], "voice": f"Error fetching meetings: {str(e)}"}

# Function to Speak Text in the Background
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)  # Adjust speed for clarity
    engine.say(text)
    engine.runAndWait()

# Voice Input Recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Could not request results, check your internet connection."

# Initialize Groq model
llm = ChatGroq(model_name="llama3-8b-8192")

async def summarize_with_groq(context, query):
    """Summarizes text using Groq model."""
    try:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"Context: {context}\nQuery: {query}")
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error summarizing with Groq: {str(e)}")
        return "Sorry, I encountered an error while processing your request."

# Chatbot Request Model
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest, req: Request, background_tasks: BackgroundTasks):
    try:
        message = request.message.strip().lower()

        if await req.is_disconnected():
            return {"response": "Client disconnected before response."}

        if "meeting" in message or "schedule" in message:
            calendar_data = get_calendar_events()  # ✅ Fetch calendar events
            response_text = "\n".join(calendar_data["meetings"])  # ✅ Format for chatbot

            background_tasks.add_task(speak_text, calendar_data["voice"])  # ✅ Speak in background

            return {"response": response_text}  # ✅ Chatbot shows response immediately

        # Retrieve relevant context for chatbot
        context = "General chatbot context"
        response = await summarize_with_groq(context, f"Answer this based on context: {message}")

        return {"response": response}

    except Exception as e:
        return {"response": f"Internal Server Error: {str(e)}"}