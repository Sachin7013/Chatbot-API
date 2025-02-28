import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("⚠️  API Key not found! Check your .env file.")

chat_model = ChatGroq(api_key=GROQ_API_KEY)

def chat_with_bot(user_input):
    response = chat_model.invoke([HumanMessage(content=user_input)])  
    return response.content
