import os
from dotenv import load_dotenv

#  加載環境變數
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('CHANNEL_SECRET')