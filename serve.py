from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get API Key
Groq_api_key = os.getenv("Groq_Api")
if not Groq_api_key:
    raise ValueError("Groq_Api key not found in .env file")

# Set up model
model = ChatGroq(model="gemma2-9b-it", groq_api_key=Groq_api_key)

# Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following into {language}:"),
    ("human", "{text}")
])

# Use OutputParser to get only the string response
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Create FastAPI app
app = FastAPI(
    title="Langchain FastAPI Server",
    version="1.0",
    description="Translate text using LangChain + Groq + FastAPI"
)

# Add the route to /chain
add_routes(app, chain, path="/chain")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=True)
