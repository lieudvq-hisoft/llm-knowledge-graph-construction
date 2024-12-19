import os
from dotenv import load_dotenv
load_dotenv()

# tag::llm[]
# Create the LLM
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# llm = ChatOpenAI(
#     openai_api_key=os.getenv('OPENAI_API_KEY'),
#     model_name="gpt-3.5-turbo"
# )
llm = ChatOllama(model="llama3.1:8b")
# end::llm[]

# tag::embedding[]
# Create the Embedding model
from langchain_openai import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings(
#     openai_api_key=os.getenv('OPENAI_API_KEY')
# )
embeddings = OllamaEmbeddings(model="bge-m3:latest")
# end::embedding[]
