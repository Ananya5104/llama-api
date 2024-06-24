from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key is None:
    raise ValueError("LANGCHAIN_API_KEY environment variable is not set")

# langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant. Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

# streamlit frameworks
st.title('Langchain demo with LLAMA2')
input_text = st.text_input("search the topic u want")

# ollamaa LLM
llm = Ollama(model='llama2')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))