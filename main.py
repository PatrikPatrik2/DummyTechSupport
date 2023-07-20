Vector_folder_path = '/var/lib/data/Vectors_2300_US/'


import openai
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
#from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import VectorDBQA
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()



class Email(BaseModel):
    from_email: str
    content: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/")
def analyse_email(email: Email):
    content = email.content
    persist_directory = Vector_folder_path

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":1000}))

    companyName ="z"
    cost = "0"
    mail_reply = qa.run(content)
    

    return {
        "companyName": companyName,
        "cost": cost,
        "mail_reply": mail_reply
    }
