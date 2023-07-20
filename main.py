Vector_folder_path = f'{root_dir}Vectors_2300_US/'
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


    
    query = f"From the mail below, extract what shall be quoted using the Price List, select the items that yields the lowest total cost, only the quantities from the price list should be used and combined to fulfill the requested items You are working at the sales office at Uponor and your name in Pamela Anderson, write a friendly quotation to the person signing the mail. If there are items in the request that are not included in the price list note that the price for those items will be given later. Include the price for the items that could be found in the price list. If the request is using meters, convert it to ft before calculating the cost: {content} "
    #system_message="price list: \n AquaPEX Pex pipe 1/2\": 1 usd/ft \n  AquaPEX Pex pipe 3/4\" : 1.3 usd/ft \n AquaPEX Pex pipe 3\" : 2 usd/ft \n Fitting 1/2\" : 2 usd/ft \n Fitting 3/4\" : 3 usd/ft \n Fitting 1\" : 4 usd/ft \n"
    system_message="""

"""
    



    
    messages = [{"role": "user", "content": query}, {"role": "system", "content": system_message} ] 

    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo-16k",
        model="gpt-4",
        temperature=0.5,
        messages=messages
    )

    arguments = response.choices[0]["message"]["content"]
    #response['choices'][0]['message']['content']})
    companyName ="z"
    cost = "0"
    mail_reply = arguments
    

    return {
        "companyName": companyName,
        "cost": cost,
        "mail_reply": mail_reply
    }
