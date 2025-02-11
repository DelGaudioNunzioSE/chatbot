from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.runnables import RunnablePassthrough


import numpy as np
import os
import gradio as gr

from langchain.memory import ConversationBufferMemory


# Load the secret key from the .env file
from dotenv import load_dotenv # load personal key (pip install python-dotenv)
dotenv_path = os.path.join(os.path.dirname(__file__), "config", "huggin_face.env")
load_dotenv(dotenv_path)
secret_key = os.getenv("SECRET_KEY")
print(secret_key)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = secret_key 


llm = HuggingFaceEndpoint( #modello llm
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature = 0.1
)

embedding_model = HuggingFaceBgeEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5", #posso scaricare il modello sul mio pc invece di scaricarlo ogni volta
    encode_kwargs = {'normalize_embeddings': True} # useful for similarity tasks 
)

#Question si riferisce alla variabile che passo quando chiamo il modello, puo chiamarsi in ogni modo quando lo chiamo.
template = """
Use the following pieces of context to answer the question at the end. 
Please follow the following rules:
1. If you don't know the answer, reply: "I can only answer questions related to the provided notes".
2. If you find the answer, write the answer in a concise way with five sentences maximum.
3. Answer in english with a correct grammar

{context}

###
Chat History: {history}
###
Question: {question}
###




Helpful Answer:
"""

prompt_template = PromptTemplate(template = template, input_variables = ["context","history" ,"question",])


loader = PyPDFDirectoryLoader("./app_pdf/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 50,
)
chunks = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(chunks, embedding_model) #chunks sono i documenti da indexare col modello embedding model
vectorstore.save_local("faiss_index_new")


retriever = vectorstore.as_retriever(
    search_type = "similarity", # use cosine similarity
    search_kwargs = {"k": 3} # use the top 3 most relevant chunks
    )

memory = ConversationBufferMemory(
                                    memory_key="history",
                                    max_len=50,
                                    return_messages=True,
                                    output_key='answer'
                                )

retrievalQA = RetrievalQAWithSourcesChain.from_chain_type(
    llm = llm,
    retriever = retriever,
    chain_type = "stuff", # concatenate retrieved chunks, concatena i chunks
    return_source_documents = True,
    memory = memory
)

def chatbot_response(query):
    response = retrievalQA.invoke(prompt_template.format(question=query, history=memory.chat_memory.messages, context=''))
    #chat_history.add_ai_message(response['result'])
    return response['answer']

iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text", title="Chatbot QA")
iface.launch()



