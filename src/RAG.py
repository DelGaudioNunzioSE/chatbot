import os
import gradio as gr
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.buffer_window import ConversationBufferWindowMemory

from langchain_ollama import OllamaLLM
from langchain_community.llms import CTransformers
from langchain_experimental.text_splitter import SemanticChunker


from langchain_ollama import OllamaEmbeddings







class ChatbotWithMemory:
    def __init__(self, pdf_directory  = "./app_pdf/", faiss_index_path ="faiss_index_new"):





        self.llm = OllamaLLM(model="mistral",num_predict=512) #default temperature = 0.8, top_p = 0.9


        self.embedding_model = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        
       
        template = """
            You are an agent specialized in Natural Language Processing. 
            Use the following pieces of context to answer the question after <<<>>>.
            Do NOT provide any explaination if something is not present in the context!

            Please follow these rules:
            1. If unrelated to context provided the answer to the question is: I can only answer questions related to the provided notes.
            Unrelated means that it does not concern the field of Natural Language Processing.
            You know only the information explicitly provided by the context, don't try to answer to something else or to give more information, do not think.

            Example of how to answer if the concept is not explictly mentioned in the context:

            QUESTION: What is Gradient Descent? (Not mentioned in the context)
            WRONG ANSWER: The concept of Gradient Descent is not explicitly mentioned in the given context. However, it is a common optimization algorithm used to minimize
            some function by iteratively moving in the direction of steepest descent as defined by the negative gradient
            RIGHT ANSWER: The concept of Gradient Descent is not explicitly mentioned in the given context.

            QUESTION: WHAT IS THE CANNY ALGORITHM? (Not mentioned in the context)
            WRONG ANSWER: The Canny Algorithm is not mentioned in this given context, so I can't provide an accurate answer based on the provided details. The context only discusses Natural Language Processing (NLP) tasks such as understanding and generation, and various models like SciBERT, BioBERT, ClinicalBERT, mBERT, FlanT5, ByT5, T5-3B, T5-11B, Ul2, 
            Multi-Modal T5, Efficient T5, etc. The Canny Algorithm is typically used in image processing for edge detection and filtering
            RIGHT ANSWER: The Canny Algorithm is not mentioned in this given context.

            When you say: The ARGUMENT is not explicitly mentioned in this given context, give a summary of the context and don't infer about the argument.
            When you say: is not explicitly mentioned in this given context. Report where you found it in the context.

            2. Avoid offensive language.
            4. Answer in proper English.
            5. Question is the actual question you have to answer.
            6. Just answer the question, don't start sayng "Based on the provided context and previous conversation, I can answer your question."
            7. Motivate your answer.
            Give me only 1 complete result please.

  
            Chat History:
            {chat_history}
            

            Context: {context}

            <<<
            Question: {question}
            >>>

            Answer:
            Before answering, check if the provided context contains the necessary information. If yes, answer based on it. If not, reply I don't know and don't try to answer.
            """

        adapt_template = """ Adapt the following question relative to Natural Language Processing.
                            Question: {question}
                            Answer with only one single better structured question.
                            Give me only 1 result please.
                            """            
        self.prompt_template = PromptTemplate(
            template=template,
            input_variables=["chat_history", "context", "question"]
        )

        self.prompt_template_adapt = PromptTemplate(
            template=adapt_template,
            input_variables=["question"]
        )


        loader = PyPDFDirectoryLoader(pdf_directory)
        self.docs = loader.load()

        #Text splitter basato sulla semantica
        self.text_splitter = SemanticChunker(
            self.embedding_model,
            min_chunk_size=1000
        )


        chunks = self.text_splitter.split_documents(self.docs)
        self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        self.vectorstore.save_local(faiss_index_path)

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        self.memory = ConversationBufferWindowMemory(
            k = 3,
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
            )



        

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm = self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt_template},
            return_source_documents=True,
        )









    def chat_response(self, query, history=None):

        result = self.qa_chain.invoke({"question": query})


        return result['answer'] 

            
    
        
if __name__ == "__main__":

    chatbot = ChatbotWithMemory()
    iface = gr.ChatInterface(
            type= "messages",
            fn=chatbot.chat_response,
            title="Chatbot",
            description="Ask questions about your documents",
        )
    iface.launch()
