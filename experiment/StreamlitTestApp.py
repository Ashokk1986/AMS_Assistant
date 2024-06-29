## Import Libraries
import os
import streamlit as st
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

## Read variables from env file
load_dotenv()  
OPENAI_KEY=os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_INDEX=os.getenv("PINECONE_INDEX_NAME")

## Set up the Streamlit framework
## Set the title of the Streamlit app
st.title('AMS Assist Application')  

## create form using st.form
with st.form("user_inputs"):
    ## Create text input field in the Streamlit app
    input_text=st.text_input("I am your Support Assist. Ask your question!",max_chars=500)  
    ## Add Button
    button=st.form_submit_button("Assist Me")

## Pinecone VectorDB and embedding initiation
embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index=pc.Index(PINECONE_INDEX)
pinecone_store = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY, index=index,embedding=embeddings)

## Cosine Similarity Retreive Results from Pinecone VectorDB
def retrieve_query(query,k=5):
    matching_results=pinecone_store.similarity_search(query,k=k)
    return matching_results

## OpenAI Model - gpt-3.5-turbo and chain creation
llm=ChatOpenAI(openai_api_key=OPENAI_KEY,model_name="gpt-3.5-turbo", temperature=0.5)
chain=load_qa_chain(llm,chain_type="stuff")

## Search answers for the user query
def retrieve_answers(query):
    doc_search=retrieve_query(query,5)
    response=chain.run(input_documents=doc_search,question=query)
    return response

# Invoke the chain with the input text and display the output
if input_text:
    st.write(f"Answer: {retrieve_answers(input_text)}")

    