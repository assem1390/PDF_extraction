import streamlit as st
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import json

# Sidebar contents
image = Image.open('pdf-parser.jpg')
with st.sidebar:
    st.title('PDF Scraping App')
    st.markdown('''
    ## About
    This app is LLM-powered and helps to scrape all kinds of information from PDFs.
    ''')
    st.image(image)

# Load environment variables
load_dotenv()

pdf_file = r'tsla-def14a_20220804-gen.pdf'

# Function to process PDF and create chunks of text
def main_process_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Chunk size
        chunk_overlap=300,  # Overlap between chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

# Main application logic
def main():
    pdf_file = r'tsla-def14a_20220804-gen.pdf'
    store_name = pdf_file[:-4]
    st.write(f'{store_name}')

    # Check if the vector store already exists
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        # Generate embeddings and store them
        embeddings = OpenAIEmbeddings()  # Ensure OpenAI API key is loaded
        chunks = main_process_pdf(pdf_file)
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

    # Button to request the summary compensation table
    exact_request = None
    if st.button("SUMMARY COMPENSATION TABLE"):
        exact_request = '''Name and Principal Position
Year
Salary
Bonus
Stock Awards
Option Awards
Non-Equity Incentive Plan Compensation
All Other Compensation
Total'''

    # If the button was clicked and the request is made
    if exact_request:
        docs = VectorStore.similarity_search(query=exact_request, k=1)
        query = f"First, find the column headers, then give me back the summary compensation table from this data {docs}. Return your answer in JSON format, ignore the Principal Position, and only use names. Use the column headers as keys. Do this for all years."
        
        llm = OpenAI(max_tokens=1000, temperature=0, model="text-davinci-003")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)

        # Attempt to parse response and store the result as JSON
        try:
            response = response.replace('_', '')
            data_dict = json.loads(response)
            with open(f'{store_name}.json', 'w') as f:
                json.dump(data_dict, f)
            st.write(response)
        except json.JSONDecodeError:
            st.error("Failed to parse JSON from LLM response")

# Display the compensation data using Streamlit
def display_compensation_data(store_name):
    try:
        with open(f'{store_name}.json', 'r') as f:
            data_dict = json.load(f)
        executive = st.selectbox('Executive officers', list(data_dict.keys()))
        df = pd.DataFrame(data_dict[executive])
        st.dataframe(df)
    except FileNotFoundError:
        st.error("JSON file not found. Please ensure the PDF was processed correctly.")
    except json.JSONDecodeError:
        st.error("Error parsing JSON data. Please check the file format.")

if __name__ == '__main__':
    main()
    display_compensation_data('tsla-def14a_20220804-gen')
