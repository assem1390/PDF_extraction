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
    st.title('PDF scraping App')
    st.markdown('''
    ## About
    This app is LLM-powered that help scrap 
    all kind of information from PDF
 
    ''')
    st.image(image)
    
load_dotenv()

pdf_file = r'tsla-def14a_20220804-gen.pdf'

def making_chuncks():
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, # chunk size
        chunk_overlap=300, # overlap between chuncks 
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)
    return chunks
def main():
    # embeddings
    store_name = pdf_file[:-4]
    st.write(f'{store_name}')
    # st.write(chunks)

    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings() # this cost money
        chunks = making_chuncks()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)


    exact_request = None
    comp_button = st.button("SUMMARY COMPENSATION TABLE")
    if comp_button:
        #query = f" "
        exact_request = '''Name and Principal Position Year
Salary

Bonus

Stock
Awards

Option
Awards

Non-Equity
Incentive
Plan
Compensation

All Other
Compensation

Total'''
        #st.write(query)

    
    if exact_request is not None:
        docs = VectorStore.similarity_search(query=exact_request, k=1)
        query = f"first find the column headres and then give me back the summary compensation table from this data {docs} return your answer in JSON fromat, ignore the Principal Position and only use names,use the column headres as keys, do this for all years"
        llm = OpenAI(max_tokens= 1000,temperature=0,model = "text-davinci-003")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        response = response.replace('_', '')
        data_dict = json.loads(response)
        with open(f'{store_name}.json', 'w') as f:
            json.dump(data_dict, f)
        st.write(response)


# store_name = run()
def main(store_name):
    with open(f'{store_name}.json', 'r') as f:
        data_dict = json.load(f)
    executive = st.selectbox('executive officers',list(data_dict.keys()))
    df = pd.DataFrame(data_dict[executive])
    st.dataframe(df)

 
if __name__ == '__main__':
    main('tsla-def14a_20220804-gen')

# if __name__ == '__main__':
#     main()