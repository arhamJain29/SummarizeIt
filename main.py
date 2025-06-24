
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")


import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import os
import torch
import base64

checkpoint = 'MBZUAI/LaMini-Flan-T5-248M'
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint,
    device_map='auto',
    torch_dtype=torch.float32,
    offload_folder="offload"
)

def file_preprocessing(file):
  loader = PyPDFLoader(file)
  pages = loader.load_and_split()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap= 50)
  texts = text_splitter.split_documents(pages)
  final_texts = " "
  for text in texts:
    final_texts = final_texts + text.page_content
  return final_texts

def llm_pipeline(filepath):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )
    input_text = file_preprocessing(filepath)
    
    chunk_size = 1000
    words = input_text.split()
    summaries = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            result = pipe_sum(chunk)
            summaries.append(result[0]['summary_text'])
    
    return "\n\n".join(summaries)

@st.cache_data

def displayPDF(file):
  with open(file, "rb") as f:
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')  

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)  

st.set_page_config(layout = 'wide', page_title = 'PDF Summarizer' )
   

def main():

  st.title("PDF Summarizer by ARHAM JAIN")

  uploaded_file = st.file_uploader("Upload your PDF file ", type = ['pdf'])

  if uploaded_file is not None:
    if st.button("Summarize"):
      col1, col2 = st.columns(2)
      # Ensure the 'data' directory exists
      os.makedirs("data", exist_ok=True)
      filepath = "data/"+uploaded_file.name
      with open(filepath, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
        
      with col1:
        st.info("Uploaded PDF File")
        displayPDF(filepath)

      with col2:
        st.info("Summarizing PDF File")
        summary_result = llm_pipeline(filepath)
        st.success(summary_result)

if __name__ == '__main__':
  main()