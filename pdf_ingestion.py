import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import fitz  # PyMuPDF

from table_extraction import extract_pdf_tables
# from table_extraction import extract_tables_all_pages

def extract_pdf_text(pdf_path):
    pdf_doc = fitz.open(pdf_path)
    text = ""
    for page in pdf_doc:
        text += page.get_text()
    return text

pdf_path = "example.pdf"
extracted_text = extract_pdf_text(pdf_path)
print(extracted_text[:500])  # Display the first 500 characters

# Path to the PDF file
pdf_path = "example.pdf"

# Extract tables from Page 6 (0-indexed, so use 5)
page_number = 5
tables = extract_pdf_tables(pdf_path, page_number)
# all_tables = extract_tables_all_pages(pdf_path)

# Print the extracted tables
print("Extracted Tables:")
# for table in all_tables:
for table in tables:
    print(table)

# to chunk the extracted text
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

chunks = chunk_text(extracted_text)
print(f"Total Chunks: {len(chunks)}")
print(chunks[:3])  # Show first 3 chunks

# To Generate Embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
print(f"Generated Embeddings: {embeddings.shape}")

#Use FAISS to store and query embeddings
import faiss
import numpy as np

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings))
print("Embeddings added to FAISS index.")


#Query the System
query = "What is the unemployment rate for Master's degree holders??"
query_embedding = model.encode([query])

distances, indices = index.search(np.array(query_embedding), k=3)
relevant_chunks = [chunks[i] for i in indices[0]]
print("Relevant Chunks:", relevant_chunks)



import openai
openai.api_key = "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"
def generate_response(context, query):
    """
    Generate a response using OpenAI's ChatCompletion API.

    Args:
        context (str): Relevant context for the query.
        query (str): User's query.

    Returns:
        str: Response from the model.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Switch to "gpt-4" if available
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )
    return response["choices"][0]["message"]["content"]
response = generate_response(" ".join(relevant_chunks), query)
print("LLM Response:")
print(response)
