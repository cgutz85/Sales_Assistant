import nltk
import os
import warnings
import faiss
import tiktoken
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Read all the available docx files and append to a list
word_docs = []
for root, dirs, files in os.walk("Testdata"):
   #print(root, dirs, files)
    for file in files:
        if file.endswith(".docx"):
            word_docs.append(os.path.join(root, file))
#Test that the testfile was correctly appended
#print(word_docs)


# Read all the available xlsx files and append to a list
xlsx_docs = []
for root,dirs,files in os.walk("Testdata"):
    for file in files:
        if file.endswith(".xlsx"):
            xlsx_docs.append(os.path.join(root, file))

# Append all the documents to the "docs" list to then be able to process them all together
docs = []
for doc in word_docs:
    loader = Docx2txtLoader(doc)
    temp = loader.load()
    docs.extend(temp)

for doc in xlsx_docs:
    loader = UnstructuredExcelLoader(doc)
    temp = loader.load()
    docs.extend(temp)
print(len(docs))
    # print(temp)
    # break

def number_documents():
    len(docs)
# Separate the text from documents into chunks for the vector database
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
#print(chunks)

encoding = tiktoken.encoding_for_model("gpt-4o-mini")


embeddings = OllamaEmbeddings(model='mxbai-embed-large', base_url='http://localhost:11434')

vector = embeddings.embed_query("Hello World")

index = faiss.IndexFlatL2(len(vector))
#print(index.ntotal)
#print(index.d)

# Definition of the vector store
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

print(vector_store.index.ntotal, vector_store.index.d)


# Save the vectors in a local vector database
ids = vector_store.add_documents(documents=chunks)

print(len(ids), vector_store.index.ntotal)

#store vector db
db_name = "test_db"
vector_store.save_local(db_name)