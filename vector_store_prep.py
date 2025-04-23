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
# Reference for os.walk: https://www.w3schools.com/python/ref_os_walk.asp
# Reference for os.path.join: https://www.geeksforgeeks.org/python-os-path-join-method/
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

# Append all the documents to the "documents" list to then be able to process them all together
documents = []
for doc in word_docs:
    loader = Docx2txtLoader(doc)
    temp = loader.load()
    documents.extend(temp)

for doc in xlsx_docs:
    loader = UnstructuredExcelLoader(doc)
    temp = loader.load()
    documents.extend(temp)
print(len(documents))
    # print(temp)
    # break

def number_documents():
    len(documents)
# Separate the text from documents into chunks for the vector database
#Reference for text splitter code: https://python.langchain.com/v0.2/docs/tutorials/rag/
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
#print(chunks)

# Referemce for tiktoken: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o-mini")


embeddings = OllamaEmbeddings(model='mxbai-embed-large', base_url='http://localhost:11434')

# Reference for vector store initialization: https://github.com/langchain-ai/langchain/discussions/25376
vector = embeddings.embed_query("Hello World")
index = faiss.IndexFlatL2(len(vector))
#print(index.ntotal)
#print(index.d)

# Definition of the vector store
# Reference for FAISS vector store: https://python.langchain.com/docs/integrations/vectorstores/faiss/
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Reference for debugging check: https://stackoverflow.com/questions/70624600/faiss-how-to-retrieve-vector-by-id-from-python
print(vector_store.index.ntotal, vector_store.index.d)


# Save the vectors in a local vector database
ids = vector_store.add_documents(documents=chunks)

print(len(ids), vector_store.index.ntotal)

#store vector db
db_name = "test_db"
vector_store.save_local(db_name)
