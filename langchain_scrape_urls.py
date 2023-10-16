from langchain.llms import Ollama
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

ollama = Ollama(base_url='http://localhost:11434',  model='llama2')

loader = WebBaseLoader('https://en.wikipedia.org/wiki/University_of_Central_Florida')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
 
question = "what are the regional campuses of UCF?"
print(qachain({"query": question}))

# For tutorial see:
# https://www.youtube.com/watch?v=CPgp8MhmGVY