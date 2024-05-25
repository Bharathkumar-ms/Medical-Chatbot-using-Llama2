from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import pinecone
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone
import os

load_dotenv()


# print(PINECONE_API_KEY)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY 

index_name = 'medical-chatbot'

from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_texts(texts = [t.page_content for t in text_chunks], embedding=embeddings, index_name=index_name)
