from src.helper import load_pdfs, split_text, dwn_huggingface_embeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as pn
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

data = load_pdfs("data/Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf")

chunks = split_text(data)

embeddings = dwn_huggingface_embeddings()

pc = pn(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

index = pc.Index(index_name)

docsearch = Pinecone.from_texts(
    [ch.page_content for ch in chunks],
    embeddings,
    index_name=index_name,
)
