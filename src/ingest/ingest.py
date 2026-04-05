from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
import os
from sklearn.manifold import TSNE
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
from src.utils.logger import get_logger
from src.utils.custom_exception import CustomException
load_dotenv(override=True)
DB_NAME = str(Path(__file__).parent.parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent.parent / "data" / "raw")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
logger = get_logger(__name__)

logger.info("Starting the ingestion process.")


def fetch_documents():
    """Fetches documents from the specified knowledge base directory."""
    try:
        logger.info(f"Fetching documents from {KNOWLEDGE_BASE}")
        folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
        documents = []
        for folder in folders:
            if os.path.isdir(folder):
                logger.info(f"Processing folder: {folder}")
                loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
                documents.extend(loader.load())
        logger.info(f"Fetched {len(documents)} documents.")
        return documents
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise CustomException("Failed to fetch documents", e)
    

if __name__ == "__main__":
    documents = fetch_documents()
    logger.info("Completed fetching documents.")