from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage,HumanMessage, convert_to_messages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.utils.logger import get_logger
from src.utils.custom_exception import CustomException
from src.config.config import GROQ_API_KEY

load_dotenv(override=True)

logger = get_logger(__name__)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = init_chat_model(
    model="groq:llama-3.3-70b-versatile",
    temperature = 0,
    api_key=GROQ_API_KEY
)
DB_NAME= str(Path(__file__).parent.parent.parent / "vector_db")
CHROMA_COLLECTION_NAME = "mortgageexpert"
RETRIEVAL_K = 5
SYSTEM_PROMPT="""
You are a knowledgeable, friendly assistant representing the company MortgageExpert.
You are chatting with a user about MortgageExpert.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

def get_retriever():
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=DB_NAME,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

def fetch_context(question : str) -> list[Document]:
    """Fetches the relevent context for a given question from vector db"""
    try:
        logger.info(f"fetching context for the question : {question}")
        relevant_docs = get_retriever().invoke(question)
        logger.info(f"fetched {len(relevant_docs)} relevant documents for the question")
        return relevant_docs
    except Exception as e:
        logger.error(f"Error fetching context: {e}")
        raise CustomException(f"Error fetching context: {e}", e)
    
def combined_question(question : str, history: list[dict] | None = None) -> str:
    """combine all the user messages into a single string"""
    try:
        history = history or []
        prior = "\n".join(m["content"] for m in history if m["role"] == "user")
        return prior + "\n" + question
    except Exception as e:
        logger.error(f"Error combining question and history: {e}")
        raise CustomException(f"Error combining question and history: {e}", e)
    
def answer_question(question: str, history: list[dict] | None = None) -> tuple[str,list[Document]]:
    """Answer the question using the llm and context from vector db"""
    try:
        history = history or []
        combined = combined_question(question, history)
        context = fetch_context(combined)
        logger.info(f"context fetched: {len(context)} documents")
        system_prompt = SystemMessage(content=SYSTEM_PROMPT.format(context="\n".join(doc.page_content for doc in context)))
        messages = [system_prompt]
        messages.extend(convert_to_messages(history))
        messages.append(HumanMessage(content=question))
        response = llm.invoke(messages)
        return response.content, context
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise CustomException(f"Error answering question: {e}", e)