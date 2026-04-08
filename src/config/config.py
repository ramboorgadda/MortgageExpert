import os 
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.custom_exception import CustomException

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
logger = get_logger(__name__)
if not GROQ_API_KEY:
    
    logger.error("GROQ_API_KEY is not set in the environment variables.")
    raise CustomException("GROQ_API_KEY is required but not set in the environment variables.", None)
logger.info("All API keys loaded successfully and config initialized.")