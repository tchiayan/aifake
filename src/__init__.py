from dotenv import load_dotenv
import logging 

logging.basicConfig(level=logging.INFO)

logging.info("Loading .env file")
load_dotenv()