import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys and Tokens (read from environment variables)
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "tabular_data")
HF_TOKEN = os.getenv("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")
TIME_BUDGET = int(os.getenv("TIME_BUDGET", 60 * 60 * 4))
MONITOR = os.getenv("MONITOR") is not None
MEMORY = os.getenv("MEMORY") is not None
GPU = os.getenv("GPU")
CPU = os.getenv("CPU") is not None
IMG_DEBUG = os.getenv("IMG_DEBUG") is not None
SAVE_PRED = os.getenv("SAVE_PRED") is not None
ALLOW_HIGHLTY_MULTICLASS = False

DEVICE = None
if GPU is not None:
    DEVICE = f"cuda:{GPU}"
if CPU:
    DEVICE = "cpu"
