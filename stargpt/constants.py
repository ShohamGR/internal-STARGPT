import os

from dotenv import load_dotenv

load_dotenv()

WANDB_API_KEY = "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"
WANDB_ENTITY = "tabular_data"
HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN
KAGGLE_USERNAME = "shohamgrunblat"
KAGGLE_KEY = "KGAT_5e15f00ca0d0a35ed695fbe6a84a5c70"
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
