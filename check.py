import os
from pathlib import Path
from dotenv import load_dotenv

here = Path(__file__).resolve().parent
print("Folder:", here)
print("Files starting with .env:", [p.name for p in here.glob(".env*")])
load_dotenv(here / ".env")
print("Key found:", bool(os.getenv("GOOGLE_API_KEY")))