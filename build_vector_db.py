import os
import re
import chromadb
from sentence_transformers import SentenceTransformer

# === Setup ChromaDB client ===
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="institutions")

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Load your text data ===
with open("institution_descriptions.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# === Split into chunks ===
chunks = [block.strip() for block in re.split(r"-{5,}", raw_text) if len(block.strip()) > 50]

# === Embed and store chunks ===
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        metadatas=[{"source": f"chunk_{i}"}],
        ids=[f"inst_{i}"]
    )

print(f"✅ Successfully embedded and stored {len(chunks)} chunks into ChromaDB.")
