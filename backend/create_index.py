import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-rag-index")

if not api_key:
    print("Error: PINECONE_API_KEY not found in environment variables.")
    sys.exit(1)

print("Connecting to Pinecone...")
pc = Pinecone(api_key=api_key)

existing_indexes = [idx.name for idx in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating Pinecone serverless index '{index_name}' (dim=768, metric='dotproduct')...")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Pinecone index created successfully!")
else:
    print(f"Pinecone index '{index_name}' already exists.")
