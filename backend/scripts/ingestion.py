import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path to allow importing backend modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from backend.app.core.config import settings

# For docling
from docling.document_converter import DocumentConverter

# For markdown splitting and chunking
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# For Ollama LLM and Embeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# For Pinecone
from backend.app.services.pinecone_service import PineconeService

class ChunkMetadata(BaseModel):
    summary: str = Field(description="A brief summary of the chunk's content.")
    entities: list[str] = Field(description="List of key entities mentioned in the chunk.")

def extract_pdf_with_docling(file_path: str):
    print(f"Extracting PDF with Docling: {file_path}")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    
    # Get markdown representation
    markdown_text = result.document.export_to_markdown()
    
    # Docling document metadata
    docling_metadata = {
        "source": file_path,
        "title": Path(file_path).name
    }
    return markdown_text, docling_metadata

def split_by_markdown_headers(markdown_text: str):
    print("Splitting by markdown headers...")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)
    return md_header_splits

def smart_chunking(splits, max_tokens=800, overlap=160):
    print("Applying smart chunking...")
    
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def num_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=max_tokens,
        chunk_overlap=overlap,
    )
    
    final_chunks = []
    current_chunk_text = ""
    current_chunk_metadata = {}
    
    def add_current_buffer():
        nonlocal current_chunk_text, current_chunk_metadata
        if current_chunk_text:
            if num_tokens(current_chunk_text) > max_tokens:
                sub_chunks = recursive_splitter.split_text(current_chunk_text)
                for sc in sub_chunks:
                    final_chunks.append({"text": sc, "metadata": current_chunk_metadata.copy()})
            else:
                final_chunks.append({"text": current_chunk_text, "metadata": current_chunk_metadata.copy()})
            current_chunk_text = ""
            current_chunk_metadata = {}

    for split in splits:
        text = split.page_content
        metadata = split.metadata
        tokens = num_tokens(text)
        
        if tokens > max_tokens:
            add_current_buffer()
            sub_chunks = recursive_splitter.split_text(text)
            for sc in sub_chunks:
                final_chunks.append({"text": sc, "metadata": metadata.copy()})
        else:
            if num_tokens(current_chunk_text + "\n\n" + text) <= max_tokens:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + text
                    current_chunk_metadata.update(metadata)
                else:
                    current_chunk_text = text
                    current_chunk_metadata = metadata.copy()
            else:
                add_current_buffer()
                current_chunk_text = text
                current_chunk_metadata = metadata.copy()
                
    add_current_buffer()
    return final_chunks

def extract_metadata_with_ollama(chunk_text: str, docling_metadata: dict, chunk_metadata: dict):
    parser = JsonOutputParser(pydantic_object=ChunkMetadata)
    
    # We use config metadata model
    model_name = settings.OLLAMA_METADATA_MODEL
    llm = ChatOllama(model=model_name, format="json", temperature=0)
    
    prompt = PromptTemplate(
        template="Extract a brief summary and key entities from the following text.\n\n{format_instructions}\n\nText: {text}",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    try:
        llm_metadata = chain.invoke({"text": chunk_text})
    except Exception as e:
        print(f"Error extracting metadata from LLM: {e}")
        llm_metadata = {"summary": "", "entities": []}
    
    combined_metadata = {
        **docling_metadata,
        **chunk_metadata,
        "llm_summary": llm_metadata.get("summary", ""),
        "llm_entities": llm_metadata.get("entities", []),
    }
    return combined_metadata

def embed_and_index(chunks_with_metadata):
    print("Embedding and indexing into Pinecone...")
    pinecone_service = PineconeService()
    
    docs = []
    for chunk in chunks_with_metadata:
        safe_metadata = {}
        for k, v in chunk["metadata"].items():
            if isinstance(v, list):
                safe_metadata[k] = ", ".join(str(i) for i in v)
            elif v is None:
                continue
            else:
                safe_metadata[k] = str(v)
                
        import uuid
        doc = {"id": str(uuid.uuid4()), "text": chunk["text"], "metadata": safe_metadata}
        docs.append(doc)
        
    pinecone_service.upsert_documents(docs)
    print(f"Successfully indexed {len(docs)} chunks.")

def main(file_path: str):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
        
    markdown_text, docling_metadata = extract_pdf_with_docling(file_path)
    splits = split_by_markdown_headers(markdown_text)
    chunks = smart_chunking(splits, max_tokens=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
    
    print(f"Generated {len(chunks)} chunks. Extracting metadata with Ollama...")
    final_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Processing metadata for chunk {i+1}/{len(chunks)}...")
        combined_meta = extract_metadata_with_ollama(chunk["text"], docling_metadata, chunk["metadata"])
        final_chunks.append({"text": chunk["text"], "metadata": combined_meta})
        
    embed_and_index(final_chunks)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/ingestion.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    main(pdf_path)
