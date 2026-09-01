from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "assignment-test-knowledge-bbl"
NAMESPACE = "knowledge"
KNOWLEDGE_FILE = os.path.join(os.path.dirname(__file__), "knowledge.txt")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
BATCH_SIZE = 90  # upsert_records caps batches around ~96 records / 2MB

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

if not pc.has_index(INDEX_NAME):
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "content"}
        }
    )

index = pc.Index(INDEX_NAME)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

def load_knowledge_file(path=KNOWLEDGE_FILE, source_name=None):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        print(f"'{path}' is empty, nothing to import.")
        return

    source_name = source_name or os.path.basename(path)
    
    # Chunking step
    chunks = splitter.split_text(text)

    records = [
        {"_id": f"{source_name}-{i}", "content": chunk, "source": source_name}
        for i, chunk in enumerate(chunks)
    ]

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        index.upsert_records(NAMESPACE, batch)
        print(f"Upserted records {i} to {i + len(batch)} of {len(records)}")

    print(f"Done. Imported {len(records)} chunks from '{path}' into '{INDEX_NAME}' (namespace '{NAMESPACE}').")


if __name__ == "__main__":
    load_knowledge_file()