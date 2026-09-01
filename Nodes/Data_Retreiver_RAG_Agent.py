from pinecone import Pinecone
import os
from dotenv import load_dotenv
import json
from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from Orchestration.state import OrchestrationState

load_dotenv()

INDEX_NAME = "assignment-test-knowledge-bbl"
NAMESPACE = "knowledge"
LLM_MODEL = "nvidia/nemotron-3.5-lightning:free"

# Define pinecone index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

# Define OpenAI client for reasoning and completion
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
)
def _create_completion(**kwargs):
    """Call the LLM, retrying with backoff on free-tier 429 rate limits."""
    return client.chat.completions.create(**kwargs)

# Provide a function to search the knowledge base using Pinecone
tools = [{
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the knowledge or information which is relevant to corgi dog.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}]

def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    results = index.search(
        namespace=NAMESPACE,
        # To scope by metadata, add "filter": {"category": {"$eq": "policies"}} to the query dict
        query={"top_k": 3, "inputs": {"text": query}},
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": 3,
            "rank_fields": ["content"]
        }
    )
    return "\n\n".join(
        hit.fields["content"]
        for hit in results["result"]["hits"]
    )

# Create Node function to handle the search and reasoning process
def data_retreiver(messages: list) -> str :
    """"LLM retreive data from knowledge base and generate answer with reasoning."""""
    retrieved_data = _create_completion(
        model=LLM_MODEL,
        messages=messages,
        parallel_tool_calls=False,
        tools = tools
    )
    while retrieved_data.choices[0].finish_reason == "tool_calls":
        #print(retrieved_data.choices[0].message)
        tool_call = retrieved_data.choices[0].message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        if tool_name == "search_knowledge_base":
            query = tool_args.get("query")
            search_results = search_knowledge_base(query)
            return search_results
    return retrieved_data.choices[0].message.content or ""

def data_retreiver_node(state: OrchestrationState) -> dict:
    """LangGraph node wrapper: reads question from state, writes retrieved_data."""
    messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that retrieves relevant information from the knowledge base"
            },
            {
                "role": "user",
                "content": f"Please retrieve relevant information from the knowledge base for the following question:{state['question']}"
            }
        ]
    retrieved_data = data_retreiver(messages)
    return {"messages": {"role": "assistant", "content": retrieved_data}, "retrieved_data": retrieved_data, "state": "reporting"}
        