from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import os
from Orchestration.state import OrchestrationState

load_dotenv()

LLM_MODEL = "nvidia/nemotron-3.5-lightning:free"

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


def report_generator_agent(messages: list) -> str:
    """Generate a report based on the provided question."""
    completion = _create_completion(
        model=LLM_MODEL,
        messages=messages,
    )
    return completion.choices[0].message.content


def report_generator_node(state: OrchestrationState) -> dict:
    """LangGraph node wrapper: reads question + retrieved_data from state, writes report."""
    messages = [
        {
            "role": "system",
            "content": "Expert writer and synthesizer. Uses the provided information snippets to formulate a comprehensive, high-quality answer for the end-user."
        },
        {
            "role": "user",
            "content": (
                f"Question: {state['question']}\n\n"
                f"Relevant information:\n{state['retrieved_data']}\n\n"
                "Please generate a comprehensive report answering the question using the information above."
            )
        }
    ]
    report = report_generator_agent(messages)
    return {"messages": {"role": "assistant", "content": report}, "report": report, "state": "done"}