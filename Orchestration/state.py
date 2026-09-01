from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class OrchestrationState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    state: Literal["initial", "retrieving", "reporting", "done"]
    question: Annotated[str, "The question to be answered."]
    retrieved_data: Annotated[str, "The data retrieved from the knowledge base."]
    report: Annotated[str, "The final report generated based on the retrieved data."]