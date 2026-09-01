from Nodes import data_retreiver_node, report_generator_node
from Orchestration.state import OrchestrationState
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(OrchestrationState)

# Add nodes to workflow
workflow.add_node("data_retreiver", data_retreiver_node)
workflow.add_node("report_generator", report_generator_node)

# Set up edges to define the flow of data between nodes
workflow.add_edge(START, "data_retreiver")
workflow.add_edge("data_retreiver", "report_generator")
workflow.add_edge("report_generator", END)

app = workflow.compile()
