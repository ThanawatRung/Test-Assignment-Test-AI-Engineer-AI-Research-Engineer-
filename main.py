from Orchestration.orchestration import app

if __name__ == "__main__":
    print("Chat with the agent. Type 'end' to quit.\n")
    print("You can ask questions about corgi dogs, and the agent will retrieve relevant information from the knowledge base and generate a report.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "end":
            print("Goodbye!")
            break
        if not question:
            continue

        result = app.invoke({
            "messages": [],
            "state": "initial",
            "question": question,
            "retrieved_data": "",
            "report": "",
        })
        print("-----------------------------")
        print(f"Agent: {result['report']}\n")
        print("-----------------------------")
