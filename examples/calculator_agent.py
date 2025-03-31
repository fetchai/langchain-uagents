"""Example of using UAgentRegisterTool with a calculator agent."""

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_uagents import UAgentRegisterTool, cleanup_uagent

# Load environment variables
load_dotenv()

# Get API token for Agentverse - NEVER hardcode API keys in source code
API_TOKEN = os.getenv("AV_API_KEY")
if not API_TOKEN:
    raise ValueError("Please set AV_API_KEY environment variable")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Define a simple calculator tool
def calculator_tool(expression: str) -> str:
    """Evaluates a basic math expression (e.g., '2 + 2 * 3')."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Create the langchain agent
tools = [
    Tool(
        name="Calculator Langchain Agent",
        func=calculator_tool,
        description="Useful for evaluating simple math expressions and calculations using langchain"
    )
]

llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Create and register the uAgent
tool = UAgentRegisterTool()
agent_info = tool.invoke({
    "agent_obj": agent,
    "name": "calculator_agent_langchain",
    "port": 8080,
    "description": "A calculator agent for evaluating simple math expressions and calculations using langchain tool",
    "api_token": API_TOKEN
})

# Print agent info
print(f"Created uAgent '{agent_info['name']}' with address {agent_info['address']} on port {agent_info['port']}")

# Keep the agent running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down calculator agent...")
    cleanup_uagent("calculator_agent_langchain")  # Fixed agent name to match creation
    print("Calculator agent stopped.") 