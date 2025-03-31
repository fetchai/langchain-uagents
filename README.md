# 🦜️🔗 LangChain UAgents

This package contains the LangChain integration with uAgents, allowing you to convert Langchain agents into uAgents and register them on Agentverse.

[langchain-uagents](https://github.com/fetchai/langchain-uagents)

[Agentverse website](https://agentverse.ai/)

## Features

- Convert Langchain agents to uAgents
- Automatic port allocation with fallback options
- Register agents on Agentverse
- Support for AI agent message forwarding
- ASI1-agentic accessible agent
- Clean shutdown and resource management

## Installation

```bash
pip install -U langchain-uagents
```

### Credentials

We also need to set our Agentverse API key. You can get an API key by visiting [this site](https://agentverse.ai/profile/api-keys), or follow the steps in the [Agentverse documentation](https://innovationlab.fetch.ai/resources/docs/agentverse/agentverse-api-key).

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API token for Agentverse
API_TOKEN = os.getenv("AV_API_KEY", "your_default_key_here")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_default_openai_key_here")
```

## UAgentRegisterTool

Here we show how to use the UAgentRegisterTool to convert a Langchain agent into a uAgent and register it on Agentverse.

### Instantiation

```python
from langchain_uagents import UAgentRegisterTool

tool = UAgentRegisterTool()
```

### Basic Usage

The UAgentRegisterTool accepts the following parameters during invocation:

- `agent_obj` (required): The Langchain agent object to convert
- `name` (required): Name for the uAgent
- `port` (optional, int): Port to run the agent on, defaults to 8000
- `description` (optional, str): Description of the agent's functionality
- `api_token` (required): Agentverse API token for registration
- `start_range` (optional, int): Start of port range for automatic allocation
- `end_range` (optional, int): End of port range for automatic allocation

```python
agent_info = tool.invoke({
    "agent_obj": agent,
    "name": "my_agent",
    "port": 8080,
    "description": "A useful agent for my tasks",
    "api_token": API_TOKEN
})

# Print agent info
print(f"Created uAgent '{agent_info['name']}' with address {agent_info['address']} on port {agent_info['port']}")
```

### Complete Example

Here's a complete example showing how to create a calculator agent with Langchain and register it as a uAgent:

```python
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_uagents import UAgentRegisterTool, cleanup_uagent

# Load environment variables
load_dotenv()

# Get API token for Agentverse
API_TOKEN = os.getenv("AV_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        name="Calculator",
        func=calculator_tool,
        description="Useful for evaluating math expressions"
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
    "name": "calculator_agent",
    "port": 8080,
    "description": "A calculator agent for testing",
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
    cleanup_uagent("calculator_agent")
    print("Calculator agent stopped.")
```

### Port Allocation

The tool automatically handles port allocation:

1. First tries to use the specified port
2. If the port is in use, searches for an available port in the range 8000-9000
3. Raises a RuntimeError if no ports are available

You can customize the port range:

```python
agent_info = tool.invoke({
    "agent_obj": agent,
    "name": "my_agent",
    "port": 8080,  # Preferred port
    "start_range": 8000,  # Start of port range
    "end_range": 9000,    # End of port range
    "description": "A useful agent",
    "api_token": API_TOKEN
})
```

### Cleanup

Always clean up your uAgent when done:

```python
from langchain_uagents import cleanup_uagent

# Clean up by agent name
cleanup_uagent("my_agent")
```

## Environment Variables

The tool requires the following environment variables:

- `AV_API_KEY`: Your [Agentverse](https://agentverse.ai/) API key for registering agents
- `OPENAI_API_KEY`: Your [OpenAI API key](https://platform.openai.com/api-keys) for the Langchain agent

You can set these in a `.env` file or export them in your environment:

```bash
export AV_API_KEY="your_agentverse_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

