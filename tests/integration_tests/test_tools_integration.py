"""Integration tests for UAgentRegisterTool."""

from typing import Type
import os

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_uagent_register.tools import UAgentRegisterTool


class TestUAgentRegisterToolIntegration(ToolsIntegrationTests):
    """Integration tests for UAgentRegisterTool."""

    @property
    def tool_constructor(self) -> Type[UAgentRegisterTool]:
        """Return the constructor for the tool."""
        return UAgentRegisterTool

    @property
    def tool_constructor_params(self) -> dict:
        """Return parameters for initializing the tool."""
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """Return an example of the parameters to pass to the tool's invoke method."""
        return {
            "agent_obj": "langchain_agent_object",  # Test mode agent
            "name": "test_integration_agent",
            "port": 8090,
            "description": "Test agent for integration testing",
            "api_token": os.getenv("AV_API_KEY")  # Use the API key from environment if available
        } 