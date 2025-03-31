"""Unit tests for UAgentRegisterTool."""

from typing import Type
from unittest.mock import MagicMock, patch

import pytest
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_uagent_register.tools import UAgentRegisterTool


class TestUAgentRegisterToolUnit(ToolsUnitTests):
    """Unit tests for UAgentRegisterTool."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, request: pytest.FixtureRequest) -> MagicMock:
        """Set up mocks for testing."""
        # Patch the Agent creation to avoid actual networking
        patcher = patch("langchain_uagent_register.tools.Agent")
        mock_agent = patcher.start()
        
        # Mock the agent instances
        mock_agent_instance = MagicMock()
        mock_agent_instance.address = "agent1mock0123456789xxxxxx"
        mock_agent.return_value = mock_agent_instance
        
        # Patch the thread handling
        thread_patcher = patch("langchain_uagent_register.tools.threading.Thread")
        mock_thread = thread_patcher.start()
        
        # Patch socket for port finding
        socket_patcher = patch("langchain_uagent_register.tools.socket.socket")
        mock_socket = socket_patcher.start()
        mock_socket.return_value.__enter__.return_value.bind.return_value = None
        
        # Use pytest's cleanup mechanism
        request.addfinalizer(patcher.stop)
        request.addfinalizer(thread_patcher.stop)
        request.addfinalizer(socket_patcher.stop)
        
        # Return the mocks in case we need them in tests
        return {
            "agent": mock_agent,
            "thread": mock_thread,
            "socket": mock_socket
        }

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
            "name": "test_agent",
            "port": 8080,
            "description": "Test agent for unit testing",
            "api_token": "test_api_token"
        }

    def test_find_available_port(self, setup_mocks):
        """Test port finding functionality."""
        tool = UAgentRegisterTool()
        
        # Test with preferred port
        port = tool._find_available_port(preferred_port=8080)
        assert port == 8080
        
        # Test with port in use for preferred port, but first port in range available
        mock_socket = setup_mocks["socket"].return_value.__enter__.return_value
        mock_socket.bind.side_effect = [
            OSError(),  # First call (preferred port) fails
            None,      # Second call (port 8000) succeeds
        ]
        port = tool._find_available_port(preferred_port=8080, start_range=8000, end_range=8002)
        assert port == 8000
        
        # Reset side effect for next test
        mock_socket.bind.reset_mock()
        
        # Test with no available ports
        mock_socket.bind.side_effect = OSError()  # All ports fail
        with pytest.raises(RuntimeError, match="Could not find an available port in range 8000-8001"):
            tool._find_available_port(preferred_port=8080, start_range=8000, end_range=8001)

    @pytest.mark.asyncio
    async def test_arun(self, setup_mocks):
        """Test async version of the tool."""
        tool = UAgentRegisterTool()
        result = await tool._arun(**self.tool_invoke_params_example)
        
        assert result["name"] == "test_agent"
        assert result["port"] == 8080
        assert result["description"] == "Test agent for unit testing"
        assert result["api_token"] == "test_api_token"
        assert "address" in result
        assert result["test_mode"] is True

    def test_agent_with_ai_address(self, setup_mocks):
        """Test agent creation with AI agent address."""
        params = self.tool_invoke_params_example.copy()
        params["ai_agent_address"] = "agent1test123"
        
        tool = UAgentRegisterTool()
        result = tool.invoke(params)
        
        assert result["ai_agent_address"] == "agent1test123"

    def test_agent_info_storage(self, setup_mocks):
        """Test agent information storage."""
        tool = UAgentRegisterTool()
        result = tool.invoke(self.tool_invoke_params_example)
        
        # Check stored agent info
        assert tool.get_agent_info() is not None
        assert tool.get_agent_info()["name"] == "test_agent"
        assert tool.get_agent_info()["port"] == 8080 