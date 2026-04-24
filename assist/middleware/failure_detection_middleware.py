"""
Middleware for detecting and handling agent execution failures
"""

import logging
from typing import Any, Dict, List, Optional, Union
from functools import wraps

from langchain_core.messages import AIMessage
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.runnables import Runnable

from assist.recovery import handle_exception, get_recovery_system

logger = logging.getLogger(__name__)

class FailureDetectionMiddleware(AgentMiddleware):
    """Middleware to detect and handle agent execution failures"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovery_system = get_recovery_system()
        self.logger = logging.getLogger(__name__)
        
    def _get_conversation_history(self, state: AgentState) -> list:
        """Extract conversation history from agent state"""
        if hasattr(state, 'values') and 'messages' in state.values:
            return state.values['messages']
        return []
    
    def _handle_exception(self, thread_id: str, exception: Exception, state: AgentState):
        """Handle an exception by triggering recovery process"""
        try:
            conversation_history = self._get_conversation_history(state)
            recovery_process_id = handle_exception(thread_id, exception, conversation_history)
            logger.info(f"Exception detected in thread {thread_id}. Recovery process started: {recovery_process_id}")
            return recovery_process_id
        except Exception as e:
            logger.error(f"Failed to initiate recovery process: {e}")
            raise
    
    def process(self, state: AgentState, request: ModelRequest) -> Union[ModelResponse, Exception]:
        """Process the agent execution with failure detection"""
        try:
            # Get thread ID from state
            thread_id = self._get_thread_id(state)
            
            # Execute the agent normally
            result = self.next.process(state, request)
            
            # If successful, return the result
            return result
            
        except Exception as e:
            # Handle the exception
            logger.error(f"Agent execution failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            
            # Trigger recovery process
            recovery_process_id = self._handle_exception(thread_id, e, state)
            
            # Return a generic error response to the user
            error_msg = f"An error occurred during processing. Recovery process initiated: {recovery_process_id}"
            return AIMessage(content=error_msg)
    
    def _get_thread_id(self, state: AgentState) -> str:
        """Extract thread ID from agent state"""
        try:
            # Try to get thread ID from config
            if hasattr(state, 'config') and 'configurable' in state.config:
                return state.config['configurable'].get('thread_id', 'unknown')
            elif hasattr(state, 'values'):
                # Check if thread_id is in values
                return state.values.get('thread_id', 'unknown')
            return 'unknown'
        except Exception:
            return 'unknown'

# Export the middleware for use in the agent system
__all__ = ['FailureDetectionMiddleware']