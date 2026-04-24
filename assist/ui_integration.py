"""
UI Integration Module

This module provides UI components for displaying exception information
to users in the agent framework.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import our recovery modules
from assist.failure_recovery import FailureRecoveryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UIIntegration:
    """Manages UI components for displaying exception information to users"""
    
    def __init__(self, recovery_storage_path: str = "./recovery_data"):
        self.recovery_manager = FailureRecoveryManager(recovery_storage_path)
        logger.info("Initialized UIIntegration")
    
    def format_exception_for_display(self, recovery_id: str) -> Optional[Dict[str, Any]]:
        """
        Format exception information for UI display
        
        Args:
            recovery_id: The ID of the recovery record to format
            
        Returns:
            Formatted exception information for UI display, or None if not found
        """
        try:
            # Retrieve the recovery context
            recovery_context = self.recovery_manager.get_recovery_context(recovery_id)
            if not recovery_context:
                logger.warning(f"No recovery context found for ID: {recovery_id}")
                return None
            
            # Format the information for UI display
            formatted_info = {
                "recovery_id": recovery_id,
                "thread_id": recovery_context.thread_id,
                "error_type": recovery_context.error_type,
                "error_message": recovery_context.error_message,
                "traceback_info": recovery_context.traceback_info,
                "timestamp": recovery_context.timestamp.isoformat(),
                "conversation_summary": self._summarize_conversation(recovery_context.conversation_history)
            }
            
            logger.info(f"Formatted exception info for UI display: {recovery_id}")
            return formatted_info
            
        except Exception as e:
            logger.error(f"Error formatting exception for UI: {e}")
            return None
    
    def _summarize_conversation(self, conversation_history: list) -> str:
        """
        Create a summary of the conversation for context
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            Summary of the conversation
        """
        if not conversation_history:
            return "No conversation history available"
        
        # Take the last few messages to provide context
        recent_messages = conversation_history[-3:]  # Last 3 messages
        summary_parts = []
        
        for msg in recent_messages:
            summary_parts.append(f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:100]}...")
            
        return "\n".join(summary_parts)

    def display_exception_info(self, recovery_id: str) -> str:
        """
        Generate a user-friendly display string for exception information
        
        Args:
            recovery_id: The ID of the recovery record to display
            
        Returns:
            Formatted string for user display
        """
        formatted_info = self.format_exception_for_display(recovery_id)
        if not formatted_info:
            return "No exception information available"
        
        display_text = f"""
=== FAILURE DETAILS ===
Recovery ID: {formatted_info['recovery_id']}
Thread ID: {formatted_info['thread_id']} 
Error Type: {formatted_info['error_type']}
Error Message: {formatted_info['error_message']}
Timestamp: {formatted_info['timestamp']}

=== CONVERSATION CONTEXT ===
{formatted_info['conversation_summary']}

=== TECHNICAL DETAILS ===
Full Traceback:
{formatted_info['traceback_info'][:500]}...
        """
        
        return display_text

# Global instance
ui_integration = UIIntegration()

# Export for use in other modules
__all__ = [
    "UIIntegration",
    "ui_integration"
]