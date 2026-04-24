"""
UI Integration Module for Failure Recovery

This module implements UI components for displaying exception information
to users in the agent framework.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import our recovery modules
from assist.failure_recovery import FailureRecoveryManager, RecoveryContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExceptionDisplayInfo:
    """Data class to hold information for displaying exceptions to users"""
    thread_id: str
    error_type: str
    error_message: str
    traceback_info: str
    timestamp: datetime
    recovery_id: str
    
    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback_info": self.traceback_info,
            "timestamp": self.timestamp.isoformat(),
            "recovery_id": self.recovery_id
        }

class UIDisplayManager:
    """Manages UI components for displaying exception information to users"""
    
    def __init__(self):
        self.recovery_manager = FailureRecoveryManager()
        logger.info("Initialized UIDisplayManager")
    
    def format_exception_for_display(self, exception_info: ExceptionDisplayInfo) -> str:
        """
        Format exception information for display in the UI
        
        Args:
            exception_info: Information about the exception to display
            
        Returns:
            Formatted string ready for UI display
        """
        display_text = f"""
=== EXCEPTION DETECTED ===

Thread ID: {exception_info.thread_id}
Error Type: {exception_info.error_type}
Error Message: {exception_info.error_message}
Timestamp: {exception_info.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Recovery ID: {exception_info.recovery_id}

=== TRACEBACK ===
{exception_info.traceback_info}
        """
        return display_text.strip()
    
    def display_exception_to_user(self, exception_info: ExceptionDisplayInfo) -> str:
        """
        Display exception information to the user in the UI
        
        Args:
            exception_info: Information about the exception to display
            
        Returns:
            Formatted display string for the UI
        """
        formatted_info = self.format_exception_for_display(exception_info)
        logger.info(f"Displaying exception to user:\n{formatted_info}")
        return formatted_info
    
    def get_exception_details(self, recovery_id: str) -> Optional[RecoveryContext]:
        """
        Retrieve detailed exception information for a recovery ID
        
        Args:
            recovery_id: The recovery ID to retrieve details for
            
        Returns:
            RecoveryContext object with detailed exception information, or None
        """
        return self.recovery_manager.get_recovery_context(recovery_id)

# Global instance
ui_display_manager = UIDisplayManager()

def display_exception_to_user(thread_id: str, error_type: str, error_message: str, 
                             traceback_info: str, recovery_id: str) -> str:
    """
    Main function to display exception information to the user
    
    Args:
        thread_id: The thread ID where the exception occurred
        error_type: The type of exception
        error_message: The error message
        traceback_info: Full traceback information
        recovery_id: The recovery ID for tracking
        
    Returns:
        Formatted string for UI display
    """
    exception_info = ExceptionDisplayInfo(
        thread_id=thread_id,
        error_type=error_type,
        error_message=error_message,
        traceback_info=traceback_info,
        timestamp=datetime.now(),
        recovery_id=recovery_id
    )
    
    return ui_display_manager.display_exception_to_user(exception_info)

# Export for use in other modules
__all__ = [
    "ExceptionDisplayInfo",
    "UIDisplayManager",
    "ui_display_manager",
    "display_exception_to_user"
]