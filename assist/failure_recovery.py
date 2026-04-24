"""
Core Failure Recovery Module

This module implements the core functionality for failure recovery in the agent framework,
handling both automatic exception recovery and user-driven response improvement.
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RecoveryContext:
    """Data class to hold context for recovery operations"""
    thread_id: str
    error_type: str
    error_message: str
    traceback_info: str
    conversation_history: list
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback_info": self.traceback_info,
            "conversation_history": self.conversation_history,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RecoveryContext':
        return cls(
            thread_id=data["thread_id"],
            error_type=data["error_type"],
            error_message=data["error_message"],
            traceback_info=data["traceback_info"],
            conversation_history=data["conversation_history"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

class FailureRecoveryManager:
    """Main manager for handling failure recovery operations"""
    
    def __init__(self, recovery_storage_path: str = "./recovery_data"):
        self.recovery_storage_path = Path(recovery_storage_path)
        self.recovery_storage_path.mkdir(exist_ok=True)
        logger.info(f"Initialized FailureRecoveryManager at {self.recoverage_storage_path}")
    
    def record_exception(self, thread_id: str, error_type: str, error_message: str, 
                       traceback_info: str, conversation_history: list) -> str:
        """Record an exception for potential recovery"""
        context = RecoveryContext(
            thread_id=thread_id,
            error_type=error_type,
            error_message=error_message,
            traceback_info=traceback_info,
            conversation_history=conversation_history,
            timestamp=datetime.now()
        )
        
        # Generate a unique recovery ID
        recovery_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Store recovery data
        recovery_file = self.recovery_storage_path / f"{recovery_id}.json"
        with open(recovery_file, 'w') as f:
            json.dump(context.to_dict(), f, indent=2)
        
        logger.info(f"Recorded exception for recovery: {recovery_id}")
        return recovery_id
    
    def get_recovery_context(self, recovery_id: str) -> Optional[RecoveryContext]:
        """Retrieve recovery context by ID"""
        recovery_file = self.recovery_storage_path / f"{recovery_id}.json"
        if recovery_file.exists():
            with open(recovery_file, 'r') as f:
                data = json.load(f)
            return RecoveryContext.from_dict(data)
        return None
    
    def cleanup_recovery_data(self):
        """Clean up old recovery data"""
        # This would be implemented to periodically clean up old recovery records
        pass

# Global instance
failure_recovery_manager = FailureRecoveryManager()

def handle_exception(thread_id: str, error_type: str, error_message: str, 
                    traceback_info: str, conversation_history: list) -> str:
    """
    Handle an exception by recording it and initiating recovery
    
    Args:
        thread_id: The ID of the thread where the exception occurred
        error_type: The type of exception
        error_message: The error message
        traceback_info: Full traceback information
        conversation_history: The conversation history at time of exception
        
    Returns:
        Recovery ID for tracking the recovery process
    """
    logger.error(f"Handling exception in thread {thread_id}: {error_type} - {error_message}")
    
    # Record the exception
    recovery_id = failure_recovery_manager.record_exception(
        thread_id, error_type, error_message, traceback_info, conversation_history
    )
    
    # Here we would typically trigger background recovery
    # For now, we'll just log that recovery was initiated
    logger.info(f"Initiating recovery process for {recovery_id}")
    
    return recovery_id

def handle_user_feedback(feedback: str, thread_id: str, original_response: str) -> str:
    """
    Handle user feedback for improving responses
    
    Args:
        feedback: User's feedback about the response
        thread_id: The thread ID where the response was generated
        original_response: The original response that received feedback
        
    Returns:
        Feedback ID for tracking the improvement process
    """
    # In a real implementation, this would:
    # 1. Store the feedback
    # 2. Potentially trigger retraining or refinement
    # 3. Return a feedback ID for tracking
    
    feedback_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    logger.info(f"Received user feedback for thread {thread_id}: {feedback_id}")
    
    # For demonstration, we'll just log the feedback
    logger.info(f"Feedback: {feedback}")
    logger.info(f"Original response: {original_response}")
    
    return feedback_id

# Export for use in other modules
__all__ = [
    "FailureRecoveryManager",
    "RecoveryContext",
    "handle_exception",
    "handle_user_feedback",
    "failure_recovery_manager"
]