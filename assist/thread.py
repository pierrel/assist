"""
Thread Management Module

This module handles thread management for the agent framework, including
conversation history management for failure recovery purposes.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Import our recovery modules
from assist.failure_recovery import handle_exception
from assist.background_recovery import background_recovery_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Represents a message in a conversation thread"""
    id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

@dataclass
class Thread:
    """Represents a conversation thread with history"""
    id: str
    title: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Thread':
        return cls(
            id=data["id"],
            title=data["title"],
            messages=[Message.from_dict(msg) for msg in data["messages"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data["metadata"]
        )

class ThreadManager:
    """Manages conversation threads for the agent framework"""
    
    def __init__(self, storage_path: str = "./threads"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.threads: Dict[str, Thread] = {}
        logger.info(f"Initialized ThreadManager at {self.storage_path}")
    
    def create_thread(self, title: str, initial_messages: List[Message] = None, 
                     metadata: Dict[str, Any] = None) -> Thread:
        """Create a new conversation thread"""
        thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        
        if initial_messages is None:
            initial_messages = []
            
        if metadata is None:
            metadata = {}
            
        thread = Thread(
            id=thread_id,
            title=title,
            messages=initial_messages,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata
        )
        
        self.threads[thread_id] = thread
        self.save_thread(thread)
        logger.info(f"Created new thread: {thread_id}")
        return thread
    
    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get a thread by ID"""
        if thread_id in self.threads:
            return self.threads[thread_id]
        return None
    
    def save_thread(self, thread: Thread):
        """Save a thread to persistent storage"""
        thread_file = self.storage_path / f"{thread.id}.json"
        with open(thread_file, 'w') as f:
            json.dump(thread.to_dict(), f, indent=2)
        logger.info(f"Saved thread: {thread.id}")
    
    def delete_thread(self, thread_id: str):
        """Delete a thread"""
        if thread_id in self.threads:
            del self.threads[thread_id]
            thread_file = self.storage_path / f"{thread_id}.json"
            if thread_file.exists():
                thread_file.unlink()
            logger.info(f"Deleted thread: {thread_id}")
    
    def add_message_to_thread(self, thread_id: str, message: Message) -> bool:
        """Add a message to a thread"""
        thread = self.get_thread(thread_id)
        if not thread:
            logger.error(f"Thread {thread_id} not found")
            return False
            
        thread.messages.append(message)
        thread.updated_at = datetime.now()
        self.save_thread(thread)
        logger.info(f"Added message to thread: {thread_id}")
        return True
    
    def get_conversation_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a thread"""
        thread = self.get_thread(thread_id)
        if not thread:
            return []
        return [msg.to_dict() for msg in thread.messages]

# Global instance
thread_manager = ThreadManager()

def handle_thread_exception(thread_id: str, error_type: str, error_message: str, 
                           traceback_info: str, conversation_history: list) -> str:
    """
    Handle an exception that occurred within a thread
    
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
    
    # Record the exception through our recovery system
    recovery_id = handle_exception(
        thread_id, error_type, error_message, traceback_info, conversation_history
    )
    
    # In a real implementation, we might also:
    # 1. Trigger background recovery
    # 2. Notify the user
    # 3. Log additional context
    
    logger.info(f"Exception handled for thread {thread_id}, recovery ID: {recovery_id}")
    return recovery_id

# Export for use in other modules
__all__ = [
    "Message",
    "Thread",
    "ThreadManager",
    "thread_manager",
    "handle_thread_exception"
]