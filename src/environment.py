from typing import Dict, Any, Tuple
from src.models import Observation, Action, Reward
from tasks.task_definitions import TASKS, TaskDefinition
import random


class APIValidatorEnv:
    """
    OpenEnv environment for API request validation and routing.
    """
    
    def __init__(self, task_id: str = "easy"):
        """
        Initialize the environment.
        
        Args:
            task_id: Which task to run ("easy", "medium", or "hard")
        """
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")
        
        self.task_id = task_id
        self.task: TaskDefinition = TASKS[task_id]
        
        # Episode state
        self.current_request = None
        self.ground_truth = None
        self.step_count = 0
        self.max_steps = 5  # Max steps per episode
        self.done = False
        self.episode_reward = 0.0
        self.previous_feedback = None
        
        # Endpoint definitions (for observation)
        self.available_endpoints = self._get_available_endpoints()
        self.endpoint_schemas = self._get_endpoint_schemas()
    
    def _get_available_endpoints(self) -> list:
        """Get list of available endpoints based on task"""
        if self.task_id == "easy":
            return ["user_registration"]
        elif self.task_id == "medium":
            return ["user_registration", "product_create", "order_submit", 
                    "payment_process", "support_ticket"]
        else:  # hard
            return ["user_registration", "product_create", "payment_process"]
    
    def _get_endpoint_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schema definitions for endpoints"""
        schemas = {
            "user_registration": {
                "required": ["username", "email", "age"],
                "optional": ["phone"],
                "types": {
                    "username": "string",
                    "email": "string (email format)",
                    "age": "integer (13-120)",
                    "phone": "string"
                }
            },
            "product_create": {
                "required": ["name", "price", "category"],
                "optional": ["description", "stock"],
                "types": {
                    "name": "string",
                    "price": "number",
                    "category": "string",
                    "description": "string",
                    "stock": "integer"
                }
            },
            "order_submit": {
                "required": ["user_id", "items", "total"],
                "optional": ["shipping_address", "coupon"],
                "types": {
                    "user_id": "string",
                    "items": "array",
                    "total": "number",
                    "shipping_address": "string",
                    "coupon": "string"
                }
            },
            "payment_process": {
                "required": ["order_id", "amount", "method"],
                "optional": ["card_last4"],
                "types": {
                    "order_id": "string",
                    "amount": "number",
                    "method": "string",
                    "card_last4": "string"
                }
            },
            "support_ticket": {
                "required": ["user_id", "subject", "priority"],
                "optional": ["description", "attachment"],
                "types": {
                    "user_id": "string",
                    "subject": "string",
                    "priority": "string (low/medium/high)",
                    "description": "string",
                    "attachment": "string"
                }
            }
        }
        
        # Return only schemas for available endpoints
        return {k: v for k, v in schemas.items() if k in self.available_endpoints}
    
    def reset(self) -> Observation:
        """
        Reset the environment to start a new episode.
        
        Returns:
            Initial observation
        """
        # Generate new request
        self.current_request, self.ground_truth = self.task.generate_request()
        
        # Reset episode state
        self.step_count = 0
        self.done = False
        self.episode_reward = 0.0
        self.previous_feedback = None
        
        # Create initial observation
        observation = Observation(
            request_payload=self.current_request,
            request_metadata={
                "timestamp": "2024-01-15T10:30:00Z",
                "source_ip": "192.168.1.100",
                "user_agent": "APIClient/1.0",
                "request_id": f"req_{random.randint(1000, 9999)}"
            },
            available_endpoints=self.available_endpoints,
            endpoint_schemas=self.endpoint_schemas,
            step_count=self.step_count,
            previous_feedback=None
        )
        
        return observation
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The agent's action
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        self.step_count += 1
        
        # Grade the action
        score, feedback, breakdown = self.task.grade(action, self.ground_truth)
        
        reward = Reward(
            score=score,
            feedback=feedback,
            breakdown=breakdown
        )
        
        self.episode_reward += score
        
        # Ensure episode_reward does not violate strict bounds (0, 1)
        from tasks.task_definitions import strict_unit_score
        self.episode_reward = strict_unit_score(self.episode_reward)
        self.previous_feedback = feedback
        
        # Episode ends if:
        # 1. Final decision made (route/reject)
        # 2. Max steps reached
        terminal_actions = ["route", "reject"]
        if action.action_type in terminal_actions or self.step_count >= self.max_steps:
            self.done = True
        
        # Create next observation
        observation = Observation(
            request_payload=self.current_request,
            request_metadata={
                "timestamp": "2024-01-15T10:30:00Z",
                "source_ip": "192.168.1.100",
                "user_agent": "APIClient/1.0",
                "request_id": f"req_{random.randint(1000, 9999)}"
            },
            available_endpoints=self.available_endpoints,
            endpoint_schemas=self.endpoint_schemas,
            step_count=self.step_count,
            previous_feedback=self.previous_feedback
        )
        
        # Info dict
        info = {
            "task_id": self.task_id,
            "ground_truth": self.ground_truth,
            "episode_reward": self.episode_reward,
            "action_type": action.action_type
        }
        
        return observation, reward, self.done, info
    
    def state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.
        
        Returns:
            Dictionary containing current state
        """
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "done": self.done,
            "episode_reward": self.episode_reward,
            "current_request": self.current_request,
            "previous_feedback": self.previous_feedback,
            "available_endpoints": self.available_endpoints
        }