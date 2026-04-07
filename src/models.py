from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal


class Observation(BaseModel):
    """
    What the agent sees at each step.
    Contains the incoming request and available routing information.
    """
    request_payload: Dict[str, Any] = Field(
        description="The incoming API request body"
    )
    request_metadata: Dict[str, Any] = Field(
        description="Metadata about the request (timestamp, source, headers, etc.)"
    )
    available_endpoints: List[str] = Field(
        description="List of available backend endpoints"
    )
    endpoint_schemas: Dict[str, Dict[str, Any]] = Field(
        description="Schema definitions for each endpoint"
    )
    step_count: int = Field(
        description="Current step number in the episode",
        default=0
    )
    previous_feedback: Optional[str] = Field(
        description="Feedback from previous action (if any)",
        default=None
    )


class Action(BaseModel):
    """
    Agent's decision on how to handle the request.
    """
    action_type: Literal["validate", "route", "reject", "fix"] = Field(
        description="Type of action to take"
    )
    target_endpoint: Optional[str] = Field(
        description="Endpoint to route to (required for 'route' action)",
        default=None
    )
    validation_issues: Optional[List[str]] = Field(
        description="List of validation issues found",
        default=None
    )
    fix_suggestion: Optional[Dict[str, Any]] = Field(
        description="Suggested fixes for malformed request",
        default=None
    )
    confidence: float = Field(
        description="Confidence in this decision (0.0-1.0)",
        ge=0.0,
        le=1.0,
        default=1.0
    )
    reasoning: str = Field(
        description="Explanation of why this action was chosen"
    )


class Reward(BaseModel):
    """
    Reward signal for the agent's action.
    """
    score: float = Field(
        description="Reward value between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    feedback: str = Field(
        description="Human-readable feedback on the action"
    )
    breakdown: Dict[str, float] = Field(
        description="Component breakdown of the reward",
        default_factory=dict
    )