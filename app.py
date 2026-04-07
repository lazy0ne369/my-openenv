from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import uvicorn
import os

from src.environment import APIValidatorEnv
from src.models import Action, Observation, Reward

app = FastAPI(
    title="API Validator Environment",
    description="OpenEnv environment for API request validation and routing",
    version="1.0.0"
)

# Store environment instances per session
environments: Dict[str, APIValidatorEnv] = {}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "name": "API Validator Environment",
        "version": "1.0.0",
        "status": "running",
        "tasks": ["easy", "medium", "hard"]
    }


@app.post("/reset")
async def reset(task_id: str = "easy", session_id: str = "default") -> Dict[str, Any]:
    """
    Reset the environment for a new episode.
    
    Args:
        task_id: Task to run (easy/medium/hard)
        session_id: Session identifier
        
    Returns:
        Initial observation
    """
    try:
        env = APIValidatorEnv(task_id=task_id)
        environments[session_id] = env
        
        observation = env.reset()
        
        return {
            "observation": observation.model_dump(),
            "session_id": session_id,
            "task_id": task_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action: Action, session_id: str = "default") -> Dict[str, Any]:
    """
    Take a step in the environment.
    
    Args:
        action: Agent's action
        session_id: Session identifier
        
    Returns:
        observation, reward, done, info
    """
    if session_id not in environments:
        raise HTTPException(
            status_code=400,
            detail="No active environment. Call /reset first."
        )
    
    try:
        env = environments[session_id]
        observation, reward, done, info = env.step(action)
        
        return {
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state(session_id: str = "default") -> Dict[str, Any]:
    """
    Get current environment state.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Current state
    """
    if session_id not in environments:
        raise HTTPException(
            status_code=400,
            detail="No active environment. Call /reset first."
        )
    
    try:
        env = environments[session_id]
        state = env.state()
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def list_tasks():
    """List available tasks"""
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Simple Schema Validation",
                "description": "Validate user registration requests against a fixed schema",
                "difficulty": "easy"
            },
            {
                "id": "medium",
                "name": "Multi-Endpoint Routing",
                "description": "Route requests to correct endpoints and validate schemas",
                "difficulty": "medium"
            },
            {
                "id": "hard",
                "name": "Malformed Request Recovery",
                "description": "Attempt to fix or reject corrupted/ambiguous requests",
                "difficulty": "hard"
            }
        ]
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)