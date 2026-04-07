#!/usr/bin/env python3
"""Baseline inference script for API Validator Environment.

The model client is configured through injected submission environment
variables and uses the OpenAI Python client against the provided API proxy.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Tuple

from openai import OpenAI

from src.environment import APIValidatorEnv
from src.models import Action

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def get_llm_client() -> OpenAI:
    """Initialize an OpenAI-compatible client for the injected API proxy."""

    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API_KEY (or HF_TOKEN/OPENAI_API_KEY) environment variable must be set")

    return OpenAI(api_key=api_key, base_url=get_api_base_url())


def get_model_name() -> str:
    """Get the Hugging Face model name from the environment."""

    return os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


def get_api_base_url() -> str:
    """Get the injected OpenAI-compatible API base URL."""

    return os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)


def build_prompt(observation: Dict[str, Any], task_id: str) -> Tuple[str, str]:
    """Build prompt for the LLM."""

    system_prompt = f"""You are an AI agent tasked with validating and routing API requests.

Task: {task_id}

Your job is to:
1. Examine incoming API requests
2. Validate against the appropriate schema
3. Route valid requests to the correct endpoint
4. Reject or attempt to fix invalid requests

Available endpoints and schemas:
{json.dumps(observation['endpoint_schemas'], indent=2)}

For each request, you must respond with a JSON object containing:
- action_type: "validate", "route", "reject", or "fix"
- target_endpoint: endpoint name (if routing)
- validation_issues: list of issues found (if any)
- fix_suggestion: suggested fixes as dict (if fixing)
- confidence: confidence score 0.0-1.0
- reasoning: explanation of your decision

Respond ONLY with the JSON object, no additional text."""

    user_prompt = f"""Current request to process:

Request payload:
{json.dumps(observation['request_payload'], indent=2)}

Request metadata:
{json.dumps(observation['request_metadata'], indent=2)}

Step count: {observation['step_count']}

Previous feedback: {observation.get('previous_feedback', 'None')}

Analyze this request and provide your action as a JSON object."""

    return system_prompt, user_prompt


def parse_llm_response(response_text: str) -> Action:
    """Parse LLM response into Action object."""

    try:
        response_text = response_text.strip()

        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        data = json.loads(response_text)

        issues = data.get("validation_issues")
        if issues is not None:
            if not isinstance(issues, list):
                issues = [str(issues)]
            else:
                normalized_issues = []
                for item in issues:
                    if isinstance(item, str):
                        normalized_issues.append(item)
                    elif isinstance(item, dict):
                        if "message" in item:
                            normalized_issues.append(str(item["message"]))
                        elif "issue" in item:
                            normalized_issues.append(str(item["issue"]))
                        elif "issue_type" in item:
                            normalized_issues.append(str(item["issue_type"]))
                        else:
                            normalized_issues.append(json.dumps(item))
                    else:
                        normalized_issues.append(str(item))
                issues = normalized_issues

        fix_suggestion = data.get("fix_suggestion")
        if fix_suggestion is not None and not isinstance(fix_suggestion, dict):
            fix_suggestion = {"suggestion": str(fix_suggestion)}

        return Action(
            action_type=data.get("action_type", "validate"),
            target_endpoint=data.get("target_endpoint"),
            validation_issues=issues,
            fix_suggestion=fix_suggestion,
            confidence=data.get("confidence", 1.0),
            reasoning=data.get("reasoning", ""),
        )

    except json.JSONDecodeError as exc:
        return Action(
            action_type="validate",
            confidence=0.5,
            reasoning=f"Failed to parse LLM response: {exc}",
        )
    except Exception as exc:
        return Action(
            action_type="reject",
            confidence=0.4,
            reasoning=f"Failed to build action from model response: {exc}",
        )


def run_episode(env: APIValidatorEnv, client: OpenAI, model_name: str, task_id: str, episode_num: int) -> float:
    """Run a single episode and return total reward."""

    observation = env.reset()
    done = False
    total_reward = 0.0
    step_num = 0

    print(f"[START] episode={episode_num} task={task_id}")

    while not done:
        step_num += 1

        obs_dict = observation if isinstance(observation, dict) else observation.model_dump()
        system_prompt, user_prompt = build_prompt(obs_dict, task_id)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            llm_response = response.choices[0].message.content or ""
        except Exception as exc:
            print(f"[ERROR] step={step_num} error={exc}", file=sys.stderr)
            llm_response = json.dumps(
                {
                    "action_type": "reject",
                    "confidence": 0.1,
                    "reasoning": f"LLM call failed: {exc}",
                }
            )

        action = parse_llm_response(llm_response)
        observation, reward, done, info = env.step(action)
        total_reward += reward.score

        print(
            f"[STEP] episode={episode_num} step={step_num} action={action.action_type} "
            f"target_endpoint={action.target_endpoint} reward={reward.score:.3f} "
            f"done={done} reasoning=\"{action.reasoning}\""
        )

    print(f"[END] episode={episode_num} total_reward={total_reward:.3f} steps={step_num}")
    return total_reward


def main() -> None:
    """Main inference loop."""

    client = get_llm_client()
    model_name = get_model_name()
    api_base = get_api_base_url()

    print("# API Validator Environment - Baseline Inference")
    print("# Provider: huggingface")
    print(f"# Model: {model_name}")
    print(f"# API Base: {api_base}")
    print()

    tasks = ["easy", "medium", "hard"]
    num_episodes_per_task = 5
    results: Dict[str, Dict[str, Any]] = {}

    for task_id in tasks:
        print(f"\n{'='*60}")
        print(f"Running task: {task_id}")
        print(f"{'='*60}\n")

        env = APIValidatorEnv(task_id=task_id)
        task_rewards = []

        for episode_num in range(1, num_episodes_per_task + 1):
            total_reward = run_episode(env, client, model_name, task_id, episode_num)
            task_rewards.append(total_reward)

        avg_reward = sum(task_rewards) / len(task_rewards)
        results[task_id] = {
            "episodes": num_episodes_per_task,
            "rewards": task_rewards,
            "average_reward": avg_reward,
        }

        print(f"\nTask {task_id} - Average Reward: {avg_reward:.3f}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    for task_id, task_results in results.items():
        print(
            f"{task_id:10s} - Avg: {task_results['average_reward']:.3f} "
            f"(episodes: {task_results['episodes']})"
        )

    overall_avg = sum(r["average_reward"] for r in results.values()) / len(results)
    print(f"\nOverall Average: {overall_avg:.3f}")


if __name__ == "__main__":
    main()