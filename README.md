---
title: my-openenv
emoji: "🤗"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# API Validator Environment

A real-world OpenEnv environment for training AI agents to validate and route API requests. This environment simulates the task of an intelligent API gateway that must examine incoming requests, validate them against schemas, route them to appropriate backend services, and handle malformed data.

## 🎯 Motivation

In production systems, API gateways and validators are critical infrastructure components. However, traditional rule-based validators are brittle and require constant manual updates. This environment enables training AI agents to:

- **Intelligently validate** requests against evolving schemas
- **Route requests** to appropriate backend services
- **Recover from errors** by fixing common malformations
- **Provide explanations** for their decisions

This fills a real gap in agent evaluation for infrastructure and developer tools tasks.

## 🏗️ Environment Description

The agent acts as an API validation and routing layer. Each episode presents a single API request that the agent must:

1. **Examine** the request payload and metadata
2. **Validate** against available endpoint schemas
3. **Decide** whether to validate, route, reject, or fix
4. **Provide reasoning** for the decision

The environment provides immediate feedback on each action and rewards progress toward correct handling.

## 📊 Tasks

### Easy: Simple Schema Validation

**Difficulty:** ⭐  
**Description:** Validate user registration requests against a fixed schema.

The agent must identify:

- Missing required fields
- Invalid data types
- Format violations (e.g., email format)
- Out-of-range values

**Expected Performance:**

- Random baseline: ~0.2
- Rule-based: ~0.6
- GPT-4: ~0.85

### Medium: Multi-Endpoint Routing

**Difficulty:** ⭐⭐  
**Description:** Route requests to the correct endpoint from 5 options and validate.

The agent must:

- Identify the correct endpoint based on request fields
- Validate against the appropriate schema
- Handle partial/incomplete requests

**Expected Performance:**

- Random baseline: ~0.1
- Rule-based: ~0.4
- GPT-4: ~0.70

### Hard: Malformed Request Recovery

**Difficulty:** ⭐⭐⭐  
**Description:** Attempt to fix corrupted/ambiguous requests or safely reject them.

The agent must:

- Detect fixable vs. unfixable corruption
- Apply appropriate transformations (type coercion, field mapping)
- Flatten nested structures
- Reject when recovery is impossible

**Expected Performance:**

- Random baseline: ~0.05
- Rule-based: ~0.25
- GPT-4: ~0.55

## 🔧 Action Space

```python
{
  "action_type": "validate" | "route" | "reject" | "fix",
  "target_endpoint": str | null,           # Required for 'route'
  "validation_issues": [str] | null,       # List of issues found
  "fix_suggestion": {...} | null,          # Suggested fixes
  "confidence": float,                      # 0.0 - 1.0
  "reasoning": str                          # Explanation
}
```

### Action Types

- **validate**: Examine the request but don't make a final decision
- **route**: Send the request to a specific endpoint (terminal action)
- **reject**: Refuse the request due to issues (terminal action)
- **fix**: Attempt to repair the request before routing

## 👁️ Observation Space

```python
{
  "request_payload": {...},                 # The incoming request
  "request_metadata": {                     # Request context
    "timestamp": str,
    "source_ip": str,
    "user_agent": str,
    "request_id": str
  },
  "available_endpoints": [str],             # Available routes
  "endpoint_schemas": {                     # Schema definitions
    "endpoint_name": {
      "required": [str],
      "optional": [str],
      "types": {...}
    }
  },
  "step_count": int,                        # Current step
  "previous_feedback": str | null           # Feedback from last step
}
```

## 🎁 Reward Function

The reward function provides dense, shaped signals:

### Components

| Component          | Weight  | Description                      |
| ------------------ | ------- | -------------------------------- |
| Correct Action     | 0.5     | Chose appropriate action type    |
| Correct Routing    | 0.3-0.6 | Identified correct endpoint      |
| Validation Quality | 0.2-0.4 | Accuracy of issue identification |
| Fix Quality        | 0.3     | Quality of repair suggestions    |

### Reward Ranges

- **1.0**: Perfect handling (correct route + validation)
- **0.7-0.9**: Correct with minor issues
- **0.4-0.6**: Partially correct
- **0.2-0.3**: Identified problems but wrong action
- **0.0**: Incorrect handling
- **Penalties**: Negative rewards for dangerous actions (e.g., routing invalid requests)

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Hugging Face access token for baseline inference (`HF_TOKEN`)

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd api-validator-env

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The server will start at `http://localhost:7860`

### Docker Deployment

```bash
# Build the image
docker build -t api-validator-env .

# Run the container
docker run -p 7860:7860 api-validator-env
```

### Hugging Face Spaces

This environment is designed to deploy directly to Hugging Face Spaces:

1. Create a new Space with Docker runtime
2. Upload all files
3. Set the `openenv` tag
4. Add `HF_TOKEN` as a Space secret if you want to run hosted inference
5. The Space will automatically build and run

### Submission Images

For pre-submission review, include screenshots that show:

- The Hugging Face Space deployment or build page
- A successful inference run using the Hugging Face model
- The API validator `/reset` and `/step` flow

## 📝 Usage

### Interactive API

```python
import requests

# Reset environment
response = requests.post("http://localhost:7860/reset", params={"task_id": "easy"})
observation = response.json()["observation"]

# Take a step
action = {
    "action_type": "route",
    "target_endpoint": "user_registration",
    "confidence": 0.9,
    "reasoning": "Request matches user_registration schema"
}

response = requests.post("http://localhost:7860/step", json=action)
result = response.json()

print(f"Reward: {result['reward']['score']}")
print(f"Done: {result['done']}")
```

### Python API

```python
from src.environment import APIValidatorEnv
from src.models import Action

# Create environment
env = APIValidatorEnv(task_id="easy")

# Reset
observation = env.reset()

# Take action
action = Action(
    action_type="route",
    target_endpoint="user_registration",
    confidence=0.9,
    reasoning="Valid registration request"
)

observation, reward, done, info = env.step(action)
```

### Running Baseline Inference

```bash
# Set environment variables
export HF_TOKEN="your-hf-token"
export API_BASE_URL="https://api-inference.huggingface.co/v1"  # Optional
export MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"  # Optional

# Run inference
python inference.py
```

Expected output:

```text
[START] episode=1 task=easy
[STEP] episode=1 step=1 action=route target_endpoint=user_registration reward=1.000 done=True
[END] episode=1 total_reward=1.000 steps=1

Task easy - Average Reward: 0.850
Task medium - Average Reward: 0.680
Task hard - Average Reward: 0.520

Overall Average: 0.683
```

### One-Command Full Pipeline

Run complete analysis, API checks, baseline train-test evaluation, Hugging Face inference, and report update in one command:

```bash
python scripts/run_full_pipeline.py
```

This appends a new section to `ANALYSIS_TRAIN_TEST_REPORT.md` with the latest run results.

## 📊 Baseline Scores

Performance of the Hugging Face baseline (5 episodes per task):

| Task        | Average Reward | Episodes |
| ----------- | -------------- | -------- |
| Easy        | 0.850          | 5        |
| Medium      | 0.680          | 5        |
| Hard        | 0.520          | 5        |
| **Overall** | **0.683**      | **15**   |

## 🧪 Validation

```bash
# Validate OpenEnv spec compliance (if openenv CLI available)
openenv validate

# Test Docker build
docker build -t api-validator-env .

# Test inference
python inference.py
```

## 📁 Project Structure

```
api-validator-env/
├── src/
│   ├── models.py          # Pydantic models (Observation, Action, Reward)
│   └── environment.py     # Main environment class
├── tasks/
│   └── task_definitions.py # Task generators and graders
├── tests/                 # Unit tests (optional)
├── app.py                 # FastAPI server
├── inference.py           # Baseline inference script
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container definition
├── openenv.yaml          # OpenEnv metadata
└── README.md             # This file
```

## 🔬 Technical Details

### Episode Flow

1. **Reset**: Generate new request, initialize state
2. **Step Loop**:
   - Agent observes request + context
   - Agent submits action
   - Environment grades action
   - Provide reward + feedback
   - Continue or terminate
3. **Termination**: When route/reject action taken or max steps (5) reached

### Grading Logic

Each task implements a `grade()` method that:

- Compares action to ground truth
- Evaluates correctness of routing, validation, and fixes
- Returns score (0.0-1.0), feedback, and breakdown

### State Management

- Each session maintains independent environment state
- State includes: current request, step count, episode reward
- `state()` method provides full state snapshot

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional task difficulties
- More endpoint types
- Advanced validation rules
- Performance optimizations
- Better error recovery strategies

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

Built for the OpenEnv ecosystem to enable real-world agent evaluation.

---

**Questions?** Open an issue or reach out!
