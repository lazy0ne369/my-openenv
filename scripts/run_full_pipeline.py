#!/usr/bin/env python3
"""Run full analysis/train-test pipeline and append results to report."""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, parse, request

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "ANALYSIS_TRAIN_TEST_REPORT.md"
INFERENCE_PATH = ROOT / "inference.py"
APP_PATH = ROOT / "app.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import APIValidatorEnv
from src.models import Action


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")


def run_cmd(args: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, method=method, data=data, headers=headers)
    with request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_server() -> Tuple[bool, Optional[Dict[str, Any]], str]:
    try:
        root = http_json("GET", "http://localhost:7860/")
        return True, root, "server reachable"
    except Exception as exc:
        return False, None, str(exc)


def start_server_if_needed() -> Tuple[Optional[subprocess.Popen], str]:
    ok, _, msg = check_server()
    if ok:
        return None, "server already running"

    proc = subprocess.Popen(
        [sys.executable, str(APP_PATH)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(20):
        time.sleep(0.5)
        ok, _, _ = check_server()
        if ok:
            return proc, "server started by pipeline"

    proc.terminate()
    return None, f"failed to start server: {msg}"


def stop_server(proc: Optional[subprocess.Popen]) -> None:
    if not proc:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def run_static_checks() -> Tuple[bool, str]:
    files = [
        "app.py",
        "inference.py",
        "src/models.py",
        "src/environment.py",
        "tasks/task_definitions.py",
    ]
    code, out, err = run_cmd([sys.executable, "-m", "py_compile", *files], cwd=ROOT)
    details = (out + "\n" + err).strip()
    return code == 0, details if details else "py_compile completed with no output"


def run_api_workflow() -> Dict[str, Any]:
    results = []
    for task in ["easy", "medium", "hard"]:
        sid = f"pipeline_{task}_{int(time.time())}"
        reset = http_json("POST", f"http://localhost:7860/reset?task_id={parse.quote(task)}&session_id={parse.quote(sid)}")
        step = http_json(
            "POST",
            f"http://localhost:7860/step?session_id={parse.quote(sid)}",
            payload={
                "action_type": "validate",
                "confidence": 0.9,
                "reasoning": "automated integration check",
            },
        )
        state = http_json("GET", f"http://localhost:7860/state?session_id={parse.quote(sid)}")
        results.append(
            {
                "task": task,
                "reset_ok": bool(reset.get("observation")),
                "step_ok": bool(step.get("reward")),
                "state_ok": bool(state.get("step_count", 0) >= 1),
                "reward": step.get("reward", {}).get("score", None),
                "done": step.get("done", None),
            }
        )
    return {"status": "PASS", "results": results}


def choose_endpoint(observation: Any) -> Optional[str]:
    payload_keys = set(observation.request_payload.keys())
    best_ep = None
    best_score = -1
    for ep, schema in observation.endpoint_schemas.items():
        req = set(schema.get("required", []))
        score = len(payload_keys & req)
        if score > best_score:
            best_score = score
            best_ep = ep
    return best_ep


def random_policy(observation: Any) -> Action:
    action_type = random.choice(["validate", "route", "reject", "fix"])
    endpoint = random.choice(observation.available_endpoints) if observation.available_endpoints else None
    return Action(
        action_type=action_type,
        target_endpoint=endpoint if action_type in ["route", "fix"] else None,
        validation_issues=["random_check"] if action_type in ["validate", "reject"] else None,
        fix_suggestion={"note": "random"} if action_type == "fix" else None,
        confidence=0.5,
        reasoning="random baseline",
    )


def heuristic_policy(observation: Any) -> Action:
    payload = observation.request_payload
    endpoint = choose_endpoint(observation)
    schema = observation.endpoint_schemas.get(endpoint, {})
    required = schema.get("required", [])
    missing = [k for k in required if k not in payload]

    if missing:
        return Action(
            action_type="reject",
            target_endpoint=endpoint,
            validation_issues=[f"missing_required_field: {k}" for k in missing],
            confidence=0.85,
            reasoning="Missing required fields found",
        )

    return Action(
        action_type="route",
        target_endpoint=endpoint,
        confidence=0.85,
        reasoning="Required fields present for selected endpoint",
    )


def evaluate(task_id: str, policy_fn, episodes: int = 60) -> Dict[str, Any]:
    env = APIValidatorEnv(task_id=task_id)
    rewards: List[float] = []
    steps: List[int] = []
    done_count = 0

    for _ in range(episodes):
        obs = env.reset()
        done = False
        total = 0.0
        step_count = 0
        while not done and step_count < 6:
            action = policy_fn(obs)
            obs, reward, done, _ = env.step(action)
            total += reward.score
            step_count += 1
        rewards.append(total)
        steps.append(step_count)
        done_count += 1 if done else 0

    return {
        "episodes": episodes,
        "avg_reward": round(sum(rewards) / len(rewards), 4),
        "min_reward": round(min(rewards), 4),
        "max_reward": round(max(rewards), 4),
        "avg_steps": round(sum(steps) / len(steps), 2),
        "done_rate": round(done_count / episodes, 3),
    }


def run_train_test_baselines() -> Dict[str, Any]:
    report = {"random_policy": {}, "heuristic_policy": {}}
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        report["random_policy"][task] = evaluate(task, random_policy)
        report["heuristic_policy"][task] = evaluate(task, heuristic_policy)

    for label in ["random_policy", "heuristic_policy"]:
        vals = [report[label][t]["avg_reward"] for t in tasks]
        report[label]["overall_avg_reward"] = round(sum(vals) / len(vals), 4)

    return report


HF_API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
HF_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")


def check_hf_connectivity() -> Dict[str, Any]:
    try:
        token = os.getenv("HF_TOKEN")
        if not token:
            return {"status": "BLOCKED", "details": "HF_TOKEN environment variable is not set"}

        client = OpenAI(api_key=token, base_url=HF_API_BASE_URL)
    except Exception as exc:
        return {"status": "BLOCKED", "details": f"Hugging Face client setup failed: {exc}"}

    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        return {
            "status": "PASS",
            "model_count": len(model_ids),
            "first_model": model_ids[0] if model_ids else "none",
            "selected_model": HF_MODEL_NAME,
            "available_models": model_ids,
        }
    except Exception as exc:
        return {"status": "BLOCKED", "details": f"Hugging Face endpoint unreachable: {exc}"}


def run_inference_with_hf_model(model_name: str) -> Dict[str, Any]:
    env = os.environ.copy()
    env["MODEL_NAME"] = model_name
    env["API_BASE_URL"] = HF_API_BASE_URL

    code, out, err = run_cmd([sys.executable, str(INFERENCE_PATH)], cwd=ROOT, env=env)
    text = (out + "\n" + err).strip()

    task_matches = re.findall(r"Task\s+(easy|medium|hard)\s+-\s+Average Reward:\s+([0-9.]+)", text)
    overall_match = re.search(r"Overall Average:\s+([0-9.]+)", text)

    metrics = {task: float(score) for task, score in task_matches}
    overall = float(overall_match.group(1)) if overall_match else None

    return {
        "status": "PASS" if code == 0 and len(metrics) == 3 and overall is not None else "FAIL",
        "exit_code": code,
        "metrics": metrics,
        "overall": overall,
        "stdout_tail": "\n".join(text.splitlines()[-30:]),
    }


def check_docker() -> Dict[str, Any]:
    try:
        code, out, err = run_cmd(["docker", "--version"], cwd=ROOT)
        if code == 0:
            return {"status": "PASS", "details": out.strip()}
        return {"status": "BLOCKED", "details": (out + "\n" + err).strip()}
    except Exception as exc:
        return {"status": "BLOCKED", "details": f"docker not available: {exc}"}


def append_report_section(results: Dict[str, Any]) -> None:
    section_lines = []
    section_lines.append("\n---\n")
    section_lines.append("## Section 3 - Automated One-Command Pipeline Run\n")
    section_lines.append(f"Run timestamp (UTC): {results['timestamp']}\n")

    section_lines.append("### Stage Results\n")
    section_lines.append(f"- Static validation: {results['static']['status']}\n")
    section_lines.append(f"- API integration workflow: {results['api']['status']}\n")
    section_lines.append(f"- Train-test baselines: {results['baselines']['status']}\n")
    section_lines.append(f"- Hugging Face runtime: {results['huggingface']['status']}\n")
    section_lines.append(f"- Inference stage (live HF model): {results['inference']['status']}\n")
    section_lines.append(f"- Docker stage: {results['docker']['status']}\n")

    section_lines.append("\n### Baseline Metrics\n")
    section_lines.append(
        f"- Random policy overall avg reward: {results['baselines']['data']['random_policy']['overall_avg_reward']}\n"
    )
    section_lines.append(
        f"- Heuristic policy overall avg reward: {results['baselines']['data']['heuristic_policy']['overall_avg_reward']}\n"
    )

    section_lines.append("\n### Live Hugging Face Inference Metrics\n")
    inf = results["inference"]
    if inf["metrics"]:
        for task in ["easy", "medium", "hard"]:
            score = inf["metrics"].get(task, "n/a")
            section_lines.append(f"- {task}: {score}\n")
    section_lines.append(f"- overall: {inf['overall']}\n")

    section_lines.append("\n### API Workflow Snapshot\n")
    for row in results["api"]["data"]["results"]:
        section_lines.append(
            f"- {row['task']}: reset_ok={row['reset_ok']}, step_ok={row['step_ok']}, "
            f"state_ok={row['state_ok']}, reward={row['reward']}, done={row['done']}\n"
        )

    section_lines.append("\n### Notes\n")
    if results["server_boot_note"]:
        section_lines.append(f"- Server bootstrap: {results['server_boot_note']}\n")
    if results["huggingface"].get("details"):
        section_lines.append(f"- Hugging Face details: {results['huggingface']['details']}\n")
    section_lines.append("- This section is auto-appended by scripts/run_full_pipeline.py.\n")

    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write("".join(section_lines))


def main() -> int:
    random.seed(42)

    results: Dict[str, Any] = {
        "timestamp": now_utc(),
        "static": {"status": "FAIL", "details": ""},
        "api": {"status": "FAIL", "data": {}},
        "baselines": {"status": "FAIL", "data": {}},
        "huggingface": {"status": "BLOCKED"},
        "inference": {"status": "BLOCKED", "metrics": {}, "overall": None},
        "docker": {"status": "BLOCKED"},
        "server_boot_note": "",
    }

    server_proc = None
    try:
        static_ok, static_details = run_static_checks()
        results["static"] = {
            "status": "PASS" if static_ok else "FAIL",
            "details": static_details,
        }

        server_proc, boot_note = start_server_if_needed()
        results["server_boot_note"] = boot_note

        api_data = run_api_workflow()
        results["api"] = {"status": "PASS", "data": api_data}

        baseline_data = run_train_test_baselines()
        results["baselines"] = {"status": "PASS", "data": baseline_data}

        hf_data = check_hf_connectivity()
        results["huggingface"] = hf_data

        if hf_data.get("status") == "PASS":
            model_name = hf_data.get("selected_model", HF_MODEL_NAME)
            inf_data = run_inference_with_hf_model(model_name)
            results["inference"] = inf_data

        results["docker"] = check_docker()

    except Exception as exc:
        results.setdefault("pipeline_error", str(exc))
    finally:
        stop_server(server_proc)

    append_report_section(results)

    print("Pipeline completed. Report updated:", REPORT_PATH)
    print(json.dumps(results, indent=2))

    final_fail = any(
        [
            results["static"]["status"] != "PASS",
            results["api"]["status"] != "PASS",
            results["baselines"]["status"] != "PASS",
            results["huggingface"]["status"] not in {"PASS", "BLOCKED"},
            results["inference"]["status"] not in {"PASS", "BLOCKED"},
        ]
    )
    return 1 if final_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
