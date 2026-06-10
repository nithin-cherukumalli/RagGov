# Codebase Cleanup Inventory Start

- Generated: `2026-06-03T07:46:14.394975+00:00`

## Harness Status
```json
{
  "harness_post_edit_validation": {
    "returncode": 0,
    "stderr": "",
    "stdout": "harness post-edit validation: warn"
  },
  "harness_preflight": {
    "returncode": 1,
    "stderr": "",
    "stdout": "harness preflight: fail"
  },
  "workspace_audit": {
    "returncode": 1,
    "stderr": "",
    "stdout": "workspace audit: fail"
  }
}
```

## Deleted Files
- ensemble_70b_eval.txt
- ensemble_8b_eval.txt
- heuristic_eval.txt
- llm_70b_eval.txt
- llm_8b_eval.txt
- run_stress_cases.py

## Protected Dirty Or Deleted
- .gitignore
- src/raggov/decision_policy.py
- src/raggov/engine.py
- evals/govrag_calib/
- src/raggov/decision_policy_support.py
