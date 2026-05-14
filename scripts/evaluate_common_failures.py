#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from stresslab.runners.rag_failure_runner import main

if __name__ == "__main__":
    # Default to saving report in reports/
    suite = "common"
    for i, arg in enumerate(sys.argv):
        if arg == "--suite" and i + 1 < len(sys.argv):
            suite = sys.argv[i + 1]

    if "--output" not in sys.argv:
        report_name = "common_failure_coverage_matrix.md" if suite == "common" else "subtle_failure_coverage_matrix.md"
        report_path = root / "reports" / report_name
        report_path.parent.mkdir(exist_ok=True)
        sys.argv.extend(["--output", str(report_path)])
    
    main()
