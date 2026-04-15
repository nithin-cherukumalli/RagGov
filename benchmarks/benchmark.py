"""Run RagGov fixture benchmarks."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raggov.engine import DiagnosisEngine  # noqa: E402
from raggov.io.serialize import load_run  # noqa: E402
from raggov.models.diagnosis import AnalyzerResult, Diagnosis  # noqa: E402
from raggov.models.run import RAGRun  # noqa: E402


console = Console()


@dataclass(frozen=True)
class FixtureBenchmark:
    """Benchmark result for one fixture."""

    fixture_name: str
    diagnosis: Diagnosis
    wall_time_ms: float


class TimedDiagnosisEngine(DiagnosisEngine):
    """Diagnosis engine variant that records per-analyzer latency."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self.analyzer_latencies_ms: dict[str, list[float]] = {}

    def _run_analyzer(self, analyzer: Any, run: RAGRun) -> AnalyzerResult:
        start = time.perf_counter()
        try:
            return super()._run_analyzer(analyzer, run)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.analyzer_latencies_ms.setdefault(analyzer.name(), []).append(elapsed_ms)


def fixture_paths(fixtures_dir: Path | None = None) -> list[Path]:
    """Return benchmark fixture paths."""
    base_dir = fixtures_dir or ROOT / "fixtures"
    return sorted(base_dir.glob("*.json"))


def run_benchmarks(fixtures_dir: Path | None = None) -> tuple[list[FixtureBenchmark], dict[str, float]]:
    """Run all fixtures through the diagnosis engine."""
    engine = TimedDiagnosisEngine()
    results: list[FixtureBenchmark] = []

    for path in fixture_paths(fixtures_dir):
        run = load_run(path)
        start = time.perf_counter()
        diagnosis = engine.diagnose(run)
        wall_time_ms = (time.perf_counter() - start) * 1000
        results.append(
            FixtureBenchmark(
                fixture_name=path.name,
                diagnosis=diagnosis,
                wall_time_ms=wall_time_ms,
            )
        )

    averages = {
        analyzer_name: sum(latencies) / len(latencies)
        for analyzer_name, latencies in sorted(engine.analyzer_latencies_ms.items())
        if latencies
    }
    return results, averages


def render_results(results: list[FixtureBenchmark], analyzer_averages: dict[str, float]) -> None:
    """Print benchmark results with Rich tables."""
    summary = Table(title="RagGov Fixture Benchmark")
    summary.add_column("Fixture")
    summary.add_column("Run ID")
    summary.add_column("Primary Failure")
    summary.add_column("Security Risk")
    summary.add_column("Should Answer")
    summary.add_column("Confidence", justify="right")
    summary.add_column("Wall Time (ms)", justify="right")
    summary.add_column("Checks", justify="right")

    total_ms = 0.0
    for result in results:
        diagnosis = result.diagnosis
        total_ms += result.wall_time_ms
        summary.add_row(
            result.fixture_name,
            diagnosis.run_id,
            diagnosis.primary_failure.value,
            diagnosis.security_risk.value,
            "Yes" if diagnosis.should_have_answered else "No",
            f"{diagnosis.confidence:.2f}" if diagnosis.confidence is not None else "",
            f"{result.wall_time_ms:.2f}",
            str(len(diagnosis.checks_run)),
        )

    console.print(summary)
    console.print(f"[bold]Total wall time:[/bold] {total_ms:.2f} ms")

    latency = Table(title="Per-Analyzer Average Latency")
    latency.add_column("Analyzer")
    latency.add_column("Avg Latency (ms)", justify="right")
    for analyzer_name, avg_ms in analyzer_averages.items():
        latency.add_row(analyzer_name, f"{avg_ms:.2f}")
    console.print(latency)


def main() -> None:
    """Run benchmarks and print the report."""
    results, analyzer_averages = run_benchmarks()
    render_results(results, analyzer_averages)


if __name__ == "__main__":
    main()
