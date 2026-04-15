# RagGov Benchmarks

Run the fixture benchmark from the repository root:

```bash
python benchmarks/benchmark.py
```

The benchmark loads every `*.json` file in `fixtures/`, validates each file as a `RAGRun`, runs the default `DiagnosisEngine`, and prints Rich tables.

## Columns

- `Fixture`: fixture JSON file name.
- `Run ID`: `RAGRun.run_id` from the fixture.
- `Primary Failure`: final diagnosis primary failure selected by taxonomy priority.
- `Security Risk`: highest security risk reported by analyzer results.
- `Should Answer`: whether the engine believes the system should have answered.
- `Confidence`: final confidence score from `ConfidenceAnalyzer`, when available.
- `Wall Time (ms)`: end-to-end diagnosis time for that fixture.
- `Checks`: number of analyzer checks run for the fixture.
- `Avg Latency (ms)`: average per-analyzer runtime across all fixtures.

## Current Baseline

- Total wall time: TODO
- Slowest analyzer: TODO
- Average fixture wall time: TODO
- Environment: TODO

## Adding Fixtures

Add a valid `RAGRun` JSON file to `fixtures/` with a `.json` extension. The benchmark automatically includes it on the next run. Keep fixture text realistic, include representative retrieval scores, and add `metadata.scenario` plus `metadata.expected_primary_failure` when useful for later regression tests.
