from raggov.evaluators.registry import create_standard_registry
from raggov.evaluators.doctor import build_provider_doctor_report, render_provider_doctor_text

registry = create_standard_registry()
report = build_provider_doctor_report(registry)
print(render_provider_doctor_text(report))
