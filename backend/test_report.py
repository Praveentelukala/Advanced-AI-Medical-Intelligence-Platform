from app.services.report_generator import generate_medical_report

report = generate_medical_report(
    prediction="notumor",
    confidence=99.85
)

print(report)