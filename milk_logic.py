from datetime import datetime

GRADE_TPC_LOG_PROXY = {"Low": 6.2, "Medium": 4.9, "High": 3.5}
FSSAI_LIMITS = {"raw_milk": 500_000, "pasteurized": 30_000, "UHT": 10}
SPOILAGE_HOURS = {
    "High": {"4C": 72, "10C": 36, "25C": 18, "37C": 8},
    "Medium": {"4C": 24, "10C": 12, "25C": 6, "37C": 2},
    "Low": {"4C": 0, "10C": 0, "25C": 0, "37C": 0},
}


def temperature_key(temp):
    if temp <= 6:
        return "4C"
    if temp <= 15:
        return "10C"
    if temp <= 30:
        return "25C"
    return "37C"


def build_report(grade, confidence, probabilities, storage_temp_c, elapsed_hrs,
                 milk_type, consumer_group, raw):
    # These are prototype rules copied from the supplied project, not validated
    # microbiology measurements or a certified food-safety assessment.
    tpc_log = GRADE_TPC_LOG_PROXY[grade]
    tpc_cfu = 10 ** tpc_log
    limit = FSSAI_LIMITS[milk_type]
    fssai_pass = tpc_cfu <= limit

    key = temperature_key(storage_temp_c)
    shelf_life = SPOILAGE_HOURS[grade][key]
    remaining = max(0.0, float(shelf_life) - elapsed_hrs)
    if grade == "Low" or remaining <= 0:
        status = "SPOILED / DISCARD"
        storage_recommendation = "Do not consume based on this prototype result; follow validated local food-safety guidance."
    elif remaining <= 6:
        status = "BORDERLINE"
        storage_recommendation = "Use caution; this rule-based estimate is not a safety guarantee."
    elif remaining <= 24:
        status = "CAUTION"
        storage_recommendation = "Keep refrigerated and follow the product label and validated shelf-life guidance."
    else:
        status = "SAFE-LOOKING (ESTIMATE)"
        storage_recommendation = "Keep refrigerated and follow the product label. This estimate does not prove safety."

    # High-risk groups should not receive reassuring recommendations from an
    # unvalidated estimate. This score is retained only as a demonstration.
    sensitivity = {"adult": 1.0, "child": 1.5, "infant": 2.5,
                   "elderly": 1.8, "immunocompromised": 2.2}[consumer_group]
    adjusted = tpc_log * sensitivity
    if adjusted < 4:
        risk_level = "Low"
    elif adjusted < 5:
        risk_level = "Moderate"
    elif adjusted < 5.7:
        risk_level = "High"
    else:
        risk_level = "Critical"
    risk_score = min(100, round(adjusted / 9 * 100))

    # Never say "safe" if this prototype's own rules say discard or the selected
    # limit is exceeded. This also fixes the contradictory verdict in Sample B.
    if grade == "Low" or remaining <= 0 or not fssai_pass:
        verdict = "FAIL / DO NOT CONSUME — prototype warning"
    elif consumer_group in ("infant", "elderly", "immunocompromised") or risk_level in ("High", "Critical"):
        verdict = "CAUTION — do not rely on this prototype to establish safety"
    elif grade == "Medium":
        verdict = "CONDITIONAL — quality needs further checking"
    else:
        verdict = "QUALITY CLASSIFIED HIGH — not a food-safety certification"

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "grade": grade,
        "confidence_pct": round(confidence * 100, 1),
        "probabilities": {k: round(v * 100, 1) for k, v in probabilities.items()},
        "microbial_quality": {"High": "Acceptable (model class)", "Medium": "Borderline (model class)",
                              "Low": "Unacceptable (model class)"}[grade],
        "tpc_log": round(tpc_log, 2),
        "tpc_cfu": f"{tpc_cfu:,.0f}",
        "fssai_limit": f"{limit:,}",
        "fssai_pass": fssai_pass,
        "storage_temp": storage_temp_c,
        "elapsed_hrs": elapsed_hrs,
        "hours_remaining": round(remaining, 1),
        "spoilage_status": status,
        "storage_recommendation": storage_recommendation,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "consumer_group": consumer_group.replace("_", " ").title(),
        "verdict": verdict,
        "input_features": raw,
    }
