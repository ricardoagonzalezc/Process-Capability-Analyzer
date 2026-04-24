import numpy as np

def calculate_metrics(data, usl, lsl):
    mean = np.mean(data)
    std_within  = np.std(data, ddof=1)
    std_overall = np.std(data, ddof=0)

    cp  = (usl - lsl) / (6 * std_within)
    cpk = min((usl - mean), (mean - lsl)) / (3 * std_within)
    pp  = (usl - lsl) / (6 * std_overall)
    ppk = min((usl - mean), (mean - lsl)) / (3 * std_overall)

    return {
        "Cp":  round(cp,  2),
        "Cpk": round(cpk, 2),
        "Pp":  round(pp,  2),
        "Ppk": round(ppk, 2),
        "Mean": round(mean, 3),
        "Std Dev": round(std_within, 2)
    }

def get_verdict(cpk):
    if cpk >= 1.33:
        return "PASS", "✅"
    elif cpk >= 1.0:
        return "WARNING", "⚠️"
    else:
        return "FAIL", "❌"