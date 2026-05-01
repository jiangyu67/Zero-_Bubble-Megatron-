#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Optional

SUMMARY_PATTERNS = {
    "mode": re.compile(r"^Mode:\s*(.*)$"),
    "avg_step_time": re.compile(r"^\s*avg step time \(ms\):\s*([0-9.]+)$"),
    "peak_memory": re.compile(r"^\s*peak memory \(bytes\):\s*([0-9]+)$"),
    "hidden_wait": re.compile(r"^\s*avg hidden wait \(ms\):\s*([0-9.]+)$"),
    "hidden_ratio": re.compile(r"^\s*hidden latency %:\s*([0-9.]+)%$"),
    "daq_metrics": re.compile(
        r"^DAQ_METRICS:.*avg_sign_rate=([0-9.]+).*avg_cos_sim=([0-9.]+).*dlrc_trigger_percent=([0-9.]+)"
    ),
    "config": re.compile(
        r"^CONFIG:.*batch_size=([0-9]+).*seq_length=([0-9]+).*hidden_size=([0-9]+).*steps=([0-9]+).*warmup=([0-9]+)"
    ),
}
METRIC_LINE = re.compile(
    r"^\[QuantMetrics\].*SignRate:\s*([0-9.]+),\s*CosSim:\s*([0-9.]+)"
)


def parse_log_file(path: str) -> Dict[str, Optional[float]]:
    data = {
        "path": path,
        "mode": None,
        "avg_step_time": None,
        "peak_memory": None,
        "hidden_wait": None,
        "hidden_ratio": None,
        "batch_size": 8,
        "seq_length": 64,
        "hidden_size": 1024,
        "steps": None,
        "warmup": None,
        "avg_sign_rate": None,
        "avg_cos_sim": None,
        "dlrc_trigger_percent": None,
        "quant_metric_count": 0,
        "sum_sign_rate": 0.0,
        "sum_cos_sim": 0.0,
        "dlrc_trigger_count": 0,
    }

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            for key, pattern in SUMMARY_PATTERNS.items():
                match = pattern.match(line)
                if not match:
                    continue

                if key == "mode":
                    data["mode"] = match.group(1)
                elif key == "avg_step_time":
                    data["avg_step_time"] = float(match.group(1))
                elif key == "peak_memory":
                    data["peak_memory"] = int(match.group(1))
                elif key == "hidden_wait":
                    data["hidden_wait"] = float(match.group(1))
                elif key == "hidden_ratio":
                    data["hidden_ratio"] = float(match.group(1))
                elif key == "daq_metrics":
                    data["avg_sign_rate"] = float(match.group(1))
                    data["avg_cos_sim"] = float(match.group(2))
                    data["dlrc_trigger_percent"] = float(match.group(3))
                elif key == "config":
                    data["batch_size"] = int(match.group(1))
                    data["seq_length"] = int(match.group(2))
                    data["hidden_size"] = int(match.group(3))
                    data["steps"] = int(match.group(4))
                    data["warmup"] = int(match.group(5))

            quant_match = METRIC_LINE.match(line)
            if quant_match:
                sign_rate = float(quant_match.group(1))
                cos_sim = float(quant_match.group(2))
                data["quant_metric_count"] += 1
                data["sum_sign_rate"] += sign_rate
                data["sum_cos_sim"] += cos_sim
                if sign_rate < 0.95 or cos_sim < 0.98:
                    data["dlrc_trigger_count"] += 1

    if data["quant_metric_count"] > 0 and data["avg_sign_rate"] is None:
        data["avg_sign_rate"] = data["sum_sign_rate"] / data["quant_metric_count"]
        data["avg_cos_sim"] = data["sum_cos_sim"] / data["quant_metric_count"]
        data["dlrc_trigger_percent"] = (
            data["dlrc_trigger_count"] / data["quant_metric_count"] * 100.0
        )

    if data["peak_memory"] is not None:
        data["peak_memory_gb"] = data["peak_memory"] / 1024 ** 3
    else:
        data["peak_memory_gb"] = None

    data["throughput_tflops"] = None
    if data["avg_step_time"] is not None:
        flops_per_step = 12.0 * data["batch_size"] * data["seq_length"] * data["hidden_size"] ** 2
        time_s = data["avg_step_time"] / 1000.0
        data["throughput_tflops"] = flops_per_step / time_s / 1e12

    return data


def format_table(rows: List[Dict[str, Optional[float]]]) -> str:
    headers = [
        "Experiment",
        "Avg Step Time (ms)",
        "Throughput (TFLOPS)",
        "Memory Peak (GB)",
        "Avg CosSim",
        "Avg SignRate",
        "DLRC Trigger %",
        "Hidden Latency %",
    ]
    col_widths = [max(len(h), 16) for h in headers]
    output = []
    output.append(" | ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    output.append("-|-".join("-" * w for w in col_widths))

    for row in rows:
        values = [
            str(os.path.basename(row["path"])),
            f"{row['avg_step_time']:.3f}" if row["avg_step_time"] is not None else "N/A",
            f"{row['throughput_tflops']:.3f}" if row["throughput_tflops"] is not None else "N/A",
            f"{row['peak_memory_gb']:.3f}" if row["peak_memory_gb"] is not None else "N/A",
            f"{row['avg_cos_sim']:.4f}" if row["avg_cos_sim"] is not None else "N/A",
            f"{row['avg_sign_rate']:.4f}" if row["avg_sign_rate"] is not None else "N/A",
            f"{row['dlrc_trigger_percent']:.2f}" if row["dlrc_trigger_percent"] is not None else "N/A",
            f"{row['hidden_ratio']:.2f}" if row["hidden_ratio"] is not None else "N/A",
        ]
        output.append(" | ".join(v.ljust(w) for v, w in zip(values, col_widths)))
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse benchmark experiment logs and summarize results.")
    parser.add_argument(
        "logs",
        nargs="+",
        help="One or more log files produced by run_all_experiments.sh",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional output CSV file to save the parsed results.",
    )
    args = parser.parse_args()

    rows = []
    for log_path in args.logs:
        if not os.path.isfile(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")
        rows.append(parse_log_file(log_path))

    print(format_table(rows))

    if args.csv:
        import csv

        fieldnames = [
            "path",
            "mode",
            "avg_step_time",
            "throughput_tflops",
            "peak_memory_gb",
            "avg_cos_sim",
            "avg_sign_rate",
            "dlrc_trigger_percent",
            "hidden_ratio",
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        print(f"Wrote CSV summary to: {args.csv}")


if __name__ == "__main__":
    main()
