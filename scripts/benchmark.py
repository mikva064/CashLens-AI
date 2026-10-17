"""Прогон отложенных видео через API сервиса — так же, как это сделает жюри.

Измеряет качество вердикта (через cashlens_ai.evaluate) и время от загрузки до
ответа. Итог — local/bench/benchmark.md с таблицей для презентации.

    .venv/Scripts/python scripts/benchmark.py --videos "C:/dataset/videos" \
        --url http://127.0.0.1:8000 --api-key <ключ>
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

from cashlens_ai.evaluate import evaluate_predictions

LIMIT_SEC = 300


def run_one(client: httpx.Client, video: Path) -> tuple[dict, float]:
    t0 = time.monotonic()
    with open(video, "rb") as f:
        r = client.post("/api/v1/analyze", files={"file": (video.name, f)}, timeout=120)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    while True:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        if time.monotonic() - t0 > LIMIT_SEC + 60:
            return {"status": "failed", "error": "no result within limit"}, time.monotonic() - t0
        time.sleep(1)
    elapsed = time.monotonic() - t0
    if job["status"] == "failed":
        return {"status": "failed", "error": job.get("error") or "failed"}, elapsed
    analysis = client.get(f"/api/v1/jobs/{job_id}/analysis").json()
    return {"status": "done", "result": analysis}, elapsed


def pct(value):
    return "—" if value is None else f"{value * 100:.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=Path("local/audit/manifest.json"))
    ap.add_argument("--videos", type=Path, required=True, help="папка с исходными .mov")
    ap.add_argument("--split", choices=["holdout", "development"], default="holdout")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--output", type=Path, default=Path("local/bench"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    groups = [g for g in manifest["groups"] if g["split"] == args.split and g["label"] is not None]
    args.output.mkdir(parents=True, exist_ok=True)

    predictions, timings = [], []
    with httpx.Client(base_url=args.url, headers={"X-API-Key": args.api_key}) as client:
        analyzer = client.get("/health").json().get("analyzer")
        for i, group in enumerate(groups, 1):
            video = args.videos / group["filenames"][0]
            outcome, elapsed = run_one(client, video)
            timings.append(elapsed)
            predictions.append({"sha256": group["sha256"], **outcome})
            verdict = outcome.get("result", {}).get("verdict", outcome["status"])
            truth = "violation" if group["label"] == 1 else "no_violation"
            print(f"[{i}/{len(groups)}] {video.name}: {verdict} (разметка: {truth}), {elapsed:.0f} с")

    pred_path = args.output / "predictions.json"
    pred_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    report = evaluate_predictions(args.manifest, pred_path, split=args.split)
    (args.output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    m, c, n = report["metrics"], report["confusion"], report["counts"]
    offset = report["weak_point_offsets"]["mean_group_nearest_abs_offset_sec"]
    lines = [
        f"# Прогон CashLens: {args.split}, анализатор {analyzer}",
        "",
        f"Видео: {n['selected_groups']} уникальных ({n['selected_positive_groups']} с нарушением, "
        f"{n['selected_negative_groups']} без). Обработано: {n['processed']}, сбоев: {n['failed']}.",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| Верный вердикт (все видео) | {pct(m['accuracy_all'])} |",
        f"| Найдено нарушений из размеченных (recall) | {pct(m['recall_processed'])} |",
        f"| Точность срабатываний (precision) | {pct(m['precision_processed'])} |",
        f"| Пропуски нарушений / ложные срабатывания | {c['fn']} / {c['fp']} |",
        "| Среднее расстояние до размеченной секунды | "
        + ("—" if offset is None else f"{offset:.1f} с") + " |",
        f"| Время ответа: среднее / максимум | {statistics.mean(timings):.0f} с / {max(timings):.0f} с |",
        f"| Уложились в 5 минут | {sum(t <= LIMIT_SEC for t in timings)} из {len(timings)} |",
        "",
        "Разметка датасета — одна секунда на ролик, без типа и границ события, поэтому "
        "расстояние до размеченной секунды — ориентир, а не метрика локализации.",
    ]
    (args.output / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
