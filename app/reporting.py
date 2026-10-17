"""Web result projection and immutable UTF-8 JSON reports, without server paths."""
import hashlib
import json
from pathlib import Path

from cashlens_ai.contracts import AnalysisResult


def web_result(job: dict, jobs_dir: Path) -> dict:
    stored = job["result"]
    result = AnalysisResult.model_validate(stored["analysis"])
    job_dir = (jobs_dir / job["id"]).resolve()
    events = []
    for index, event in enumerate(result.violations):
        frames = []
        for frame in event.evidence_frames:
            relative = frame.image_path.resolve().relative_to(job_dir)
            frames.append({"timestamp_sec": frame.timestamp_sec,
                           "url": f"/media/{job['id']}/{relative.as_posix()}"})
        item = {"type": event.type, "timecode_sec": event.timecode_sec,
                "description": event.description, "rule_id": event.rule_id,
                "source": event.source.model_dump(mode="json"), "evidence_frames": frames}
        filename = f"event_{index + 1:03d}.mp4"
        clip = stored.get("clips", {}).get(str(index))
        if clip and clip.get("filename") == filename and (job_dir / "clips" / filename).is_file():
            item["clip_url"] = f"/media/{job['id']}/clips/{filename}"
        events.append(item)
    observations = []
    for index, event in enumerate(result.observations):
        frames = []
        for frame in event.evidence_frames:
            relative = frame.image_path.resolve().relative_to(job_dir)
            frames.append({"timestamp_sec": frame.timestamp_sec,
                           "url": f"/media/{job['id']}/{relative.as_posix()}"})
        item = {"observation_id": event.observation_id, "type": event.type,
                "timecode_sec": event.timecode_sec, "description": event.description,
                "review_reason": event.review_reason, "evidence_frames": frames}
        filename = f"observation_{index + 1:03d}.mp4"
        clip = stored.get("observation_clips", {}).get(str(index))
        if clip and clip.get("filename") == filename and (job_dir / "clips" / filename).is_file():
            item["clip_url"] = f"/media/{job['id']}/clips/{filename}"
        observations.append(item)
    completed_at = stored.get("completed_at") or job.get("finished_at") or job["created_at"]
    return {"job_id": job["id"], "status": "done", "verdict": result.verdict,
            "violations": events, "observations": observations, "review_required": bool(observations),
            "filename": job["filename"], "created_at": job["created_at"],
            "video_url": f"/media/{job['id']}/preview.mp4",
            "original_url": f"/media/{job['id']}/video.mov",
            "analyzer": stored["analyzer"], "duration_sec": stored["duration_sec"],
            "processing_time_sec": round(completed_at - job["created_at"], 2),
            "timings": stored["timings"], "input_sha256": job.get("input_sha256")}


def write_result_report(job: dict, jobs_dir: Path) -> str:
    """Hash exactly the downloaded file bytes; feedback and the hash stay outside.

    Using a stable artifact avoids a self-referential checksum and prevents a
    later reviewer decision from changing the result being checked.
    """
    content = (json.dumps(web_result(job, jobs_dir), ensure_ascii=False, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    target = jobs_dir / job["id"] / "result.json"
    temporary = target.with_name("result.part.json")
    try:
        temporary.write_bytes(content)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()
