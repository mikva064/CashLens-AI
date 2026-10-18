"""Helpers for the server owner; no HTTP server or API keys are implemented here."""

import base64
from pathlib import Path

from PIL import Image

from .contracts import AnalysisResult
from .errors import CashLensError


def to_api_payload(result: AnalysisResult, *, job_id: str, evidence_root: str | Path) -> dict:
    """Convert validated internal results to the jury's exact completed-job schema.

    Evidence paths must be inside this job's directory. Legal citations are
    included in description because the published API has no separate source
    field. The service should show the structured source in its UI as well.
    """
    if result.verdict == "requires_review":
        raise CashLensError("Review observations cannot be projected as a qualified binary verdict")
    root = Path(evidence_root).resolve(strict=True)
    violations = []
    for event in result.violations:
        frames = []
        for evidence in event.evidence_frames:
            path = evidence.image_path.resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise CashLensError("Evidence must be a file inside this job's output directory")
            with Image.open(path) as image:
                if image.format != "JPEG":
                    raise CashLensError("Evidence must be encoded as JPEG")
                image.verify()
            frames.append({"timestamp_sec": evidence.timestamp_sec,
                           "image_base64": base64.b64encode(path.read_bytes()).decode("ascii")})
        citation = f"Основание: {event.source.document}, {event.source.section}. {event.source.url}"
        prefix = {"generated": "Описание локальной модели по кадрам: ",
                  "uncertain": "Локальная модель не смогла уверенно описать эпизод. ",
                  "unavailable": "Описание эпизода недоступно. ",
                  "legacy_template": "Критерий выбранной категории (шаблон): "}.get(event.commentary_status, "")
        violations.append({"type": event.type, "timecode_sec": event.timecode_sec,
                           "description": prefix + event.description + "\n" + citation,
                           "evidence_frames": frames})
    return {"job_id": job_id, "status": "done", "verdict": result.verdict,
            "violations": violations}
