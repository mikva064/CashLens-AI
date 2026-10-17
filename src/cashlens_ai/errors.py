"""Failures must be surfaced to the job runner, never converted to clean verdicts."""


class CashLensError(Exception):
    """Base error that a server can convert to a failed job."""


class DatasetValidationError(CashLensError):
    pass


class VideoReadError(CashLensError):
    pass


class ModelUnavailableError(CashLensError):
    pass


class InvalidModelOutputError(CashLensError):
    """Invalid output with an allowlisted diagnostic, separate from scene text."""

    REASON_CODES = frozenset({
        "invalid_model_output", "invalid_compact_json", "duplicate_json_key", "nonfinite_json",
        "invalid_compact_shape", "too_many_events", "invalid_event_fields", "unknown_category",
        "invalid_action_index", "duplicate_event", "invalid_proof_index", "invalid_proof_order",
        "invalid_cash_context", "candidate_focus", "response_truncated", "invalid_event_json",
        "event_outside_window", "invalid_source_proof", "missing_proof_context",
        "refinement_frame_cap", "proof_frame_recovery",
    })

    def __init__(self, message="", *, reason_code="invalid_model_output"):
        super().__init__(message)
        self.reason_code = self.safe_reason_code(reason_code)

    @classmethod
    def safe_reason_code(cls, value):
        return value if type(value) is str and value in cls.REASON_CODES else "invalid_model_output"


class RuleValidationError(CashLensError):
    pass


class AnalysisTimeoutError(CashLensError):
    pass
