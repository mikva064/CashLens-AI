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
    pass


class RuleValidationError(CashLensError):
    pass


class AnalysisTimeoutError(CashLensError):
    pass
