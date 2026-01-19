__all__ = [
    "UnsupportedSyntaxError",
    "ModelNotReadyError",
]


class UnsupportedSyntaxError(SyntaxError):
    pass


class ModelNotReadyError(RuntimeError):
    pass
