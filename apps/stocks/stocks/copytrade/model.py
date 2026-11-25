"""The copy books: they live in models/copy.py, with the other models. Here so copytrade's own modules can say
`from .model import …`."""
from ..models.copy import FULL, BAND, SLACK, Copy, CopyBook, CopyData, Period

__all__ = ["FULL", "BAND", "SLACK", "Copy", "CopyBook", "CopyData", "Period"]
