"""
Structure Align - A tool for aligning protein structures with differing lengths.
"""

import warnings

# Suppress known warnings from dependencies
warnings.filterwarnings("ignore", message="IProgress not found.*", category=UserWarning)
warnings.filterwarnings(
    "ignore", message=".*Bio.Application.*", category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore", message=".*Bio.Application.*", module="Bio.Application"
)

# Also suppress tqdm warnings specifically
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm.auto")

from .aligner import StructuralAligner
from .models import AlignmentResult, SequenceAlignment

__version__ = "0.1.0"
__all__ = ["StructuralAligner", "AlignmentResult", "SequenceAlignment"]
