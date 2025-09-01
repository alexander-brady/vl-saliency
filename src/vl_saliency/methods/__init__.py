"""
Methods for Saliency Computation from Attention and Gradients

Each method takes as input the attention maps and gradients, and returns a mask tensor.
"""

from . import cam, primitives
from .registry import resolve

__all__ = ["primitives", "cam", "resolve"]
