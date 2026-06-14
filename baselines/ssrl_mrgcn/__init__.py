"""SSRL-style frozen face embedding + residual MR-GCN baseline."""

from .model import ResidualMRConv, SSRLMRGCN

__all__ = ["ResidualMRConv", "SSRLMRGCN"]
