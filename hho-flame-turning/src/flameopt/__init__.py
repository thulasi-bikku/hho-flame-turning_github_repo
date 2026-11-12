"""Metaheuristic optimization of surface roughness and MRR in flame-assisted turning of Al/SiC composites."""
from .models import (LB, UB, COEFFS, ra_carburizing, ra_oxidizing, ra_dry, mrr, fitness,
                     RA_MODELS, CONDITIONS)
from . import algorithms

__all__ = ["LB", "UB", "COEFFS", "ra_carburizing", "ra_oxidizing", "ra_dry", "mrr", "fitness",
           "RA_MODELS", "CONDITIONS", "algorithms"]
__version__ = "1.0.0"
