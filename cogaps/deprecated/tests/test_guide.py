# test_guide.py
import torch
import pyro
import pytest
from cogaps.guide import guide

def test_guide_params():
    num_genes = 100
    num_samples = 50
    num_patterns = 10
    D = torch.rand(num_genes, num_samples)
    
    trace = pyro.poutine.trace(guide).get_trace(D, num_patterns)
    A = trace.nodes['A']['value']
    P = trace.nodes['P']['value']
    
    assert torch.all(A >= 0)
    assert torch.all(P >= 0)