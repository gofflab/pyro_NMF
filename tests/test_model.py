# test_model.py
import torch
import pyro
import pytest
from cogaps.model import cogaps_model

def test_shapes():
    num_genes = 100
    num_samples = 50
    num_patterns = 10
    D = torch.rand(num_genes, num_samples)
    
    print(D.shape)
    
    trace = pyro.poutine.trace(cogaps_model).get_trace(D, num_patterns)
    A = trace.nodes['A']['value']
    P = trace.nodes['P']['value']
    
    assert A.shape == (num_genes, num_patterns)
    assert P.shape == (num_patterns, num_samples)

def test_positive():
    num_genes = 100
    num_samples = 50
    num_patterns = 10
    D = torch.rand(num_genes, num_samples)
    
    trace = pyro.poutine.trace(cogaps_model).get_trace(D, num_patterns)
    A = trace.nodes['A']['value']
    P = trace.nodes['P']['value']
    
    assert torch.all(A >= 0)
    assert torch.all(P >= 0)