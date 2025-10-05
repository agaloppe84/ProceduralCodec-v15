import torch
from pc15core.rng import derive_seed64, seed_tensor

def test_derive_seed64_stable():
    s1 = derive_seed64(1, 2, 3)
    s2 = derive_seed64(1, 2, 3)
    assert s1 == s2

def test_seed_tensor_shape():
    t = torch.tensor([1, 2, 3], dtype=torch.int64)
    s = seed_tensor(t)
    assert s.shape == t.shape
