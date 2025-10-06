
import torch
from pc15vq.qv import train_qv_kmeans, quantize_batch

def test_qv_train_k_leq_n_and_shapes():
    data = torch.randn(10, 4)
    cb = train_qv_kmeans(data, K=5, iters=5, seed=0)
    assert cb.shape == (5,4)
    # K==N
    cb2 = train_qv_kmeans(data, K=10, iters=1, seed=0)
    assert cb2.shape == (10,4)
    # Quantize batch
    idx, res = quantize_batch(data, cb)
    assert idx.shape[0] == data.shape[0]
    assert res.shape == data.shape
