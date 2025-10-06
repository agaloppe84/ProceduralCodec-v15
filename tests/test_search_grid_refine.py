
import torch
from pc15codec.search import grid_linspace, local_refine, select_best_param_from_scores

def _quad_cost(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # params: (N,D), target: (D,)
    diff = params - target.unsqueeze(0)
    return torch.sum(diff*diff, dim=1)

def test_grid_linspace_and_refine_find_minimum():
    # Minimum vrai en (0.3, -0.2)
    t = torch.tensor([0.3, -0.2], dtype=torch.float32)
    # Coarse grid
    coarse = grid_linspace([(-1.0,1.0),(-1.0,1.0)], [9,9])
    c_scores = _quad_cost(coarse, t)
    c_best, _ = select_best_param_from_scores(coarse, c_scores)

    # Refinement autour de c_best
    ref = local_refine(c_best, deltas=torch.tensor([0.3,0.3]), steps=[7,7])
    r_scores = _quad_cost(ref, t)
    r_best, _ = select_best_param_from_scores(ref, r_scores)

    # Le refine doit être plus proche du minimum
    def dist(a,b): return torch.sqrt(torch.sum((a-b)**2))
    assert dist(r_best, t) <= dist(c_best, t) + 1e-6
