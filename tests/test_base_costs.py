import torch

from nlmfsp.base_costs import SmoothObstacleCost
from nlmfsp.problem_setup import create_multi_obstacle_cost


def test_smooth_obstacle_cost_gradients():
    center = torch.tensor([0.0, 0.0])
    cost = SmoothObstacleCost(center, radius=1.0, penalty=10.0, beta=5.0)

    # Point deep inside should be near full penalty
    inside = torch.tensor([0.1, 0.0])
    val_inside = cost(inside)
    inside_norm = inside.norm()
    expected_inside = 10.0 * torch.sigmoid(5.0 * (1.0 - inside_norm))
    assert torch.isclose(val_inside, expected_inside, atol=1e-5)

    # Point far outside should be nearly zero
    outside = torch.tensor([3.0, 0.0])
    val_outside = cost(outside)
    assert val_outside < 1e-3

    # Gradient should be non-zero near boundary
    near_boundary = torch.tensor([0.95, 0.0], requires_grad=True)
    val = cost(near_boundary)
    val.backward()
    assert near_boundary.grad.norm() > 0


def test_multi_obstacle_cost_soft_and_hard():
    obstacles = [
        {
            "center": [0.0, 0.0],
            "radius": 1.0,
            "penalty": 5.0,
            "mode": "hard",
        },
        {
            "center": [2.0, 0.0],
            "radius": 1.0,
            "penalty": 10.0,
            "mode": "soft",
            "beta": 4.0,
        },
    ]
    device = torch.device("cpu")
    cost_fn = create_multi_obstacle_cost(obstacles, device)

    xs = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    ts = torch.zeros(xs.shape[0])
    cost_values = cost_fn(xs, ts)

    soft_center = torch.tensor([2.0, 0.0])
    distances = torch.norm(xs - soft_center, dim=-1)
    expected_soft = 10.0 * torch.sigmoid(4.0 * (1.0 - distances))
    expected = torch.tensor([5.0, 0.0, 0.0], device=xs.device, dtype=cost_values.dtype) + expected_soft
    assert torch.allclose(cost_values, expected, atol=1e-5)
