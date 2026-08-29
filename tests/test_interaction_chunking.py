import math

import torch

from nlmfsp.interaction import (
    InteractionLearner,
    RandomBatchCost,
    RandomBatchDrift,
    TransformedInteractionModule,
    create_interactions,
)
from nlmfsp.config import Config
from nlmfsp.costs import (
    CongestionCost,
    CostPotential,
    CoulombCost,
    EntropyCost,
    FlockingAlignmentCost,
    GaussianConvolutionCost,
    GaussianConvolutionMeanCost,
    LocalExclusionCost,
    MeanDeviationCost,
    PedestrianVisualCost,
    QuadraticDeviationCost,
    QuadraticPotentialCost,
    ZeroCost,
)
from nlmfsp.drifts import (
    AttractionRepulsionDrift,
    CoulombDrift,
    Drift,
    GaussianDrift,
    LaplacianDiffusionDrift,
    MeanReversionDrift,
    NonstationaryAnisotropicDrift,
    OpinionMeanFieldDrift,
    SmoothBoundedConfidenceDrift,
    ZeroDrift,
    build_opinion_f_mul,
)


def _assert_allclose(a: torch.Tensor, b: torch.Tensor) -> None:
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (a - b).abs().max().item()


def test_drift_chunking_matches_full_pair_evaluation():
    torch.manual_seed(0)
    x = torch.randn(9, 3)
    empirical = torch.randn(11, 3)

    drifts = [
        GaussianDrift(sigma=0.7, weight=0.9),
        AttractionRepulsionDrift(
            attract=1.2,
            sigma_attract=0.8,
            repel=0.4,
            sigma_repel=0.3,
            weight=0.5,
        ),
        CoulombDrift(alpha=1.3, eps=0.05, weight=0.2),
        NonstationaryAnisotropicDrift(sigma=0.7, weight=0.9, alpha=0.6),
        LaplacianDiffusionDrift(sigma=1.1, kernel_type="gaussian", weight=0.7),
        LaplacianDiffusionDrift(sigma=1.1, kernel_type="exponential", weight=0.7),
        LaplacianDiffusionDrift(sigma=1.1, kernel_type="inverse", weight=0.7),
        SmoothBoundedConfidenceDrift(sigma=0.8, weight=1.2),
    ]

    for drift in drifts:
        full = drift.evaluate(x, empirical)
        drift.set_chunk_sizes(query_chunk_size=4, ref_chunk_size=5)
        chunked = drift.evaluate(x, empirical)
        _assert_allclose(full, chunked)


def test_cost_chunking_matches_full_pair_evaluation():
    torch.manual_seed(1)
    x = torch.randn(8, 2)
    empirical = torch.randn(13, 2)

    costs = [
        GaussianConvolutionCost(sigma=0.6, weight=1.2),
        GaussianConvolutionMeanCost(sigma=0.6, weight=1.2),
        PedestrianVisualCost(range_scale=0.8, weight=1.2),
        QuadraticPotentialCost(weight=0.4),
        CoulombCost(alpha=1.1, eps=0.03),
        LocalExclusionCost(epsilon=0.5, strength=0.8),
        EntropyCost(weight=0.9),
        CongestionCost(weight=1.3),
    ]

    for cost in costs:
        full = cost.evaluate(x, empirical)
        cost.set_chunk_sizes(query_chunk_size=3, ref_chunk_size=4)
        chunked = cost.evaluate(x, empirical)
        _assert_allclose(full, chunked)


def test_create_interactions_propagates_chunk_sizes():
    config = Config(
        device="cpu",
        n_agents=8,
        drift_type="gaussian",
        cost_type="gaussian_conv",
        int_drift_sigma=0.8,
        int_cost_sigma=0.9,
        interaction_backend="nn",
        learn_interaction_drift=True,
        learn_interaction_cost=True,
        learning_rate_interaction=1e-3,
        interaction_query_chunk_size=7,
        interaction_ref_chunk_size=9,
        interaction_time_block_size=3,
    )

    _, _, f_interaction_fn, F_interaction_fn, analytical_drift, analytical_cost = (
        create_interactions(config)
    )
    assert f_interaction_fn is not None
    assert F_interaction_fn is not None

    assert getattr(analytical_drift.drift, "query_chunk_size") == 7
    assert getattr(analytical_drift.drift, "ref_chunk_size") == 9
    assert getattr(analytical_cost.cost, "query_chunk_size") == 7
    assert getattr(analytical_cost.cost, "ref_chunk_size") == 9


def test_create_interactions_supports_gaussian_convolution_mean_cost():
    config = Config(
        device="cpu",
        n_agents=8,
        drift_type="zero",
        cost_type="gaussian_conv_mean",
        int_cost_sigma=0.9,
        int_cost_weight=1.1,
        interaction_backend="exact",
    )

    _, _, _, _, _, analytical_cost = create_interactions(config)

    assert analytical_cost.cost.name == "gaussian_conv_mean"
    assert analytical_cost.cost.random_batch_aggregation == "mean"


def test_create_interactions_supports_pedestrian_visual_cost():
    config = Config(
        device="cpu",
        n_agents=8,
        drift_type="zero",
        cost_type="pedestrian_visual",
        int_cost_sigma=1.3,
        int_cost_weight=0.7,
        interaction_backend="exact",
    )

    _, _, _, _, _, analytical_cost = create_interactions(config)

    assert analytical_cost.cost.name == "pedestrian_visual"
    assert analytical_cost.cost.range_scale == 1.3
    assert analytical_cost.cost.weight == 0.7
    assert analytical_cost.cost.random_batch_aggregation == "mean"


def test_transformed_interaction_module_scales_input_and_unstandardizes_output():
    class EchoNet(torch.nn.Module):
        def forward(self, x, t):
            return x[:, :1] + t.unsqueeze(1)

    module = TransformedInteractionModule(
        EchoNet(),
        output_dim=1,
        input_scale=2.0,
        standardize_targets=True,
    )
    module.target_mean.copy_(torch.tensor([3.0]))
    module.target_std.copy_(torch.tensor([4.0]))

    x = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
    t = torch.tensor([0.5, 1.0])

    normalized = module.forward_normalized(x, t)
    physical = module(x, t)

    _assert_allclose(normalized, torch.tensor([[1.5], [3.0]]))
    _assert_allclose(physical, torch.tensor([[9.0], [15.0]]))


def test_transformed_interaction_module_handles_single_constant_target():
    module = TransformedInteractionModule(
        torch.nn.Identity(),
        output_dim=1,
        input_scale=1.0,
        standardize_targets=True,
    )

    module.set_target_stats(torch.tensor([[[3.0]]]))

    _assert_allclose(module.target_mean, torch.tensor([3.0]))
    _assert_allclose(module.target_std, torch.tensor([1e-8]))
    assert torch.isfinite(module.normalize_targets(torch.tensor([[[3.0]]]))).all()


def test_interaction_learner_standardizes_targets_but_exposes_physical_output():
    config = Config(
        device="cpu",
        dim=2,
        n_agents=8,
        n_interaction_steps=5,
        learning_rate_interaction=1e-3,
        standardize_interaction_targets=True,
        interaction_input_scale=2.0,
    )

    def cost_fn(x, empirical):
        return 2.0 + 3.0 * x[:, :1]

    learner = InteractionLearner(config, None, cost_fn)
    xs = torch.randn(3, 8, 2)
    ts = torch.linspace(0.0, 1.0, 3)
    learner.train(xs, ts)

    assert isinstance(learner.F_net, TransformedInteractionModule)
    true_targets = torch.stack([cost_fn(x, x) for x in xs])
    _assert_allclose(learner.F_net.target_mean, true_targets.reshape(-1, 1).mean(0))
    _assert_allclose(
        learner.F_net.target_std,
        true_targets.reshape(-1, 1).std(0, correction=0),
    )
    assert "target_mean" in learner.F_net.state_dict()
    assert "target_std" in learner.F_net.state_dict()


def test_pedestrian_visual_cost_excludes_self_interaction():
    cost = PedestrianVisualCost(range_scale=1.0, weight=1.0)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    actual = cost.evaluate(x, x)

    expected = torch.tensor(
        [
            [math.exp(-1.0)],
            [0.0],
        ]
    )
    _assert_allclose(actual, expected)


def test_pedestrian_visual_cost_matches_direct_pairwise_formula():
    torch.manual_seed(21)
    x = torch.randn(6, 2)
    cost = PedestrianVisualCost(range_scale=0.7, weight=1.3)

    actual = cost.evaluate(x, x)

    expected = []
    goal = torch.tensor([7.0, 0.0])
    for i in range(x.shape[0]):
        desired = (goal - x[i]) / (torch.linalg.vector_norm(goal - x[i]) + 1e-6)
        terms = []
        for j in range(x.shape[0]):
            if i == j:
                continue
            diff = x[j] - x[i]
            dist = torch.linalg.vector_norm(diff)
            direction = diff / (dist + 1e-6)
            visibility = 0.5 * (1.0 + torch.dot(desired, direction))
            terms.append(visibility * torch.exp(-dist / 0.7))
        expected.append(1.3 * torch.stack(terms).mean())
    _assert_allclose(actual, torch.stack(expected).unsqueeze(1))


def test_pedestrian_visual_batched_evaluation_matches_slice_loop():
    torch.manual_seed(22)
    x = torch.randn(4, 7, 2)
    cost = PedestrianVisualCost(range_scale=1.1, weight=0.8)
    cost.set_chunk_sizes(query_chunk_size=3, ref_chunk_size=4)

    expected = torch.stack([cost.evaluate(x_t, x_t) for x_t in x])
    actual = cost.evaluate_batched(x, x, time_block_size=2)

    _assert_allclose(actual, expected)


def test_create_interactions_supports_bounded_confidence_drift():
    config = Config(
        device="cpu",
        n_agents=8,
        drift_type="bounded_confidence",
        cost_type="zero",
        int_drift_sigma=1.3,
        int_drift_weight=0.7,
        interaction_backend="exact",
    )

    _, _, _, _, analytical_drift, _ = create_interactions(config)

    assert analytical_drift.drift.name == "bounded_confidence"
    assert analytical_drift.drift.sigma == 1.3
    assert analytical_drift.drift.weight == 0.7


def test_create_interactions_supports_nonstationary_anisotropic_drift():
    config = Config(
        device="cpu",
        n_agents=8,
        drift_type="nonstationary_anisotropic",
        cost_type="zero",
        int_drift_sigma=1.3,
        int_drift_weight=0.7,
        int_drift_strength=0.4,
        interaction_backend="exact",
    )

    _, _, _, _, analytical_drift, _ = create_interactions(config)

    assert analytical_drift.drift.name == "nonstationary_anisotropic"
    assert analytical_drift.drift.sigma == 1.3
    assert analytical_drift.drift.weight == 0.7
    assert analytical_drift.drift.alpha == 0.4


def test_create_interactions_preserves_zero_nonstationary_anisotropic_strength():
    config = Config(
        device="cpu",
        n_agents=8,
        drift_type="nonstationary_anisotropic",
        cost_type="zero",
        int_drift_sigma=1.3,
        int_drift_weight=0.7,
        int_drift_strength=0.0,
        interaction_backend="exact",
    )

    _, _, _, _, analytical_drift, _ = create_interactions(config)

    assert analytical_drift.drift.alpha == 0.0


def test_nonstationary_anisotropic_zero_strength_matches_gaussian_drift():
    torch.manual_seed(12)
    x = torch.randn(7, 2)
    empirical = torch.randn(9, 2)
    gaussian = GaussianDrift(sigma=1.2, weight=0.8)
    anisotropic = NonstationaryAnisotropicDrift(
        sigma=1.2, weight=0.8, alpha=0.0
    )

    _assert_allclose(gaussian.evaluate(x, empirical), anisotropic.evaluate(x, empirical))


def test_gaussian_drift_time_block_evaluation_matches_slice_loop():
    torch.manual_seed(2)
    drift = GaussianDrift(sigma=0.8, weight=1.1)
    drift.set_chunk_sizes(query_chunk_size=3, ref_chunk_size=4)
    x = torch.randn(5, 7, 2)

    expected = torch.stack([drift.evaluate(x_t, x_t) for x_t in x], dim=0)
    actual = drift.evaluate_batched(x, x, time_block_size=2)
    _assert_allclose(expected, actual)


def test_all_supported_drift_time_block_evaluation_matches_slice_loop():
    torch.manual_seed(4)
    x = torch.randn(4, 6, 3)
    drifts = [
        MeanReversionDrift(weight=0.4),
        ZeroDrift(),
        AttractionRepulsionDrift(
            attract=1.1,
            sigma_attract=0.7,
            repel=0.5,
            sigma_repel=0.2,
            weight=0.8,
        ),
        CoulombDrift(alpha=1.2, eps=0.07, weight=0.6),
        NonstationaryAnisotropicDrift(sigma=0.9, weight=0.3, alpha=0.6),
        LaplacianDiffusionDrift(sigma=0.9, kernel_type="gaussian", weight=0.3),
        LaplacianDiffusionDrift(sigma=0.9, kernel_type="exponential", weight=0.3),
        LaplacianDiffusionDrift(sigma=0.9, kernel_type="inverse", weight=0.3),
        SmoothBoundedConfidenceDrift(sigma=0.8, weight=1.2),
    ]

    for drift in drifts:
        drift.set_chunk_sizes(query_chunk_size=2, ref_chunk_size=4)
        expected = torch.stack([drift.evaluate(x_t, x_t) for x_t in x], dim=0)
        actual = drift.evaluate_batched(x, x, time_block_size=3)
        _assert_allclose(expected, actual)


def test_drifts_declare_random_batch_aggregation():
    drifts = [
        ZeroDrift(),
        MeanReversionDrift(weight=0.4),
        GaussianDrift(sigma=0.8, weight=1.1),
        NonstationaryAnisotropicDrift(sigma=0.8, weight=1.1, alpha=0.5),
        AttractionRepulsionDrift(weight=0.8),
        CoulombDrift(alpha=1.2, eps=0.07, weight=0.6),
        LaplacianDiffusionDrift(sigma=0.9, kernel_type="gaussian", weight=0.3),
        SmoothBoundedConfidenceDrift(sigma=0.8, weight=1.2),
    ]

    for drift in drifts:
        assert "random_batch_aggregation" in type(drift).__dict__
        assert drift.random_batch_aggregation in {"mean", "sum"}


def test_all_drift_subclasses_declare_random_batch_aggregation():
    def subclasses(cls):
        for subcls in cls.__subclasses__():
            yield subcls
            yield from subclasses(subcls)

    for drift_cls in subclasses(Drift):
        assert "random_batch_aggregation" in drift_cls.__dict__, drift_cls.__name__
        assert drift_cls.random_batch_aggregation in {"mean", "sum"}, drift_cls.__name__


class _SumDrift(torch.nn.Module):
    random_batch_aggregation = "sum"

    def evaluate(self, x, empirical):
        return empirical.sum(dim=0, keepdim=True).expand_as(x)

    def evaluate_batched(self, x, empirical, *, time_block_size=None):
        return empirical.sum(dim=1, keepdim=True).expand_as(x)


def test_random_batch_drift_rescales_sum_subset():
    x = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    rb_drift = RandomBatchDrift(_SumDrift(), sample_size=2)
    rb_drift._sample_empirical_subset = lambda empirical, take: empirical[:take]

    actual = rb_drift.forward(x, x.new_zeros(x.shape[0]))
    expected = _SumDrift().evaluate(x, x[:2]) * (x.shape[0] / 2.0)

    _assert_allclose(actual, expected)


def test_entropy_and_congestion_time_block_evaluation_matches_slice_loop():
    torch.manual_seed(3)
    x = torch.randn(4, 6, 2)
    costs = [
        EntropyCost(weight=0.9),
        CongestionCost(weight=1.2),
    ]

    for cost in costs:
        cost.set_chunk_sizes(query_chunk_size=2, ref_chunk_size=5)
        expected = torch.stack([cost.evaluate(x_t, x_t) for x_t in x], dim=0)
        actual = cost.evaluate_batched(x, x, time_block_size=3)
        _assert_allclose(expected, actual)


def test_entropy_cost_uses_normalized_gaussian_kde_density():
    cost = EntropyCost(weight=1.0)
    x = torch.zeros(1, 1)

    actual = cost.evaluate(x, x)

    expected_log_density = -0.5 * math.log(2.0 * math.pi)
    expected = torch.tensor([[expected_log_density + 1.0]])
    _assert_allclose(actual, expected)


def test_all_supported_cost_time_block_evaluation_matches_slice_loop():
    torch.manual_seed(5)
    x = torch.randn(3, 5, 2)
    costs = [
        QuadraticDeviationCost(weight=0.4),
        MeanDeviationCost(weight=0.7),
        QuadraticPotentialCost(weight=0.5),
        CoulombCost(alpha=1.3, eps=0.05),
        LocalExclusionCost(epsilon=0.4, strength=0.6),
    ]

    for cost in costs:
        cost.set_chunk_sizes(query_chunk_size=2, ref_chunk_size=3)
        expected = torch.stack([cost.evaluate(x_t, x_t) for x_t in x], dim=0)
        actual = cost.evaluate_batched(x, x, time_block_size=2)
        _assert_allclose(expected, actual)


def test_cost_potentials_declare_random_batch_aggregation():
    costs = [
        ZeroCost(),
        QuadraticDeviationCost(weight=0.4),
        GaussianConvolutionCost(sigma=0.6, weight=1.2),
        GaussianConvolutionMeanCost(sigma=0.6, weight=1.2),
        MeanDeviationCost(weight=0.7),
        QuadraticPotentialCost(weight=0.5),
        CoulombCost(alpha=1.3, eps=0.05),
        LocalExclusionCost(epsilon=0.4, strength=0.6),
        EntropyCost(weight=0.9),
        CongestionCost(weight=1.2),
        FlockingAlignmentCost(sigma=0.8, weight=0.3),
    ]

    for cost in costs:
        assert "random_batch_aggregation" in type(cost).__dict__
        assert cost.random_batch_aggregation in {"mean", "sum"}


def test_all_cost_potential_subclasses_declare_random_batch_aggregation():
    def subclasses(cls):
        for subcls in cls.__subclasses__():
            yield subcls
            yield from subclasses(subcls)

    for cost_cls in subclasses(CostPotential):
        assert "random_batch_aggregation" in cost_cls.__dict__, cost_cls.__name__
        assert cost_cls.random_batch_aggregation in {"mean", "sum"}, cost_cls.__name__


def test_random_batch_cost_rescales_gaussian_convolution_subset_sum():
    torch.manual_seed(6)
    x = torch.randn(5, 2)
    cost = GaussianConvolutionCost(sigma=0.9, weight=1.3)
    rb_cost = RandomBatchCost(cost, sample_size=2)
    rb_cost._sample_empirical_subset = lambda empirical, take: empirical[:take]

    actual = rb_cost.forward(x, x.new_zeros(x.shape[0]))
    expected = cost.evaluate(x, x[:2]) * (x.shape[0] / 2.0)

    _assert_allclose(actual, expected)


def test_random_batch_cost_does_not_rescale_mean_costs():
    torch.manual_seed(7)
    x = torch.randn(5, 2)
    cost = QuadraticPotentialCost(weight=0.8)
    rb_cost = RandomBatchCost(cost, sample_size=2)
    rb_cost._sample_empirical_subset = lambda empirical, take: empirical[:take]

    actual = rb_cost.forward(x, x.new_zeros(x.shape[0]))
    expected = cost.evaluate(x, x[:2])

    _assert_allclose(actual, expected)


def test_random_batch_cost_does_not_rescale_gaussian_convolution_mean():
    torch.manual_seed(11)
    x = torch.randn(5, 2)
    cost = GaussianConvolutionMeanCost(sigma=0.9, weight=1.3)
    rb_cost = RandomBatchCost(cost, sample_size=2)
    rb_cost._sample_empirical_subset = lambda empirical, take: empirical[:take]

    actual = rb_cost.forward(x, x.new_zeros(x.shape[0]))
    expected = cost.evaluate(x, x[:2])

    _assert_allclose(actual, expected)


def test_random_batch_cost_rescales_batched_gaussian_convolution_subset_sum():
    torch.manual_seed(8)
    x = torch.randn(3, 5, 2)
    cost = GaussianConvolutionCost(sigma=1.1, weight=0.7)
    rb_cost = RandomBatchCost(cost, sample_size=2)
    rb_cost._sample_empirical_subset_batched = lambda empirical, take: empirical[:, :take]

    actual = rb_cost.evaluate_batched(x, x, time_block_size=2)
    expected = cost.evaluate_batched(x, x[:, :2], time_block_size=2) * (x.shape[1] / 2.0)

    _assert_allclose(actual, expected)


def test_random_batch_cost_does_not_rescale_batched_gaussian_convolution_mean():
    torch.manual_seed(12)
    x = torch.randn(3, 5, 2)
    cost = GaussianConvolutionMeanCost(sigma=1.1, weight=0.7)
    rb_cost = RandomBatchCost(cost, sample_size=2)
    rb_cost._sample_empirical_subset_batched = lambda empirical, take: empirical[:, :take]

    actual = rb_cost.evaluate_batched(x, x, time_block_size=2)
    expected = cost.evaluate_batched(x, x[:, :2], time_block_size=2)

    _assert_allclose(actual, expected)


def test_random_batch_opinion_drift_preserves_time_dependence():
    dim = 2
    xi = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    f_mul = torch.ones(2)
    drift = OpinionMeanFieldDrift(
        xi_schedule=xi,
        f_mul=f_mul,
        dt=1.0,
        T=1.0,
        polarize_strength=1.0,
    )
    rb_drift = RandomBatchDrift(drift, sample_size=4)
    x = torch.tensor(
        [
            [2.0, 1.0],
            [1.0, 2.0],
            [-2.0, 1.0],
            [-1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    t0 = torch.zeros(x.shape[0])
    t1 = torch.ones(x.shape[0])

    actual_t0 = rb_drift(x, t0)
    actual_t1 = rb_drift(x, t1)

    _assert_allclose(actual_t0, drift.evaluate(x, x, t0))
    _assert_allclose(actual_t1, drift.evaluate(x, x, t1))
    assert not torch.allclose(actual_t0, actual_t1)


def test_random_batch_opinion_drift_batched_passes_time_grid():
    dim = 2
    xi = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    f_mul = build_opinion_f_mul(n_steps=2, T=1.0)
    drift = OpinionMeanFieldDrift(
        xi_schedule=xi,
        f_mul=f_mul,
        dt=1.0,
        T=1.0,
        polarize_strength=1.0,
    )
    rb_drift = RandomBatchDrift(drift, sample_size=4)
    x = torch.tensor(
        [
            [[2.0, 1.0], [1.0, 2.0], [-2.0, 1.0], [-1.0, 2.0]],
            [[2.0, 1.0], [1.0, 2.0], [-2.0, 1.0], [-1.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    time_grid = torch.tensor([0.0, 1.0])

    actual = rb_drift.evaluate_batched(x, x, time_grid=time_grid, time_block_size=1)
    expected = drift.evaluate_batched(x, x, time_grid=time_grid, time_block_size=1)

    _assert_allclose(actual, expected)
    assert not torch.allclose(actual[0], actual[1])


class _BatchedOnlyDrift:
    def evaluate_batched(self, x, empirical, *, time_block_size=None):
        assert time_block_size == 2
        return x + 1.0


class _BatchedOnlyCost:
    def evaluate_batched(self, x, empirical, *, time_block_size=None):
        assert time_block_size == 2
        return x.sum(dim=-1, keepdim=True)


def test_interaction_learner_uses_batched_exact_targets_when_available():
    config = Config(
        dim=2,
        T=0.2,
        dt=0.1,
        n_agents=3,
        device="cpu",
        learning_rate_interaction=1e-3,
        interaction_time_block_size=2,
    )

    def f_interaction_fn(x, empirical, t):
        raise AssertionError("loop fallback should not be used for drift targets")

    def F_interaction_fn(x, empirical):
        raise AssertionError("loop fallback should not be used for cost targets")

    setattr(f_interaction_fn, "base_drift", _BatchedOnlyDrift())
    setattr(F_interaction_fn, "base_cost", _BatchedOnlyCost())

    learner = InteractionLearner(config, f_interaction_fn, F_interaction_fn)
    xs = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
    time_grid = torch.linspace(0, config.T, 3)

    f_targets, F_targets = learner.compute_true_interactions(xs, time_grid)

    _assert_allclose(f_targets, xs + 1.0)
    _assert_allclose(F_targets, xs.sum(dim=-1, keepdim=True))
