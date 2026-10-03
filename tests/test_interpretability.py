import torch
import torch.nn as nn

from llm_interpretability.hooks import HookManager
from llm_interpretability.cka import CKAAnalyzer
from llm_interpretability.outliers import OutlierDetector
from llm_interpretability.steering import ActivationSteering


class DummyTransformer(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=64, num_layers=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def test_hook_manager_lifecycle():
    model = DummyTransformer()
    with HookManager(model) as hm:
        hm.register_layer_hooks(layers=list(model.layers), track_timing=True)
        assert len(hm.handles) == 8  # 4 pre-hooks + 4 forward hooks

        input_ids = torch.randint(0, 100, (2, 8))
        _ = model(input_ids)

        assert len(hm.activations) == 4
        assert "layer_0" in hm.activations
        assert hm.activations["layer_0"].shape == (2, 8, 64)
        assert "layer_0" in hm.stats
        assert hm.stats["layer_0"].duration_ms >= 0.0

    # Ensure hooks are cleared on context exit
    assert len(hm.handles) == 0


def test_linear_cka_invariance():
    # Two identical representations should have CKA = 1.0
    x = torch.randn(20, 64)
    cka = CKAAnalyzer.linear_cka(x, x)
    assert abs(cka - 1.0) < 1e-4

    # Orthogonal representations should have low CKA
    y = torch.randn(20, 64)
    cka_diff = CKAAnalyzer.linear_cka(x, y)
    assert 0.0 <= cka_diff <= 1.0


def test_outlier_detector():
    detector = OutlierDetector(std_threshold=3.0, absolute_threshold=5.0)
    # Create activations with an extreme spike on dimension 7
    act = torch.randn(10, 16, 64)
    act[:, :, 7] = 25.0

    report = detector.analyze_layer(act, layer_name="layer_3")
    assert 7 in report.outlier_dimensions
    assert report.global_max >= 25.0


def test_activation_steering():
    model = DummyTransformer()
    steering = ActivationSteering(model)

    pos_act = torch.ones(5, 8, 64) * 2.0
    neg_act = torch.zeros(5, 8, 64)

    v = steering.compute_steering_vector(pos_act, neg_act, normalize=True)
    assert v.shape == (64,)
    assert abs(v.norm().item() - 1.0) < 1e-4

    hook = steering.attach_steering(model.layers[0], steering_vector=v, coefficient=2.0)
    assert len(steering.active_hooks) == 1

    input_ids = torch.randint(0, 100, (2, 4))
    _ = model(input_ids)

    steering.clear_steering()
    assert len(steering.active_hooks) == 0
