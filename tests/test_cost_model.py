"""Tests for cascade.cost_model module."""

from __future__ import annotations

from cascade.cost_model import CostModel, TokenPricing


class TestTokenPricing:
    """Tests for the TokenPricing dataclass."""

    def test_cost_calculation(self):
        pricing = TokenPricing(input_per_1k=0.005, output_per_1k=0.015)
        # 1000 input + 500 output
        cost = pricing.cost(1000, 500)
        expected = 1.0 * 0.005 + 0.5 * 0.015
        assert abs(cost - expected) < 1e-9

    def test_zero_tokens(self):
        pricing = TokenPricing(input_per_1k=0.005, output_per_1k=0.015)
        assert pricing.cost(0, 0) == 0.0


class TestCostModel:
    """Tests for the CostModel class."""

    def test_default_pricing_includes_gpt4o(self):
        model = CostModel()
        pricing = model.get_pricing("gpt-4o")
        assert pricing.input_per_1k == 0.005
        assert pricing.output_per_1k == 0.015

    def test_default_pricing_includes_sonnet(self):
        model = CostModel()
        pricing = model.get_pricing("sonnet")
        assert pricing.input_per_1k == 0.003
        assert pricing.output_per_1k == 0.015

    def test_unknown_model_falls_back_to_default(self):
        model = CostModel()
        pricing = model.get_pricing("some-future-model")
        gpt4o = model.get_pricing("gpt-4o")
        assert pricing.input_per_1k == gpt4o.input_per_1k
        assert pricing.output_per_1k == gpt4o.output_per_1k

    def test_step_cost(self):
        model = CostModel()
        cost = model.step_cost("sonnet", input_tokens=500, output_tokens=200)
        expected = 500 / 1000 * 0.003 + 200 / 1000 * 0.015
        assert abs(cost - expected) < 1e-9

    def test_pipeline_cost(self):
        model = CostModel()
        steps = [
            ("sonnet", 500, 200),
            ("haiku", 500, 200),
        ]
        cost = model.pipeline_cost(steps)
        expected_sonnet = 500 / 1000 * 0.003 + 200 / 1000 * 0.015
        expected_haiku = 500 / 1000 * 0.00025 + 200 / 1000 * 0.00125
        assert abs(cost - (expected_sonnet + expected_haiku)) < 1e-9

    def test_register_custom_model(self):
        model = CostModel()
        model.register("my-model", input_per_1k=0.01, output_per_1k=0.05)
        pricing = model.get_pricing("my-model")
        assert pricing.input_per_1k == 0.01
        assert pricing.output_per_1k == 0.05

    def test_register_overwrites(self):
        model = CostModel()
        model.register("sonnet", input_per_1k=0.999, output_per_1k=0.888)
        pricing = model.get_pricing("sonnet")
        assert pricing.input_per_1k == 0.999

    def test_custom_models_dict(self):
        custom = {
            "fast": TokenPricing(input_per_1k=0.001, output_per_1k=0.002),
        }
        model = CostModel(models=custom)
        pricing = model.get_pricing("fast")
        assert pricing.input_per_1k == 0.001
