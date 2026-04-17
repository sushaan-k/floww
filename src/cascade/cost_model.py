"""Cost model for token-level pricing of LLM API calls.

Maps model names to per-token prices and provides helpers for
computing step- and pipeline-level cost estimates. Default pricing
is based on GPT-4o-class models so simulations produce realistic
dollar figures out of the box.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default pricing tiers (USD per 1K tokens) mirroring GPT-4o-class models.
_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "opus": (0.015, 0.075),
    "sonnet": (0.003, 0.015),
    "haiku": (0.00025, 0.00125),
}


@dataclass
class TokenPricing:
    """Per-token pricing for a single model.

    Attributes:
        input_per_1k: Cost in USD per 1,000 input tokens.
        output_per_1k: Cost in USD per 1,000 output tokens.
    """

    input_per_1k: float
    output_per_1k: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the total cost for a given token count.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Total cost in USD.
        """
        return (
            input_tokens / 1000 * self.input_per_1k
            + output_tokens / 1000 * self.output_per_1k
        )


@dataclass
class CostModel:
    """Maps model names to per-token prices.

    Provides a unified interface for looking up pricing across
    different model families. Falls back to GPT-4o pricing for
    unknown models by default.

    Attributes:
        models: Mapping of model name to TokenPricing.
        default_model: Model name to use as fallback for unknown names.
    """

    models: dict[str, TokenPricing] = field(default_factory=dict)
    default_model: str = "gpt-4o"

    def __post_init__(self) -> None:
        if not self.models:
            for name, (inp, out) in _DEFAULT_PRICING.items():
                self.models[name] = TokenPricing(input_per_1k=inp, output_per_1k=out)

    def get_pricing(self, model: str) -> TokenPricing:
        """Look up pricing for a model, falling back to the default.

        Args:
            model: Model name (e.g. "sonnet", "gpt-4o").

        Returns:
            TokenPricing for the model.
        """
        if model in self.models:
            return self.models[model]
        logger.debug(
            "No pricing for model %r, falling back to %r", model, self.default_model
        )
        return self.models[self.default_model]

    def step_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate the cost for a single step execution.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in USD.
        """
        pricing = self.get_pricing(model)
        return pricing.cost(input_tokens, output_tokens)

    def pipeline_cost(
        self,
        steps: list[tuple[str, int, int]],
    ) -> float:
        """Calculate total cost for a sequence of steps.

        Args:
            steps: List of (model, input_tokens, output_tokens) tuples.

        Returns:
            Total cost in USD.
        """
        return sum(self.step_cost(model, inp, out) for model, inp, out in steps)

    def register(
        self,
        model: str,
        input_per_1k: float,
        output_per_1k: float,
    ) -> None:
        """Register or update pricing for a model.

        Args:
            model: Model name.
            input_per_1k: Cost per 1K input tokens.
            output_per_1k: Cost per 1K output tokens.
        """
        self.models[model] = TokenPricing(
            input_per_1k=input_per_1k, output_per_1k=output_per_1k
        )
