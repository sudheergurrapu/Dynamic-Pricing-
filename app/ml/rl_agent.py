"""Simple Q-learning based pricing adjustment with business constraints."""

from __future__ import annotations

import numpy as np

from app.core.config import settings


class QLearningPricer:
    """Discrete action RL agent to fine-tune price."""

    def __init__(self) -> None:
        self.actions = np.array([-0.05, -0.02, 0.0, 0.02, 0.05])
        self.q_table = np.zeros((5, len(self.actions)))
        self.alpha = 0.15
        self.gamma = 0.9

    def _state(self, demand_index: float, inventory_level: int) -> int:
        demand_bucket = 0 if demand_index < 0.8 else (1 if demand_index < 1.1 else 2)
        stock_bucket = 0 if inventory_level < 100 else (1 if inventory_level < 500 else 2)
        return min(demand_bucket + stock_bucket, 4)

    def _reward(self, price: float, base_price: float, demand_index: float) -> float:
        conversion = np.clip(1.1 - 0.35 * ((price - base_price) / base_price), 0.2, 1.4)
        revenue = price * demand_index * conversion
        return revenue + 10 * conversion

    def adjust_price(self, model_price: float, base_price: float, demand_index: float, inventory_level: int) -> float:
        """Run one-step epsilon-greedy update and return bounded adjusted price."""
        state = self._state(demand_index, inventory_level)
        epsilon = 0.1

        if np.random.random() < epsilon:
            action_idx = np.random.randint(len(self.actions))
        else:
            action_idx = int(np.argmax(self.q_table[state]))

        multiplier = 1 + self.actions[action_idx]
        candidate = model_price * multiplier

        min_price = base_price * settings.min_price_factor
        max_price = base_price * settings.max_price_factor
        volatility_bound = model_price * settings.max_step_volatility
        bounded = float(np.clip(candidate, max(min_price, model_price - volatility_bound), min(max_price, model_price + volatility_bound)))

        reward = self._reward(bounded, base_price, demand_index)
        next_state = self._state(demand_index * 0.99, max(inventory_level - 1, 0))
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action_idx]
        self.q_table[state, action_idx] += self.alpha * td_error

        return bounded
