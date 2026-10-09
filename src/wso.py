"""Binary White Shark Optimization (WSO) for Feature Selection.

Based on:
- Braik et al. (2022): "White Shark Optimizer: A novel bio-inspired meta-heuristic algorithm for global optimization problems"
- Albahadili et al. (2024): "Detection of phishing URLs with deep learning based on GAN-CNN-LSTM network and swarm intelligence algorithms", Eqs. 8-25.

Algorithm Mechanics:
1. Candidate representation: Binary vector of length d (1 = keep CNN feature, 0 = discard).
2. Continuous kinematic state: Position x_i in R^d and velocity v_i in R^d.
3. Velocity update (Eq. 8):
     v_i^(t+1) = mu * [v_i^t + p1 * (X_gbest - x_i^t) * c1 + p2 * (X_pbest^i - x_i^t) * c2]
4. Wave motion and moving force update (Eq. 12-17).
5. Prey following and fish schooling behavior (Eq. 18-21).
6. Transfer function binarization (V-shape arctan Eq. 23 or S-shape sigmoid Eq. 22).
7. Fitness evaluation via MLP classifier (Eq. 25): Cost = alpha * E + beta * (k / d).
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class WSOResult:
    """Encapsulates the final outcome of WSO feature selection."""
    best_mask: np.ndarray             # 1D binary array of shape (d,)
    selected_indices: np.ndarray      # Indices of features where mask == 1
    best_fitness: float               # Lowest objective cost achieved
    total_features: int               # Original feature dimension d
    num_selected: int                 # Count of selected features k
    history: List[Dict[str, float]] = field(default_factory=list)  # Cost per iteration

    def to_dict(self) -> Dict:
        return {
            "num_selected": self.num_selected,
            "total_features": self.total_features,
            "selection_ratio": float(self.num_selected / self.total_features) if self.total_features > 0 else 0.0,
            "best_fitness": float(self.best_fitness),
            "selected_indices": self.selected_indices.tolist(),
            "best_mask": self.best_mask.tolist(),
            "history": self.history,
        }

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves optimization result to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class WhiteSharkOptimizer:
    """Binary White Shark Optimizer for deep feature subset selection."""

    def __init__(
        self,
        num_features: int,
        population_size: int = 20,
        max_iterations: int = 10,
        p_min: float = 0.5,
        p_max: float = 1.5,
        tau: float = 4.125,
        a0: float = 1.0,
        a1: float = 1.0,
        a2: float = 1.0,
        f_min: float = 0.07,
        f_max: float = 0.75,
        transfer_function: str = "v_shape",
        random_seed: int = 42,
    ):
        """Initializes the WSO meta-heuristic.

        Args:
            num_features: Dimension d of feature vectors to optimize over.
            population_size: Number of white sharks in the swarm (default: 20).
            max_iterations: Maximum optimization iterations (default: 10).
            p_min: Lower bound for prey movement parameter (default: 0.5, paper Eq. 9).
            p_max: Upper bound for prey movement parameter (default: 1.5, paper Eq. 10).
            tau: Acceleration coefficient (default: 4.125, paper Eq. 11).
            a0, a1, a2: Algorithmic exploration/exploitation control constants.
            f_min, f_max: Wavy frequency movement bounds (paper Eq. 16).
            transfer_function: 'v_shape' (arctan, Eq. 23) or 's_shape' (sigmoid, Eq. 22).
            random_seed: Random seed for reproducible stochastic exploration.
        """
        self.num_features = int(num_features)
        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.tau = float(tau)
        self.a0 = float(a0)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.transfer_function = transfer_function.lower()
        self.random_seed = int(random_seed)

        # Precompute contraction coefficient mu (paper Eq. 11)
        # mu = 2 / |2 - tau - sqrt(tau^2 - tau)|
        discriminant = self.tau**2 - self.tau
        sqrt_term = np.sqrt(discriminant) if discriminant >= 0 else 0.0
        denom = abs(2.0 - self.tau - sqrt_term)
        self.mu = 2.0 / denom if denom != 0 else 0.35

        # Precompute wavy frequency f (paper Eq. 16)
        denom_f = self.f_max + self.f_min
        self.f = self.f_min + (self.f_max - self.f_min) / denom_f if denom_f != 0 else 0.8

        self.rng = np.random.default_rng(self.random_seed)

    def _apply_transfer_function(self, values: np.ndarray) -> np.ndarray:
        """Applies S-shape or V-shape transfer function to map continuous values to probabilities."""
        if self.transfer_function == "v_shape":
            # Paper Eq. 23: T(x) = | (2 / pi) * arctan((pi / 2) * x) |
            return np.abs((2.0 / np.pi) * np.arctan((np.pi / 2.0) * values))
        else:
            # Paper Eq. 22: T(x) = 1 / (1 + e^(-x))
            # Clip to prevent overflow
            clipped = np.clip(values, -20.0, 20.0)
            return 1.0 / (1.0 + np.exp(-clipped))

    def _binarize(self, continuous_position: np.ndarray) -> np.ndarray:
        """Transforms continuous position into a valid binary mask (paper Eq. 24).

        Guarantees that at least one feature is selected (handles empty mask safely).
        """
        prob = self._apply_transfer_function(continuous_position)
        mask = (prob >= 0.5).astype(np.int32)

        # Safety: if mask has 0 features, activate the feature with highest probability
        if mask.sum() == 0:
            best_idx = int(np.argmax(prob))
            mask[best_idx] = 1

        return mask

    def optimize(
        self,
        fitness_func: Callable[[np.ndarray], float],
        verbose: bool = True,
    ) -> WSOResult:
        """Executes the White Shark Optimization algorithm for feature selection.

        Args:
            fitness_func: Callable taking a 1D binary mask of shape (d,) and returning a float cost.
            verbose: If True, prints iteration logs.

        Returns:
            WSOResult containing the optimal mask, indices, best cost, and search history.
        """
        d = self.num_features
        N = self.population_size
        MaxIter = self.max_iterations

        # 1. Initialize Continuous Population X and Velocities V
        # Positions initialized uniformly in [-1, 1]
        X = self.rng.uniform(-1.0, 1.0, size=(N, d))
        V = np.zeros((N, d), dtype=np.float64)

        # 2. Evaluate Initial Population
        pbest_positions = np.copy(X)
        pbest_masks = np.zeros((N, d), dtype=np.int32)
        pbest_fitnesses = np.full(N, np.inf)

        gbest_position = np.zeros(d, dtype=np.float64)
        gbest_mask = np.ones(d, dtype=np.int32)
        gbest_fitness = np.inf

        for i in range(N):
            mask_i = self._binarize(X[i])
            cost_i = fitness_func(mask_i)
            pbest_masks[i] = mask_i
            pbest_fitnesses[i] = cost_i

            if cost_i < gbest_fitness:
                gbest_fitness = cost_i
                gbest_position = np.copy(X[i])
                gbest_mask = np.copy(mask_i)

        history: List[Dict[str, float]] = []
        if verbose:
            print(f"[WSO] Init: Best Cost = {gbest_fitness:.5f} | Selected Feats = {gbest_mask.sum()}/{d}")

        # 3. Main Iterative Optimization Loop
        for t in range(1, MaxIter + 1):
            # Dynamic movement coefficients (Eqs. 9, 10)
            exp_term = np.exp(-((4.0 * t / MaxIter) ** 2))
            p1 = self.p_max + (self.p_max - self.p_min) * exp_term
            p2 = self.p_min + (self.p_max - self.p_min) * exp_term

            # Moving force mv (Eq. 17)
            exp_mv = np.exp(((MaxIter / 2.0) - t) / self.a1)
            mv = 1.0 / (self.a0 + exp_mv)

            # Sensory strength Ss (Eq. 20)
            Ss = abs(1.0 - np.exp(-self.a2 * t / MaxIter))

            # Loop over sharks to update velocity and position
            for i in range(N):
                c1 = self.rng.uniform(0.0, 1.0, size=d)
                c2 = self.rng.uniform(0.0, 1.0, size=d)

                # Velocity Update (Eq. 8)
                V[i] = self.mu * (
                    V[i]
                    + p1 * (gbest_position - X[i]) * c1
                    + p2 * (pbest_positions[i] - X[i]) * c2
                )

                # Position Update (Eq. 12)
                r_mv = self.rng.uniform(0.0, 1.0)
                if r_mv < mv:
                    # Wave exploration behavior
                    # Boundary disturbance vectors
                    a = (X[i] > 1.0).astype(np.float64)
                    b = (X[i] < -1.0).astype(np.float64)
                    w0 = np.logical_xor(a > 0, b > 0).astype(np.float64)
                    X[i] = X[i] * (1.0 - w0) + (1.0 * a + (-1.0) * b)
                else:
                    # Speed wave movement
                    X[i] = X[i] + (V[i] / self.f)

                # Prey following & school behavior (Eq. 18-21)
                r_ss = self.rng.uniform(0.0, 1.0)
                if r_ss <= Ss:
                    r1 = self.rng.uniform(0.0, 1.0, size=d)
                    r2 = self.rng.uniform(0.0, 1.0, size=d)
                    r3 = self.rng.uniform(0.0, 1.0)
                    D_vec = np.abs(r3 * (gbest_position - X[i]))
                    sgn_term = np.sign(r2 - 0.5)

                    X_prime = gbest_position + r1 * D_vec * sgn_term

                    if i == 0:
                        X[i] = X_prime
                    else:
                        r_rand = max(self.rng.uniform(0.0, 1.0), 0.01)
                        X[i] = (X_prime + X[i]) / (2.0 * r_rand)

                # Boundary clamping
                X[i] = np.clip(X[i], -10.0, 10.0)

                # Binarize and Evaluate
                mask_i = self._binarize(X[i])
                cost_i = fitness_func(mask_i)

                # Update Personal Best
                if cost_i < pbest_fitnesses[i]:
                    pbest_fitnesses[i] = cost_i
                    pbest_positions[i] = np.copy(X[i])
                    pbest_masks[i] = np.copy(mask_i)

                # Update Global Best
                if cost_i < gbest_fitness:
                    gbest_fitness = cost_i
                    gbest_position = np.copy(X[i])
                    gbest_mask = np.copy(mask_i)

            history.append({
                "iteration": t,
                "best_cost": float(gbest_fitness),
                "mean_cost": float(np.mean(pbest_fitnesses)),
                "num_selected": int(gbest_mask.sum()),
            })

            if verbose:
                print(
                    f"[WSO] Iter {t:02d}/{MaxIter:02d} | Best Cost: {gbest_fitness:.5f} | "
                    f"Mean Cost: {np.mean(pbest_fitnesses):.5f} | "
                    f"Selected: {gbest_mask.sum()}/{d}"
                )

        selected_indices = np.where(gbest_mask == 1)[0]
        return WSOResult(
            best_mask=gbest_mask,
            selected_indices=selected_indices,
            best_fitness=float(gbest_fitness),
            total_features=d,
            num_selected=len(selected_indices),
            history=history,
        )
