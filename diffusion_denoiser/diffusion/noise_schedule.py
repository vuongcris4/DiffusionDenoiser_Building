"""Discrete noise schedules for D3PM.

Reference:
    Austin et al., "Structured Denoising Diffusion Models in Discrete
    State-Spaces", NeurIPS 2021.

In D3PM, the forward process is defined by transition matrices Q_t of
shape (K, K), where K is the number of classes. At each timestep t,
each pixel independently transitions from class i to class j with
probability Q_t[i, j].

This module provides several transition matrix strategies:
    - 'uniform':  With probability β_t, jump to any class uniformly.
    - 'absorbing': With probability β_t, jump to an absorbing (mask) state.
    - 'discretized_gaussian': Transition probabilities decay with class distance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class DiscreteNoiseSchedule(nn.Module):
    """Precompute and cache all transition matrices for D3PM.

    Given a noise schedule β_1, ..., β_T and a transition type, this module
    precomputes:
        Q_t:         single-step transition matrix at timestep t
        Q_bar_t:     cumulative product Q_1 * Q_2 * ... * Q_t
        Q_bar_t_inv: pseudo-inverse of Q_bar_t (for posterior computation)

    Args:
        num_classes (int): Number of segmentation classes K.
        num_timesteps (int): Total diffusion timesteps T.
        transition_type (str): One of 'uniform', 'absorbing'.
            Default: 'uniform'.
        beta_schedule (str): Schedule for β_t. One of 'linear', 'cosine'.
            Default: 'cosine'.
        beta_start (float): Starting β value (for linear schedule). Default: 1e-4.
        beta_end (float): Ending β value (for linear schedule). Default: 0.02.
    """

    def __init__(self,
                 num_classes: int,
                 num_timesteps: int = 100,
                 transition_type: str = 'uniform',
                 beta_schedule: str = 'cosine',
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.transition_type = transition_type

        # Compute beta schedule
        betas = self._get_beta_schedule(
            beta_schedule, num_timesteps, beta_start, beta_end)

        # Build single-step transition matrices Q_t: shape (T, K, K)
        Q_t = self._build_transition_matrices(betas, num_classes, transition_type)

        # Compute cumulative products Q_bar_t = Q_1 * Q_2 * ... * Q_t
        Q_bar = self._compute_cumulative_products(Q_t)

        # Register as buffers (not parameters, but moved to device with model)
        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float64))
        self.register_buffer('Q_t', Q_t.float())         # (T, K, K)
        self.register_buffer('Q_bar', Q_bar.float())      # (T, K, K)

    @staticmethod
    def _get_beta_schedule(schedule: str, T: int,
                           beta_start: float, beta_end: float) -> np.ndarray:
        """Compute noise schedule β_1, ..., β_T."""
        if schedule == 'linear':
            return np.linspace(beta_start, beta_end, T)
        elif schedule == 'cosine':
            # Cosine schedule from Nichol & Dhariwal (2021)
            steps = np.arange(T + 1, dtype=np.float64) / T
            alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            return np.clip(betas, 0, 0.999)
        else:
            raise ValueError(f'Unknown beta schedule: {schedule}')

    @staticmethod
    def _build_transition_matrices(betas: np.ndarray, K: int,
                                   transition_type: str) -> torch.Tensor:
        """Build single-step transition matrices.

        Args:
            betas: (T,) noise levels.
            K: number of classes.
            transition_type: 'uniform' or 'absorbing'.

        Returns:
            Tensor: (T, K, K) transition matrices.
        """
        T = len(betas)
        Q = torch.zeros(T, K, K, dtype=torch.float64)

        for t in range(T):
            beta_t = betas[t]

            if transition_type == 'uniform':
                # Q_t = (1 - β_t) * I + β_t * (1/K) * 1 * 1^T
                Q[t] = (1 - beta_t) * torch.eye(K, dtype=torch.float64) + \
                        beta_t / K * torch.ones(K, K, dtype=torch.float64)

            elif transition_type == 'absorbing':
                # Last class (K-1) is the absorbing (mask) state.
                # Non-absorbing states transition to absorbing with prob β_t.
                # Absorbing state stays absorbing with prob 1.
                Q[t] = (1 - beta_t) * torch.eye(K, dtype=torch.float64)
                Q[t, :K - 1, K - 1] += beta_t  # transition to mask
                Q[t, K - 1, K - 1] = 1.0        # mask stays mask

            else:
                raise ValueError(f'Unknown transition type: {transition_type}')

        return Q

    @staticmethod
    def _compute_cumulative_products(Q_t: torch.Tensor) -> torch.Tensor:
        """Compute Q_bar_t = Q_1 * Q_2 * ... * Q_t via sequential matmul.

        Args:
            Q_t: (T, K, K) single-step matrices.

        Returns:
            Tensor: (T, K, K) cumulative product matrices.
        """
        T, K, _ = Q_t.shape
        Q_bar = torch.zeros_like(Q_t)
        Q_bar[0] = Q_t[0]
        for t in range(1, T):
            Q_bar[t] = Q_bar[t - 1] @ Q_t[t]
        return Q_bar

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0) using the cumulative transition matrix.

        Args:
            x_0 (Tensor): Clean class indices (B, H, W), values in [0, K-1].
            t (Tensor): Timestep indices (B,), values in [0, T-1].

        Returns:
            Tensor: Noisy class indices x_t (B, H, W).
        """
        B, H, W = x_0.shape
        K = self.num_classes

        # Get cumulative transition probs for each sample in the batch
        # Q_bar[t]: (B, K, K)
        Q_bar_t = self.Q_bar[t]  # (B, K, K)

        # One-hot encode x_0: (B, H, W, K)
        x_0_onehot = torch.nn.functional.one_hot(
            x_0.long(), K).float()  # (B, H, W, K)

        # Compute transition probs: p(x_t | x_0) = x_0_onehot @ Q_bar_t^T
        # x_0_onehot: (B, H*W, K), Q_bar_t: (B, K, K)
        x_0_flat = x_0_onehot.view(B, H * W, K)
        probs = torch.bmm(x_0_flat, Q_bar_t.transpose(1, 2))  # (B, H*W, K)
        probs = probs.clamp(min=1e-10)

        # Sample from categorical distribution
        x_t = torch.multinomial(
            probs.view(-1, K), num_samples=1).view(B, H, W)

        return x_t

    def q_posterior(self, x_0: torch.Tensor, x_t: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        """Compute posterior q(x_{t-1} | x_t, x_0).

        Using Bayes' rule:
            q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)

        Args:
            x_0 (Tensor): Clean class indices (B, H, W).
            x_t (Tensor): Noisy class indices at step t (B, H, W).
            t (Tensor): Timestep indices (B,), values in [1, T-1].

        Returns:
            Tensor: Posterior probabilities (B, H, W, K).
        """
        B, H, W = x_0.shape
        K = self.num_classes

        # q(x_t | x_{t-1}) = Q_t[x_{t-1}, x_t]
        Q_t_cur = self.Q_t[t]  # (B, K, K)

        # q(x_{t-1} | x_0) from Q_bar_{t-1}
        # Handle t=0 edge case: Q_bar_{-1} = I
        t_minus_1 = (t - 1).clamp(min=0)
        Q_bar_prev = self.Q_bar[t_minus_1]  # (B, K, K)
        # For t=0, use identity
        is_t0 = (t == 0).float().view(B, 1, 1)
        Q_bar_prev = is_t0 * torch.eye(K, device=x_0.device).unsqueeze(0) + \
                     (1 - is_t0) * Q_bar_prev

        # One-hot encode
        x_0_oh = torch.nn.functional.one_hot(x_0.long(), K).float()  # (B, H, W, K)
        x_t_oh = torch.nn.functional.one_hot(x_t.long(), K).float()  # (B, H, W, K)

        # q(x_{t-1}=j | x_0): x_0_oh @ Q_bar_prev^T → (B, H, W, K) indexed by j
        x_0_flat = x_0_oh.view(B, H * W, K)
        prob_xtm1_given_x0 = torch.bmm(
            x_0_flat, Q_bar_prev.transpose(1, 2)).view(B, H, W, K)  # p(x_{t-1}=j | x_0)

        # q(x_t | x_{t-1}=j): Q_t[j, x_t] for each j
        # x_t_oh: (B, H, W, K), Q_t_cur: (B, K, K)
        # We need Q_t[j, :] @ x_t_oh for each pixel → (B, H*W, K)
        x_t_flat = x_t_oh.view(B, H * W, K)
        # Q_t_cur @ x_t_oh^T gives (B, K, H*W) where [b, j, n] = sum_k Q_t[j,k]*x_t[n,k] = Q_t[j, x_t[n]]
        prob_xt_given_xtm1 = torch.bmm(
            Q_t_cur, x_t_flat.transpose(1, 2)).transpose(1, 2).view(B, H, W, K)

        # Posterior: q(x_{t-1}=j | x_t, x_0) ∝ prob_xt_given_xtm1[j] * prob_xtm1_given_x0[j]
        unnormalized = prob_xt_given_xtm1 * prob_xtm1_given_x0
        posterior = unnormalized / (unnormalized.sum(dim=-1, keepdim=True) + 1e-10)

        return posterior
