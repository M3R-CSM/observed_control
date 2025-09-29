# Copyright (c) 2025 Andrew Petruska
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# src/core/observed_control.py
from typing import List, Tuple

import numpy as np
import time
from core.dynamic_system import DynamicSystemBase
from core.anticipated_condition import AnticipatedConditionBase


class UnscentedObservedControl:
    """Implements the Observed Control algorithm for NMPC with an unscented backend.

    This class is based on Algorithm 2 (Forward-Only Observed Control) from the
    paper "Linearly Scalable Nonlinear Model Predictive Control with
    Adaptive Horizons".
    """

    def __init__(
            self,
            dynamic_system: DynamicSystemBase,
            anticipated_conditions: List[Tuple[float, AnticipatedConditionBase]],
            expected_update_period: float,
            min_horizon: int,
            max_horizon: int,
            adaptive_tolerances_trace_p: float,
            adaptive_tolerances_gamma: float,
            delta_control_penalty: np.ndarray,
            alpha: float = 1,
            beta: float = 0,
            kappa: float = None
    ):
        """Initializes the Observed Control instance."""
        self.dynamic_system = dynamic_system
        self.anticipated_conditions = anticipated_conditions
        self.expected_update_period = expected_update_period
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.tol_trace_p = adaptive_tolerances_trace_p
        self.tol_gamma = adaptive_tolerances_gamma
        self.process_noise = np.linalg.inv(delta_control_penalty)

        self.n_x = dynamic_system.n_x
        self.n_u = dynamic_system.n_u
        self.n_chi = self.n_x + self.n_u

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        if self.kappa is None:
            self.kappa = 3.0 * self.n_chi / 2.0  # https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter

        assert (self.kappa > 0)
        assert (self.alpha > 0)

        self.weights_0a = (self.alpha ** 2 * self.kappa - self.n_chi) / (self.alpha ** 2 * self.kappa)
        self.weights_0c = self.weights_0a + 1 - self.alpha ** 2 + self.beta
        self.weights_j = 1.0 / (2.0 * self.alpha ** 2 * self.kappa)
        self.sigma_pt_shift = self.alpha * np.sqrt(self.kappa)

    def control_law(self, current_time: float, initial_state: np.ndarray, initial_control: np.ndarray) -> np.ndarray:
        """ Computes the optimal control action and diagnostic information.

        This method implements the main loop of the Observed Control algorithm.

        Args:
            current_time: The time at which the control law is being called.
            initial_state: The current state of the system, x_k.
            initial_control: The initial guess for the control sequence, u_k.

        Returns:
            A tuple containing:
                - final_control (np.ndarray): The optimal control action.
                - diagnostics (Dict): A dictionary containing informational
                  outputs like cost, final horizon length, and termination
                  condition values, useful for analysis and debugging.
        """
        start_time = time.perf_counter()

        augmented_state = np.concatenate([initial_state, initial_control])
        covariance_matrix = np.block([
            [np.zeros((self.n_x, self.n_x)), np.zeros((self.n_x, self.n_u))],
            [np.zeros((self.n_u, self.n_x)), self.process_noise],
        ])

        accumulator = np.block([np.zeros((self.n_u, self.n_x)), np.eye(self.n_u)])
        final_control = initial_control.copy()
        tr_init_cov = np.trace(covariance_matrix)
        horizon_time = current_time
        total_cost = 0

        for k in range(self.max_horizon):
            # Predict step
            if k > 0:
                augmented_state, covariance_matrix, cross_covariance = self._predict_step(horizon_time, augmented_state,
                                                                                          covariance_matrix)
                information_matrix = np.linalg.pinv(covariance_matrix)
                accumulator = accumulator @ cross_covariance @ information_matrix
            else:
                information_matrix = np.linalg.pinv(covariance_matrix)

            # Update step
            mean_y, cov_yy, cov_xy = self._compute_residuals_and_gains(horizon_time, augmented_state, covariance_matrix)

            # Robust extraction of the weighting matrix from the coss-covariance
            u, s, vh = np.linalg.svd(information_matrix @ cov_xy)
            s[s <= 1e-14] = 0
            s[s > 1e-14] = s[s > 1e-14] ** -1
            R_inv = vh.T @ np.diag(s) @ vh  # extract weighting matrix, it should be symmetric
            s[s > 1e-14] = 1  # get observation matrix
            H = vh.T @ np.diag(s) @ vh

            innovation_cov = H @ covariance_matrix @ H.T + R_inv

            kalman_gain = covariance_matrix @ H.T @ np.linalg.pinv(innovation_cov)
            delta_state = -kalman_gain @ (R_inv @ mean_y)

            augmented_state += delta_state
            final_control += accumulator @ delta_state

            delta_cov = - kalman_gain @ innovation_cov @ kalman_gain.T
            dp = np.trace(accumulator @ delta_cov @ accumulator.T)

            covariance_matrix += delta_cov

            tr_init_cov += dp
            trace_p_term_cond = dp / tr_init_cov  # if tr_init_cov > 1e-9 else 0.0
            gamma_term_cond = np.linalg.norm(accumulator @ kalman_gain, 'fro')

            if k >= self.min_horizon and (trace_p_term_cond < self.tol_trace_p and gamma_term_cond < self.tol_gamma):
                break

            horizon_time += self.expected_update_period

        end_time = time.perf_counter()
        diagnostics = {
            'horizon cost': total_cost,
            'final_horizon': k,
            'final_trace_p': tr_init_cov,
            'trace_p_term_cond': trace_p_term_cond,
            'gamma_term_cond': gamma_term_cond,
            'compute_time': end_time - start_time
        }
        return final_control, diagnostics

    def _predict_step(self, t: float, aug_state: np.ndarray, cov_matrix: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Propagates the augmented state and covariance forward in time."""
        sigma_points = self._update_sigma_points(aug_state, cov_matrix, self.sigma_pt_shift)
        transformed_sigmal_points = list()
        for sig_pt in sigma_points:
            x_k = sig_pt[:self.n_x]
            u_k = sig_pt[self.n_x:]

            x_k_plus_1 = self.dynamic_system.solve_ode(t, t + self.expected_update_period, x_k.flatten(), u_k.flatten())

            aug_state_k_plus_1 = np.concatenate([x_k_plus_1, u_k.flatten()])
            transformed_sigmal_points.append(np.reshape(aug_state_k_plus_1, (-1,)))

        mean_y, cov_yy, cov_xy = self._compute_statistics(sigma_points, transformed_sigmal_points, self.weights_0a,
                                                     self.weights_0c, self.weights_j)
        cov_yy[self.n_x:, self.n_x:] += self.process_noise

        return mean_y, cov_yy, cov_xy

    def _compute_residuals_and_gains(self, t: float, aug_state: np.ndarray, cov_matrix: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, float]:
        """Computes measurement residuals and gains from anticipated conditions."""
        transformed_sigmal_points = list()
        sigma_points = self._update_sigma_points(aug_state, cov_matrix, self.sigma_pt_shift)
        for sig_pt in sigma_points:

            x_k = sig_pt[:self.n_x]
            u_k = sig_pt[self.n_x:]

            total_residual = np.zeros(self.n_chi)

            # The cost is evaluated at the *end* of the time step
            for magnitude, condition in self.anticipated_conditions:
                total_residual += magnitude * condition.sensitivity(t, x_k, u_k)

            transformed_sigmal_points.append(total_residual)

        return self._compute_statistics(sigma_points, transformed_sigmal_points, self.weights_0a, self.weights_0c,
                                   self.weights_j)

    def _update_sigma_points(self, mean: np.ndarray, covariance: np.ndarray, sigma_pt_shift: float) -> list():
        try:
            l = np.linalg.cholesky(covariance)
        except Exception:
            u, s, _ = np.linalg.svd(covariance)
            s[s < 1e-14] = 0
            l = u * np.sqrt(s)

        sigma_points = list()
        sigma_points.append(mean)
        for i in range(mean.shape[0]):
            sigma_points.append(mean + sigma_pt_shift * l[:, i].flatten())
            sigma_points.append(mean - sigma_pt_shift * l[:, i].flatten())

        return sigma_points

    def _compute_statistics(self, pts_1: list, pts_2: list, w0a: float, w0c: float, wj: float) -> [np.ndarray,
                                                                                                   np.ndarray,
                                                                                                   np.ndarray]:
        mean_1 = pts_1[0].copy()
        mean_2 = (w0a / wj) * pts_2[0].copy()
        for i in range(1, len(pts_2)):
            mean_2 += pts_2[i]
        mean_2 *= wj

        covariance_2 = (w0c / wj) * np.reshape(pts_2[0] - mean_2, (-1, 1)) @ np.reshape(pts_2[0] - mean_2, (1, -1))
        cross_covariance_12 = (w0c / wj) * np.reshape(pts_1[0] - mean_1, (-1, 1)) @ np.reshape(pts_2[0] - mean_2,
                                                                                               (1, -1))

        for i in range(1, len(pts_1)):
            covariance_2 += np.reshape(pts_2[i] - mean_2, (-1, 1)) @ np.reshape(pts_2[i] - mean_2, (1, -1))
            cross_covariance_12 += np.reshape(pts_1[i] - mean_1, (-1, 1)) @ np.reshape(pts_2[i] - mean_2, (1, -1))
        covariance_2 *= wj
        cross_covariance_12 *= wj

        return mean_2, covariance_2, cross_covariance_12
