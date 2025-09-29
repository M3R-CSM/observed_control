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


class IteratedObservedControl:
    """Implements the Observed Control algorithm for NMPC.

    This class is based on Algorithm 3 (Efficient Observed Control) from the
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
        final_control = initial_control.copy()

        for j in range(10):
            augmented_state = np.concatenate([initial_state.copy(), final_control])
            covariance_matrix = np.block([
                [np.zeros((self.n_x, self.n_x)), np.zeros((self.n_x, self.n_u))],
                [np.zeros((self.n_u, self.n_x)), 100 * self.process_noise],
            ])

            phi = np.eye(self.n_chi)
            accumulator = np.block([np.zeros((self.n_u, self.n_x)), 100 * self.process_noise])
            tr_init_cov = np.trace(covariance_matrix)
            horizon_time = current_time
            total_cost = 0
            du = 0 * final_control

            for k in range(self.max_horizon):
                # Predict step
                if k > 0:
                    augmented_state, covariance_matrix, phi = self._predict_step(
                        horizon_time, augmented_state, covariance_matrix
                    )

                # Update step
                r_k, R_k, H_k, state_cost = self._compute_residuals_and_gains(horizon_time, augmented_state)

                if k == 0:
                    J = np.linalg.pinv(R_k)
                    y_k = -J @ r_k
                    delta_u_cost = np.linalg.pinv(self.process_noise)
                    J[self.n_x:, self.n_x:] += delta_u_cost
                    y_k[self.n_x:] += delta_u_cost @ (final_control - initial_control)
                    R_k = np.linalg.pinv(J)
                    r_k = -R_k @ y_k

                total_cost += state_cost

                innovation_cov = H_k @ covariance_matrix @ H_k.T + R_k
                ht_innov_cov_inv = H_k.T @ np.linalg.pinv(innovation_cov)

                kalman_gain = covariance_matrix @ ht_innov_cov_inv
                augmented_state += kalman_gain @ r_k
                # print("delta_state: ", kalman_gain @ r_k)

                cl_system_t = (np.eye(self.n_chi) - kalman_gain @ H_k).T
                covariance_matrix @= cl_system_t

                # Accumulator and adaptive horizon update
                accumulator @= phi.T
                g_phi_s = accumulator @ ht_innov_cov_inv
                du += g_phi_s @ r_k

                dp = np.trace(g_phi_s @ H_k @ accumulator.T)
                tr_init_cov -= dp
                # print("tr_init_cov\n", tr_init_cov)
                trace_p_term_cond = dp / tr_init_cov  # if tr_init_cov > 1e-9 else 0.0
                gamma_term_cond = np.linalg.norm(g_phi_s, 'fro')

                if k >= self.min_horizon and (
                        trace_p_term_cond < self.tol_trace_p and gamma_term_cond < self.tol_gamma):
                    break

                accumulator @= cl_system_t
                horizon_time += self.expected_update_period

            print("j: ", j, " du: ", du)
            final_control += du

            if np.abs(du) < gamma_term_cond:
                break

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

    def _predict_step(self, t: float, aug_state: np.ndarray, cov: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Propagates the augmented state and covariance forward in time."""
        x_k = aug_state[:self.n_x]
        u_k = aug_state[self.n_x:]

        x_k_plus_1, phi_x, phi_u = self.dynamic_system.solve(
            t, t + self.expected_update_period, x_k.flatten(), u_k.flatten()
        )

        aug_state_k_plus_1 = np.concatenate([x_k_plus_1, u_k.flatten()])
        phi = np.block([[phi_x, phi_u], [np.zeros((self.n_u, self.n_x)), np.eye(self.n_u)]])
        cov_k_plus_1 = phi @ cov @ phi.T
        cov_k_plus_1[self.n_x:, self.n_x:] += self.process_noise

        return aug_state_k_plus_1, cov_k_plus_1, phi

    def _compute_residuals_and_gains(self, t: float, aug_state: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, float]:
        """Computes measurement residuals and gains from anticipated conditions."""
        x_k = aug_state[:self.n_x]
        u_k = aug_state[self.n_x:]

        total_hessian = np.zeros((self.n_chi, self.n_chi))
        total_residual = np.zeros(self.n_chi)

        # The cost is evaluated at the *end* of the time step
        total_cost = 0.0

        for magnitude, condition in self.anticipated_conditions:
            cond_cost = condition.value(t, x_k, u_k)
            cond_hess = condition.hessian(t, x_k, u_k)
            cond_res = condition.sensitivity(t, x_k, u_k)

            total_hessian += magnitude * np.reshape(cond_hess, (self.n_chi, self.n_chi))
            total_residual += magnitude * np.reshape(cond_res, (-1,))
            total_cost += magnitude * cond_cost

        R_k = np.linalg.pinv(total_hessian)
        H_k = R_k * total_hessian  # np.eye(self.n_chi)  # Observation model is identity
        r_k = -R_k @ total_residual

        return r_k, R_k, H_k, total_cost
