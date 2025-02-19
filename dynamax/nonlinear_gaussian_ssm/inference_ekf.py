import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jscipy
from jax import lax
from jax import jacfwd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import List, Optional, NamedTuple, Optional, Union, Callable

from dynamax.utils.utils import psd_solve, symmetrize, inv_via_cholesky, rotate_subspace
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed
from dynamax.types import PRNGKey

# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x

FnStateToState = Callable[ [Float[Array, "state_dim"]], Float[Array, "state_dim"]]
FnStateAndInputToState = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "state_dim"]]
FnStateToEmission = Callable[ [Float[Array, "state_dim"]], Float[Array, "emission_dim"]]
FnStateAndInputToEmission = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"] ], Float[Array, "emission_dim"]]

class ParamsNLGSSM(NamedTuple):
    """Parameters for a NLGSSM model.

    $$p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$
    $$p(z_1) = N(z_1 | m, S)$$

    If you have no inputs, the dynamics and emission functions do not to take $u_t$ as an argument.

    :param dynamics_function: $f$
    :param dynamics_covariance: $Q$
    :param emissions_function: $h$
    :param emissions_covariance: $R$
    :param initial_mean: $m$
    :param initial_covariance: $S$

    """

    initial_mean: Float[Array, "state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    dynamics_covariance: Float[Array, "state_dim state_dim"]
    emission_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_covariance: Float[Array, "emission_dim emission_dim"]

def _predict(m, P, Q):
    r"""Predict next mean and covariance using first-order additive EKF

        p(z_{t+1}) = \int N(z_t | m, S) N(z_{t+1} | f(z_t, u), Q)
                    = N(z_{t+1} | f(m, u), F(m, u) S F(m, u)^T + Q)

    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        f (Callable): dynamics function.
        F (Callable): Jacobian of dynamics function.
        Q (D_hid,D_hid): dynamics covariance matrix.
        u (D_in,): inputs.

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    return m, P + Q


def _condition_on(m, P, h, H, R, u, y, num_iter):
    r"""Condition a Gaussian potential on a new observation.

       p(z_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(z_t | y_{1:t-1}, u_{1:t-1}) p(y_t | z_t, u_t)
         = N(z_t | m, S) N(y_t | h_t(z_t, u_t), R_t)
         = N(z_t | mm, SS)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = h(m, u)
         S = R + H(m,u) * P * H(m,u)'
         K = P * H(m, u)' * S^{-1}
         SS = P - K * S * K' = Sigma_cond
     **Note! This can be done more efficiently when R is diagonal.**

    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         h (Callable): emission function.
         H (Callable): Jacobian of emission function.
         R (D_obs,D_obs): emission covariance matrix.
         u (D_in,): inputs.
         y (D_obs,): observation.
         num_iter (int): number of re-linearizations around posterior for update step.

     Returns:
         mu_cond (D_hid,): filtered mean.
         Sigma_cond (D_hid,D_hid): filtered covariance.
    """
    def _step(carry, _):
        prior_mean, prior_cov = carry
        H_x = H(prior_mean, u)
        S = R + H_x @ prior_cov @ H_x.T
        K = psd_solve(S, H_x @ prior_cov).T
        posterior_cov = prior_cov - K @ S @ K.T
        posterior_mean = prior_mean + K @ (y - h(prior_mean, u))
        return (posterior_mean, posterior_cov), None

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, symmetrize(Sigma_cond)

_zeros_if_none = lambda x, shape: x if x is not None else jnp.zeros(shape)

def extended_kalman_filter_augmented_state(
    params: ParamsNLGSSM,
    model_params,
    emissions: Float[Array, "ntime emission_dim"],
    conditions,
    output_fields: Optional[List[str]] = ["filtered_means", "filtered_covariances", "predicted_means",
                                              "predicted_covariances"],
    trial_masks = None,
) -> PosteriorGSSMFiltered:
    r"""Run an (iterated) extended Kalman filter to produce the
    marginal likelihood and filtered velocity estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 1).
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".

    Returns:
        post: posterior object.

    """
    num_trials, num_timesteps, emissions_dim = emissions.shape
    dim_x = model_params.initial.mean.shape[-1]
    dim_v = params.initial_mean.shape[-1]
    
    # Dynamics and emission functions and their Jacobians
    h = params.emission_function
    H = jacfwd(h)

    initial_velocity_mean = params.initial_mean
    initial_velocity_cov = params.initial_covariance

    initial_state_means = model_params.initial.mean
    initial_state_covs = model_params.initial.cov

    tau = params.dynamics_covariance

    initial_condition = conditions[0]

    def _step(carry, r):
        ll, _pred_mean, _pred_cov = carry

        # Get parameters
        A = model_params.dynamics.weights
        Q = model_params.dynamics.cov
        R = model_params.emissions.cov
        b = _zeros_if_none(model_params.dynamics.bias, (dim_x,))

        A_augmented = jscipy.linalg.block_diag(A, jnp.eye(dim_v))
        Q_augmented = jscipy.linalg.block_diag(Q, jnp.zeros((dim_v, dim_v)))
        b_augmented = jnp.concatenate([b, jnp.zeros((dim_v,))])

        y = emissions[r]
        next_condition = conditions[r+1]
        trial_mask = trial_masks[r]

        def _inner_step(inner_carry, t):
            ll, _, _, _pred_mean, _pred_cov = inner_carry

            # Get parameters and inputs for time index t
            y_t = y[t]

            # Get the Jacobian of the emission function
            H_u = H(_pred_mean)  # (N x (V+D))

            # Get the predicted emission
            y_pred = h(_pred_mean)  # N

            # Get the innovation covariance
            s_k = H_u @ _pred_cov @ H_u.T + R
            s_k = symmetrize(s_k)

            # Update the log likelihood
            ll += MVN(y_pred, s_k).log_prob(jnp.atleast_1d(y_t))

            # Get the Kalman gain
            K = psd_solve(s_k, H_u @ _pred_cov, diagonal_boost=1e-9).T

            # Get the filtered mean
            filtered_mean = _pred_mean + K @ (y_t - y_pred) 

            # Get the filtered covariance
            filtered_cov = _pred_cov - K @ s_k @ K.T
            filtered_cov = symmetrize(filtered_cov)

            # Get the predicted mean
            pred_mean = A_augmented @ filtered_mean + b_augmented

            # Get the predicted covariance  
            pred_cov = A_augmented @ filtered_cov @ A_augmented.T + Q_augmented
            pred_cov = symmetrize(pred_cov)

            return (ll, filtered_mean, filtered_cov, pred_mean, pred_cov), None

        def true_fun(inputs):
            (ll, filtered_mean, filtered_cov, pred_mean, pred_cov), _ = lax.scan(_inner_step, inputs, jnp.arange(num_timesteps))
            return ll, filtered_mean, filtered_cov

        def false_fun(inputs):
            ll, _, _, _pred_mean, _pred_cov = inputs
            return ll, _pred_mean, _pred_cov

        inputs = (ll, jnp.zeros_like(_pred_mean), jnp.zeros_like(_pred_cov), _pred_mean, _pred_cov)
        ll, filtered_mean, filtered_cov = jax.lax.cond(trial_mask, true_fun, false_fun, inputs)

        A_augmented_across_trial = jscipy.linalg.block_diag(jnp.zeros_like(A), jnp.eye(dim_v))
        Q_augmented_across_trial = jscipy.linalg.block_diag(initial_state_covs[next_condition], tau)
        b_augmented_across_trial = jnp.concatenate([initial_state_means[next_condition], jnp.zeros((dim_v,))])

        # Predict the next state
        pred_mean = A_augmented_across_trial @ filtered_mean + b_augmented_across_trial
        pred_cov = A_augmented_across_trial @ filtered_cov @ A_augmented_across_trial.T + Q_augmented_across_trial
        pred_cov = symmetrize(pred_cov)

        # Build carry and output states
        carry = (ll, pred_mean, pred_cov)
        outputs = {
            "filtered_means": filtered_mean[dim_x:],
            "filtered_covariances": filtered_cov[dim_x:, dim_x:],
        }

        return carry, outputs

    # Run the extended Kalman filter
    carry = (0.0, 
             jnp.concatenate([initial_state_means[initial_condition], initial_velocity_mean]), 
             jscipy.linalg.block_diag(initial_state_covs[initial_condition], initial_velocity_cov))
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_trials))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered

    

def extended_kalman_filter_x_marginalized(
        params: ParamsNLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        conditions,
        output_fields: Optional[List[str]] = ["filtered_means", "filtered_covariances", "predicted_means",
                                              "predicted_covariances"],
        trial_masks = None,
) -> PosteriorGSSMFiltered:
    r"""Run an (iterated) extended Kalman filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 1).
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".

    Returns:
        post: posterior object.

    """
    num_trials, num_timesteps, emissions_dim = emissions.shape

    # Dynamics and emission functions and their Jacobians
    h, h_s = params.emission_function
    H = jacfwd(h, argnums=0, has_aux=True)

    def _step(carry, t):
        ll, _pred_mean, _pred_cov = carry

        # Get parameters and inputs for time index t
        Q = params.dynamics_covariance
        y = emissions[t]
        y_flattened = y.flatten()
        condition = conditions[t]
        trial_mask = trial_masks[t]

        # Update the log likelihood
        H_x, pred_obs_covs, _ = H(_pred_mean, y, condition)  # (TN x V), (TN x T x N)
        # H_eps = jscipy.linalg.block_diag(*pred_obs_covs)

        y_pred, _, _ = h(_pred_mean, y, condition)  # TN

        # block diagonal
        s_k = H_x @ _pred_cov @ H_x.T + jscipy.linalg.block_diag(*pred_obs_covs)
        s_k = symmetrize(s_k)

        ll += trial_mask * MVN(y_pred, s_k).log_prob(jnp.atleast_1d(y_flattened))

        K = psd_solve(s_k, H_x @ _pred_cov, diagonal_boost=1e-9).T
        filtered_cov = _pred_cov - trial_mask * (K @ s_k @ K.T)
        filtered_mean = _pred_mean + trial_mask * (K @ (y_flattened - y_pred))
        filtered_cov = symmetrize(filtered_cov)

        # def true_fun(inputs):
        #     ll, _pred_mean, _pred_cov = inputs
        #     y_pred, pred_obs_covs, pred_x_means = h(_pred_mean, y, condition)  # TN
            
        #     def compute_jacobian(V, mu_pred):
        #         def f(V_flat):
        #             return jnp.ravel(h_s(V_flat).reshape(emissions_dim, mu_pred.shape[0]) @ mu_pred)
        #         return jax.jacobian(f)(V)
                
        #     def step_fn(carry, inputs):
        #         ll_t, x, P = carry
        #         y_t, z_pred, R_t, mu_t  = inputs  # y_t: (N,), z_pred: (N,), R_t: (N,N)
        #         H_t = compute_jacobian(_pred_mean, mu_t)    # (N, dim_V)
        #         S_t = H_t @ P @ H_t.T + R_t                # innovation covariance # (N,N)
        #         S_t_inv = inv_via_cholesky(S_t)
        #         K_t = P @ H_t.T @ S_t_inv                  # Kalman gain, shape (dim_V, N)
        #         innov = y_t - (z_pred + H_t @ (x - _pred_mean))      # innovation
        #         innov_cov = H_t @ P @ H_t.T + S_t
        #         ll_t +=  MVN(0.0, innov_cov).log_prob(jnp.atleast_1d(innov))
        #         x_new = x + K_t @ innov
        #         P_new = (jnp.eye(K_t.shape[0]) - K_t @ H_t) @ P
        #         return (ll_t, x_new, P_new), None
        #     (ll, filtered_mean, filtered_cov), _ = lax.scan(step_fn, 
        #                                                 (ll, _pred_mean, _pred_cov), 
        #                                                 (y, y_pred.reshape(num_timesteps, emissions_dim), pred_obs_covs, pred_x_means))
        #     return ll, filtered_mean, filtered_cov

        # def false_fun(inputs):
        #     ll, _pred_mean, _pred_cov = inputs
        #     return ll, _pred_mean, _pred_cov

        # ll, filtered_mean, filtered_cov = jax.lax.cond(trial_mask, true_fun, false_fun, (ll, _pred_mean, _pred_cov))

        # Predict the next state
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, Q)

        # Build carry and output states
        carry = (ll, pred_mean, pred_cov)
        outputs = {
            "filtered_means": filtered_mean,
            "filtered_covariances": filtered_cov,
            "predicted_means": _pred_mean,
            "predicted_covariances": _pred_cov,
            "marginal_loglik": ll,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}

        return carry, outputs

    # Run the extended Kalman filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_trials))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered

def extended_kalman_filter(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"],
    trial_masks = None,
    mode = 'hybrid',
) -> PosteriorGSSMFiltered:
    r"""Run an (iterated) extended Kalman filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 1).
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".

    Returns:
        post: posterior object.

    """
    num_trials = len(emissions)

    # Dynamics and emission functions and their Jacobians
    h = params.emission_function
    H = jacfwd(h)
    # HH = hessian(h)

    def _step(carry, t):
        ll, _pred_mean, _pred_cov = carry

        # Get parameters and inputs for time index t
        Q = params.dynamics_covariance
        R = params.emission_covariance[t]
        y = emissions[t]
        trial_mask = trial_masks[t]

        # Update the log likelihood
        H_x = H(_pred_mean)  # (ND x V)
        y_pred = h(_pred_mean)  # ND

        if mode == 'hybrid':
            def true_fun(inputs):
                _pred_mean, _pred_cov = inputs
                _pred_pre = inv_via_cholesky(_pred_cov)
                filtered_pre = _pred_pre + H_x.T @ jscipy.linalg.block_diag(*R) @ H_x
                filtered_cov = inv_via_cholesky(filtered_pre)
                filtered_mean = _pred_mean - filtered_cov @ H_x.T @ (jscipy.linalg.block_diag(*R) @ y_pred - y.flatten())
                return filtered_mean, filtered_cov

            def false_fun(inputs):
                _pred_mean, _pred_cov = inputs
                return _pred_mean, _pred_cov

            inputs = (_pred_mean, _pred_cov)
            filtered_mean, filtered_cov = jax.lax.cond(trial_mask, true_fun, false_fun, inputs)

            pred_mean = filtered_mean
            pred_cov = Q + filtered_cov

        elif mode == 'cov':
            s_k = H_x @ _pred_cov @ H_x.T + jscipy.linalg.block_diag(*R)
            s_k = symmetrize(s_k)

            # Condition on this emission
            K = psd_solve(s_k, H_x @ _pred_cov, diagonal_boost=1e-9).T
            filtered_cov = _pred_cov - trial_mask * (K @ s_k @ K.T)
            filtered_mean = _pred_mean + trial_mask * (K @ (y.flatten() - y_pred))
            filtered_cov = symmetrize(filtered_cov)

            # Predict the next state
            pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, Q)

        # Build carry and output states
        carry = (ll, pred_mean, pred_cov)
        outputs = {
            "filtered_means": filtered_mean,
            "filtered_covariances": filtered_cov,
            "predicted_means": _pred_mean,
            "predicted_covariances": _pred_cov,
            "marginal_loglik": ll,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}

        return carry, outputs

    # Run the extended Kalman filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_trials))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered


def iterated_extended_kalman_filter(
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    num_iter: int = 2,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMFiltered:
    r"""Run an iterated extended Kalman filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 2).
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """
    filtered_posterior = extended_kalman_filter(params, emissions, num_iter, inputs)
    return filtered_posterior


def extended_kalman_smoother(
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    filtered_posterior: Optional[PosteriorGSSMFiltered] = None,
    trial_masks = None,
    mode = 'hybrid',
) -> PosteriorGSSMSmoothed:
    r"""Run an extended Kalman (RTS) smoother.

    Args:
        params: model parameters.
        emissions: observation sequence.
        filtered_posterior: optional output from filtering step.
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """
    num_trials = len(emissions)

    # Get filtered posterior
    if filtered_posterior is None:
        filtered_posterior = extended_kalman_filter(params, emissions, trial_masks=trial_masks, mode=mode)
    ll = filtered_posterior.marginal_loglik
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next, smoothed_cov_sum, smoothed_cc_sum = carry
        t, filtered_mean, filtered_cov = args

        # Get parameters and inputs for time index t
        Q = params.dynamics_covariance

        # Prediction step
        m_pred = filtered_mean
        S_pred = filtered_cov + Q
        G = psd_solve(S_pred, filtered_cov, diagonal_boost=1e-9).T

        # Compute smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T
        smoothed_cov = symmetrize(smoothed_cov)
        smoothed_cov_sum += smoothed_cov

        # Compute the smoothed expectation of z_t z_{t+1}^T
        smoothed_cc_sum += G @ smoothed_cov_next + jnp.outer(smoothed_mean, smoothed_mean_next)

        return ((smoothed_mean, smoothed_cov, smoothed_cov_sum, smoothed_cc_sum),
                smoothed_mean)

    dof = filtered_covs.shape[-1]
    smoothed_cross_cov_sum_init = jnp.zeros((dof, dof))
    smoothed_cov_sum_init = jnp.zeros((dof, dof))
    # Run the extended Kalman smoother
    ((_, smoothed_cov_0, smoothed_cov_sum, smoothed_cross_cov_sum),
     smoothed_means) = lax.scan(
        _step,
        (filtered_means[-1], filtered_covs[-1],
         smoothed_cov_sum_init, smoothed_cross_cov_sum_init),
        (jnp.arange(num_trials - 1), filtered_means[:-1], filtered_covs[:-1]),
        reverse=True,
    )

    # Concatenate the arrays and return
    smoothed_means = jnp.vstack((smoothed_means, filtered_means[-1][None, ...]))
    smoothed_cross = smoothed_cross_cov_sum
    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances_0=smoothed_cov_0,
        smoothed_covariances_p=smoothed_cov_sum,
        smoothed_covariances_n=smoothed_cov_sum - smoothed_cov_0 + filtered_covs[-1],
        smoothed_cross_covariances=smoothed_cross,
    )

def extended_kalman_smoother_marginal_log_prob(
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    conditions=None,
    trial_masks = None,
    filtered_posterior: Optional[PosteriorGSSMFiltered] = None,
):
    r"""Run an extended Kalman (RTS) smoother.

    Args:
        params: model parameters.
        emissions: observation sequence.
        filtered_posterior: optional output from filtering step.
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """
    num_trials = len(emissions)
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    h = params.emission_function
    H = jacfwd(h, argnums=0, has_aux=True)

    def _step(carry, args):
        # Unpack the inputs
        ll, smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args
        y = emissions[t]
        y_flattened = y.flatten()
        condition = conditions[t]
        trial_mask = trial_masks[t]

        # Get parameters and inputs for time index t
        Q = params.dynamics_covariance

        # Prediction step
        m_pred = filtered_mean
        S_pred = filtered_cov + Q
        G = psd_solve(S_pred, filtered_cov, diagonal_boost=1e-9).T

        # Compute smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T
        smoothed_cov = symmetrize(smoothed_cov)

        H_x, pred_obs_covs = H(smoothed_mean, y, condition)  # (TN x V), (TN x T x N)

        y_pred, _ = h(smoothed_mean, y, condition)  # TN
        s_k = H_x @ smoothed_cov @ H_x.T + jscipy.linalg.block_diag(*pred_obs_covs)
        s_k = symmetrize(s_k)

        ll += trial_mask * MVN(y_pred, s_k).log_prob(jnp.atleast_1d(y_flattened))

        return (ll, smoothed_mean, smoothed_cov), None

    # Run the extended Kalman smoother
    (marginal_loglik, _, _), _ = lax.scan(
        _step,
        (0.0, filtered_means[-1], filtered_covs[-1]),
        (jnp.arange(num_trials - 1), filtered_means[:-1], filtered_covs[:-1]),
        reverse=True,
    )

    return marginal_loglik


def extended_kalman_posterior_sample(
    key: PRNGKey,
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> Float[Array, "ntime state_dim"]:
    r"""Run forward-filtering, backward-sampling to draw samples.

    Args:
        key: random number key.
        params: model parameters.
        emissions: observation sequence.
        inputs: optional array of inputs.

    Returns:
        Float[Array, "ntime state_dim"]: one sample of $z_{1:T}$ from the posterior distribution on latent states.
    """
    num_timesteps = len(emissions)

    # Get filtered posterior
    filtered_posterior = extended_kalman_filter(params, emissions, inputs=inputs)
    ll = filtered_posterior.marginal_loglik
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    # Dynamics and emission functions and their Jacobians
    f = params.dynamics_function
    F = jacfwd(f)
    f, F = (_process_fn(fn, inputs) for fn in (f, F))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        # Unpack the inputs
        next_state = carry
        key, filtered_mean, filtered_cov, t = args

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        u = inputs[t]

        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(filtered_mean, filtered_cov, f, F, Q, u, next_state, 1)
        state = MVN(smoothed_mean, smoothed_cov).sample(seed=key)
        return state, state

    # Initialize the last state
    key, this_key = jr.split(key, 2)
    last_state = MVN(filtered_means[-1], filtered_covs[-1]).sample(seed=this_key)

    _, states = lax.scan(
        _step,
        last_state,
        (
            jr.split(key, num_timesteps - 1),
            filtered_means[:-1],
            filtered_covs[:-1],
            jnp.arange(num_timesteps - 1),
        ),
        reverse=True,
    )
    return jnp.vstack([states, last_state])


def iterated_extended_kalman_smoother(
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    num_iter: int = 2,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMSmoothed:
    r"""Run an iterated extended Kalman smoother (IEKS).

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 2).
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """

    def _step(carry, _):
        # Relinearize around smoothed posterior from previous iteration
        smoothed_prior = carry
        smoothed_posterior = extended_kalman_smoother(params, emissions, smoothed_prior, inputs)
        return smoothed_posterior, None

    smoothed_posterior, _ = lax.scan(_step, None, jnp.arange(num_iter))
    return smoothed_posterior
