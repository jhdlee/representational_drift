import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import vmap
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import NamedTuple, Optional, List

from dynamax.utils.utils import psd_solve
from dynamax.nonlinear_gaussian_ssm.models import  ParamsNLGSSM
from dynamax.linear_gaussian_ssm.models import PosteriorGSSMFiltered, PosteriorGSSMSmoothed
from dynamax.types import PRNGKey

class UKFHyperParams(NamedTuple):
    """Lightweight container for UKF hyperparameters.

    Default values taken from https://github.com/sbitzer/UKF-exposed
    """
    alpha: float = jnp.sqrt(3)
    beta: int = 2
    kappa: int = 1


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,)) if x is None else x
_compute_lambda = lambda x, y, z: x**2 * (y + z) - z


def _compute_sigmas(m, P, n_total, n, lamb):
    """Compute (2n+1) sigma points used for inputs to  unscented transform.

    Args:
        m (D_hid,): mean.
        P (D_hid,D_hid): covariance.
        n (int): number of state dimensions.
        lamb (Scalar): unscented parameter lambda.

    Returns:
        sigmas (2*D_hid+1,): 2n+1 sigma points.
    """
    distances = jnp.sqrt(n_total + lamb) * jnp.linalg.cholesky(P)
    sigma_plus = jnp.array([m + distances[:, i] for i in range(n)])
    sigma_minus = jnp.array([m - distances[:, i] for i in range(n)])
    return jnp.concatenate((jnp.array([m]), sigma_plus, sigma_minus))


def _compute_weights(n, alpha, beta, lamb):
    """Compute weights used to compute predicted mean and covariance (Sarkka 5.77).

    Args:
        n (int): number of state dimensions.
        alpha (float): hyperparameter that determines the spread of sigma points
        beta (float): hyperparameter that incorporates prior information
        lamb (float): lamb = alpha**2 *(n + kappa) - n

    Returns:
        w_mean (2*n+1,): 2n+1 weights to compute predicted mean.
        w_cov (2*n+1,): 2n+1 weights to compute predicted covariance.
    """
    factor = 1 / (2 * (n + lamb))
    w_mean = jnp.concatenate((jnp.array([lamb / (n + lamb)]), jnp.ones(2 * n) * factor))
    w_cov = jnp.concatenate((jnp.array([lamb / (n + lamb) + (1 - alpha**2 + beta)]), jnp.ones(2 * n) * factor))
    return w_mean, w_cov


def _predict(m, P, f, Q, lamb, w_mean, w_cov, u, n, n_noise):
    """Predict next mean and covariance using additive UKF

    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        f (Callable): dynamics function.
        Q (D_hid,D_hid): dynamics covariance matrix.
        lamb (float): lamb = alpha**2 *(n + kappa) - n.
        w_mean (2*D_hid+1,): 2n+1 weights to compute predicted mean.
        w_cov (2*D_hid+1,): 2n+1 weights to compute predicted covariance.
        u (D_in,): inputs.

    Returns:
        m_pred (D_hid,): predicted mean.
        P_pred (D_hid,D_hid): predicted covariance.

    """
    # Form sigma points and propagate
    sigmas_pred = _compute_sigmas(m, P, n + n_noise, n, lamb)
    sigmas_noise_pred = _compute_sigmas(jnp.zeros(n_noise),
                                        jnp.eye(n_noise),
                                        n + n_noise, n_noise, lamb)

    sigmas_pred = jnp.concatenate([sigmas_pred,
                                   jnp.tile(m[None], (2*n_noise, 1))], axis=0)
    sigmas_noise_pred = jnp.concatenate([jnp.tile(jnp.zeros((1, n_noise)), (2*n, 1)),
                                         sigmas_noise_pred], axis=0)

    sigmas_pred_prop = vmap(f, (0, 0), 0)(sigmas_pred, sigmas_noise_pred)

    # Compute predicted mean and covariance
    m_pred = jnp.tensordot(w_mean, sigmas_pred_prop, axes=1)
    P_pred = jnp.tensordot(w_cov, _outer(sigmas_pred_prop - m_pred, sigmas_pred_prop - m_pred), axes=1)
    P_cross = jnp.tensordot(w_cov, _outer(sigmas_pred - m, sigmas_pred_prop - m_pred), axes=1)
    return m_pred, P_pred, P_cross


def _condition_on(m, P, h, R, lamb, w_mean, w_cov, u, y, t, n, n_noise):
    """Condition a Gaussian potential on a new observation

    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        h (Callable): emission function.
        R (D_obs,D_obs): emssion covariance matrix
        lamb (float): lamb = alpha**2 *(n + kappa) - n.
        w_mean (2*D_hid+1,): 2n+1 weights to compute predicted mean.
        w_cov (2*D_hid+1,): 2n+1 weights to compute predicted covariance.
        u (D_in,): inputs.
        y (D_obs,): observation.black

    Returns:
        ll (float): log-likelihood of observation
        m_cond (D_hid,): filtered mean.
        P_cond (D_hid,D_hid): filtered covariance.

    """
    # Form sigma points and propagate
    sigmas_cond = _compute_sigmas(m, P, n + n_noise, n, lamb)
    sigmas_noise_cond = _compute_sigmas(jnp.zeros(n_noise),
                                        jnp.eye(n_noise),
                                        n + n_noise, n_noise, lamb)

    sigmas_cond = jnp.concatenate([sigmas_cond,
                                   jnp.tile(m[None], (2*n_noise, 1))], axis=0)
    sigmas_noise_cond = jnp.concatenate([jnp.tile(jnp.zeros((1, n_noise)), (2*n, 1)),
                                         sigmas_noise_cond], axis=0)

    sigmas_cond_prop = vmap(h, (0, 0, None), 0)(sigmas_cond, sigmas_noise_cond, t)

    # Compute parameters needed to filter
    pred_mean = jnp.tensordot(w_mean, sigmas_cond_prop, axes=1)
    pred_cov = jnp.tensordot(w_cov, _outer(sigmas_cond_prop - pred_mean, sigmas_cond_prop - pred_mean), axes=1)
    pred_cross = jnp.tensordot(w_cov, _outer(sigmas_cond - m, sigmas_cond_prop - pred_mean), axes=1)

    # Compute log-likelihood of observation
    ll = MVN(pred_mean, pred_cov).log_prob(y.flatten())

    # Compute filtered mean and covariace
    K = psd_solve(pred_cov, pred_cross.T).T  # Filter gain
    m_cond = m + K @ (y.flatten() - pred_mean)
    P_cond = P - K @ pred_cov @ K.T
    return ll, m_cond, P_cond


def unscented_kalman_filter(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    hyperparams: UKFHyperParams,
    inputs: Optional[Float[Array, "ntime input_dim"]]=None,
    output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"],
) -> PosteriorGSSMFiltered:
    """Run a unscented Kalman filter to produce the marginal likelihood and
    filtered state estimates.

    Args:
        params: model parameters.
        emissions: array of observations.
        hyperparams: hyper-parameters.
        inputs: optional array of inputs.

    Returns:
        filtered_posterior: posterior object.

    """
    num_trials, num_timesteps, emissions_dim = emissions.shape
    state_dim = params.dynamics_covariance.shape[0]
    cov_dim = num_timesteps * emissions_dim
    n_p = state_dim + state_dim
    n_pp = state_dim + cov_dim

    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb_p = _compute_lambda(alpha, kappa, n_p)
    w_mean_p, w_cov_p = _compute_weights(n_p, alpha, beta, lamb_p)

    lamb_pp = _compute_lambda(alpha, kappa, n_pp)
    w_mean_pp, w_cov_pp = _compute_weights(n_pp, alpha, beta, lamb_pp)

    # Dynamics and emission functions
    f, h = params.dynamics_function, params.emission_function
    inputs = _process_input(inputs, num_trials)

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Get parameters and inputs for time t
        Q = None #_get_params(params.dynamics_covariance, 2, t)
        R = None #_get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on this emission
        log_likelihood, filtered_mean, filtered_cov = _condition_on(
            pred_mean, pred_cov, h, R, lamb_pp, w_mean_pp, w_cov_pp, u, y, t,
            state_dim, cov_dim
        )

        # Update the log likelihood
        ll += log_likelihood

        # Predict the next state
        pred_mean, pred_cov, _ = _predict(filtered_mean, filtered_cov, f, Q,
                                          lamb_p, w_mean_p, w_cov_p, u, state_dim, state_dim)

        # Build carry and output states
        carry = (ll, pred_mean, pred_cov)
        outputs = {
            "filtered_means": filtered_mean,
            "filtered_covariances": filtered_cov,
            "predicted_means": pred_mean,
            "predicted_covariances": pred_cov,
            "marginal_loglik": ll,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}
        return carry, outputs


    # Run the Unscented Kalman Filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_trials))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered


def unscented_kalman_smoother(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    hyperparams: UKFHyperParams,
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> PosteriorGSSMSmoothed:
    """Run a unscented Kalman (RTS) smoother.

    Args:
        params: model parameters.
        emissions: array of observations.
        hyperperams: hyper-parameters.
        inputs: optional inputs.

    Returns:
        nlgssm_posterior: posterior object.

    """
    num_timesteps = len(emissions)
    state_dim = params.dynamics_covariance.shape[0]

    # Run the unscented Kalman filter
    ukf_posterior = unscented_kalman_filter(params, emissions, hyperparams, inputs)
    ll = ukf_posterior.marginal_loglik
    filtered_means = ukf_posterior.filtered_means
    filtered_covs = ukf_posterior.filtered_covariances

    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, state_dim)
    w_mean, w_cov = _compute_weights(state_dim, alpha, beta, lamb)

    # Dynamics and emission functions
    f, h = params.dynamics_function, params.emission_function
    f, h = (_process_fn(fn, inputs) for fn in (f, h))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Get parameters and inputs for time t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Prediction step
        m_pred, S_pred, S_cross = _predict(filtered_mean, filtered_cov, f, Q, lamb, w_mean, w_cov, u)
        G = psd_solve(S_pred, S_cross.T).T

        # Compute smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    # Run the unscented Kalman smoother
    _, (smoothed_means, smoothed_covs) = lax.scan(
        _step,
        (filtered_means[-1], filtered_covs[-1]),
        (jnp.arange(num_timesteps - 1), filtered_means[:-1], filtered_covs[:-1]),
        reverse=True,
    )

    # Concatenate the arrays and return
    smoothed_means = jnp.vstack((smoothed_means, filtered_means[-1][None, ...]))
    smoothed_covs = jnp.vstack((smoothed_covs, filtered_covs[-1][None, ...]))

    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
    )

def unscented_kalman_posterior_sample(
    key: PRNGKey,
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    hyperparams: UKFHyperParams,
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> Float[Array, "ntime state_dim"]:
    """Run a unscented Kalman (RTS) posterior sampler.

    Args:
        params: model parameters.
        emissions: array of observations.
        hyperperams: hyper-parameters.
        inputs: optional inputs.

    Returns:
        nlgssm_posterior: posterior object.

    """
    num_trials, num_timesteps, emissions_dim = emissions.shape
    state_dim = params.dynamics_covariance.shape[0]
    n_p = state_dim + state_dim

    # Run the unscented Kalman filter
    ukf_posterior = unscented_kalman_filter(params, emissions, hyperparams, inputs)
    ll = ukf_posterior.marginal_loglik
    filtered_means = ukf_posterior.filtered_means
    filtered_covs = ukf_posterior.filtered_covariances

    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb_p = _compute_lambda(alpha, kappa, n_p)
    w_mean_p, w_cov_p = _compute_weights(n_p, alpha, beta, lamb_p)

    # Dynamics and emission functions
    f, h = params.dynamics_function, params.emission_function
    # f, h = (_process_fn(fn, inputs) for fn in (f, h))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        key, filtered_mean, filtered_cov, t = args

        # Get parameters and inputs for time t
        Q = None #_get_params(params.dynamics_covariance, 2, t)
        R = None #_get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Prediction step
        m_pred, S_pred, S_cross = _predict(filtered_mean, filtered_cov, f, Q,
                                           lamb_p, w_mean_p, w_cov_p, u, state_dim, state_dim)
        G = psd_solve(S_pred, S_cross.T).T

        # Compute smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T

        state = MVN(smoothed_mean, smoothed_cov).sample(seed=key)
        return (smoothed_mean, smoothed_cov), state

    # Run the unscented Kalman smoother
    key, this_key = jr.split(key, 2)
    last_state = MVN(filtered_means[-1], filtered_covs[-1]).sample(seed=this_key)

    args = (
        jr.split(key, num_trials - 1),
        filtered_means[:-1][::-1],
        filtered_covs[:-1][::-1],
        jnp.arange(num_trials - 2, -1, -1),
    )
    _, reversed_states = lax.scan(_step, (filtered_means[-1], filtered_covs[-1]), args)
    states = jnp.vstack([reversed_states[::-1], last_state])
    return states