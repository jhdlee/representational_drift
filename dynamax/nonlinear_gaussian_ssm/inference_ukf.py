import jax.numpy as jnp
import jax.scipy as jscipy
from jax import lax
from jax import vmap
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import List, Optional, NamedTuple, Optional, Union, Callable

from dynamax.utils.utils import psd_solve, symmetrize
from dynamax.linear_gaussian_ssm.models import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

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


def _compute_sigmas(m, P, n, lamb):
    """Compute (2n+1) sigma points used for inputs to  unscented transform.

    Args:
        m (D_hid,): mean.
        P (D_hid,D_hid): covariance.
        n (int): number of state dimensions.
        lamb (Scalar): unscented parameter lambda.

    Returns:
        sigmas (2*D_hid+1,): 2n+1 sigma points.
    """
    distances = jnp.sqrt(n + lamb) * jnp.linalg.cholesky(P)
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


def _predict(m, P, Q):
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
    return m, P + Q

def _condition_on_x_marginalized(m, P, h, lamb, w_mean, w_cov, y, condition, n, n_r, trial_mask):
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

    n_prime = n + n_r
    m_tilde = jnp.concatenate([m, jnp.zeros(n_r)])
    P_tilde = jscipy.linalg.block_diag(P, jnp.eye(n_r))

    sigmas_cond = _compute_sigmas(m_tilde, P_tilde, n_prime, lamb)
    sigmas_cond_m, sigmas_cond_r = jnp.hsplit(sigmas_cond, [n])
    sigmas_cond_prop = vmap(h, (0, None, None, 0), 0)(sigmas_cond_m, y, condition, sigmas_cond_r)

    # Compute parameters needed to filter
    pred_mean = jnp.tensordot(w_mean, sigmas_cond_prop, axes=1)
    pred_cov = jnp.tensordot(w_cov, _outer(sigmas_cond_prop - pred_mean, sigmas_cond_prop - pred_mean), axes=1)
    pred_cross = jnp.tensordot(w_cov, _outer(sigmas_cond_m - m, sigmas_cond_prop - pred_mean), axes=1)

    # Compute log-likelihood of observation
    ll = MVN(pred_mean, pred_cov).log_prob(y.flatten())

    # Compute filtered mean and covariace
    K = psd_solve(pred_cov, pred_cross.T).T  # Filter gain
    m_cond = m + trial_mask * K @ (y.flatten() - pred_mean)
    P_cond = P - trial_mask * K @ pred_cov @ K.T
    P_cond = symmetrize(P_cond)
    return ll, m_cond, P_cond

def unscented_kalman_filter_x_marginalized(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    hyperparams: UKFHyperParams,
    conditions,
    output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"],
    trial_masks = None,
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
    n = state_dim = params.dynamics_covariance.shape[0]
    n_r = cov_dim = num_timesteps * emissions_dim
    n_prime = state_dim + cov_dim

    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, n_prime)
    w_mean, w_cov = _compute_weights(n_prime, alpha, beta, lamb)

    # Dynamics and emission functions
    h = params.emission_function

    def _step(carry, t):
        ll, _pred_mean, _pred_cov = carry

        # Get parameters and inputs for time t
        Q = params.dynamics_covariance
        y = emissions[t]
        condition = conditions[t]
        trial_mask = trial_masks[t]

        # Condition on this emission
        log_likelihood, filtered_mean, filtered_cov = _condition_on_x_marginalized(
            _pred_mean, _pred_cov, h, lamb, w_mean, w_cov, y, condition, n, n_r, trial_mask
        )

        # Update the log likelihood
        ll += trial_mask * log_likelihood

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


    # Run the Unscented Kalman Filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_trials))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered

def _condition_on(m, P, h, R, lamb, w_mean, w_cov, y, trial_mask):
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
    n = len(m)
    # Form sigma points and propagate
    sigmas_cond = _compute_sigmas(m, P, n, lamb)
    sigmas_cond_prop = vmap(h, 0, 0)(sigmas_cond)

    # Compute parameters needed to filter
    pred_mean = jnp.tensordot(w_mean, sigmas_cond_prop, axes=1)
    pred_cov = jnp.tensordot(w_cov, _outer(sigmas_cond_prop - pred_mean, sigmas_cond_prop - pred_mean), axes=1) + jscipy.linalg.block_diag(*R)
    pred_cross = jnp.tensordot(w_cov, _outer(sigmas_cond - m, sigmas_cond_prop - pred_mean), axes=1)

    # Compute log-likelihood of observation
    ll = MVN(pred_mean, pred_cov).log_prob(y.flatten())

    # Compute filtered mean and covariace
    K = psd_solve(pred_cov, pred_cross.T).T  # Filter gain
    m_cond = m + trial_mask * K @ (y.flatten() - pred_mean)
    P_cond = P - trial_mask * K @ pred_cov @ K.T
    P_cond = symmetrize(P_cond)
    return ll, m_cond, P_cond


def unscented_kalman_filter(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    hyperparams: UKFHyperParams,
    output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"],
    trial_masks = None,
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
    num_trials = len(emissions)
    state_dim = params.dynamics_covariance.shape[0]

    # Compute lambda and weights from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, state_dim)
    w_mean, w_cov = _compute_weights(state_dim, alpha, beta, lamb)

    # Dynamics and emission functions
    h = params.emission_function

    def _step(carry, t):
        ll, _pred_mean, _pred_cov = carry

        # Get parameters and inputs for time t
        Q = params.dynamics_covariance
        R = params.emission_covariance[t]
        y = emissions[t]
        trial_mask = trial_masks[t]

        # Condition on this emission
        log_likelihood, filtered_mean, filtered_cov = _condition_on(
            _pred_mean, _pred_cov, h, R, lamb, w_mean, w_cov, y, trial_mask
        )

        # Update the log likelihood
        ll += trial_mask * log_likelihood

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
    filtered_posterior: Optional[PosteriorGSSMFiltered] = None,
    trial_masks = None,
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
    num_trials = len(emissions)
    state_dim = params.dynamics_covariance.shape[0]

    # Run the unscented Kalman filter
    if filtered_posterior is None:
        filtered_posterior = unscented_kalman_filter(params, emissions, hyperparams, trial_masks=trial_masks)
    ll = filtered_posterior.marginal_loglik
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, state_dim)
    w_mean, w_cov = _compute_weights(state_dim, alpha, beta, lamb)

    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next, smoothed_cov_sum, smoothed_cc_sum = carry
        t, filtered_mean, filtered_cov = args

        # Get parameters and inputs for time t
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
    # Run the unscented Kalman smoother
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
