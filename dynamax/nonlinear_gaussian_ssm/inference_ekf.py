import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jscipy
from jax import lax, vmap
from jax import jacfwd, jacrev
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import List, Optional, NamedTuple, Optional, Union, Callable

from dynamax.utils.utils import psd_solve, symmetrize, inv_via_cholesky, rotate_subspace, power_iteration
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

# --- SMC and auxiliary functions ---
def ess_criterion(log_weights, unused_t: int):
    """A criterion that resamples based on effective sample size."""
    del unused_t
    num_particles = log_weights.shape[0]
    ess_num = 2 * jscipy.special.logsumexp(log_weights)
    ess_denom = jscipy.special.logsumexp(2 * log_weights)
    log_ess = ess_num - ess_denom
    return log_ess <= jnp.log(num_particles / 2.0)


def never_resample_criterion(log_weights, t: int):
    """A criterion that never resamples."""
    del log_weights
    del t
    return jnp.array(False)


def always_resample_criterion(log_weights, t: int):
    """A criterion that always resamples."""
    del log_weights
    del t
    return jnp.array(True)


def multinomial_resampling(
    key, log_weights, states):
    """Resample states with multinomial resampling.
    
    Args:
    key: A JAX PRNG key.
    log_weights: A [num_particles] ndarray, the log weights for each particle.
    states: A pytree of [num_particles, ...] ndarrays that
      will be resampled.
    Returns:
    resampled_states: A pytree of [num_particles, ...] ndarrays resampled via
      multinomial sampling.
    parents: A [num_particles] array containing index of parent of each state
    """
    num_particles = log_weights.shape[0]
    cat = tfd.Categorical(logits=log_weights)
    parents = cat.sample(sample_shape=(num_particles,), seed=key)
    
    # Check if JAX's default dtype is float64, and set parents accordingly
    default_dtype = jax.dtypes.canonicalize_dtype(jnp.float_)
    parents = parents.astype(jnp.int64 if default_dtype == jnp.float64 else jnp.int32)
    
    assert isinstance(parents, jnp.ndarray)
    return (jax.tree_util.tree_map(lambda item: item[parents], states), parents)


def stratified_resampling(
        key, log_weights, states):
    """Resample states with stratified resampling.
    Args:
    key: A JAX PRNG key.
    log_weights: A [num_particles] ndarray, the log weights for each particle.
    states: A pytree of [num_particles, ...] ndarrays that
      will be resampled.
    Returns:
    resampled_states: A pytree of [num_particles, ...] ndarrays resampled via
      multinomial sampling.
    parents: A [num_particles] array containing index of parent of each state
    """
    num_particles = log_weights.shape[0]
    us = jax.random.uniform(key, shape=[num_particles])
    us = (jnp.arange(num_particles) + us) / num_particles
    norm_log_weights = log_weights - jax.nn.logsumexp(log_weights)
    bins = jnp.cumsum(jnp.exp(norm_log_weights))
    inds = jnp.digitize(us, bins)
    return (jax.tree_util.tree_map(lambda x: x[inds], states), inds)

def smc_ekf_proposal_augmented_state(
    key,
    params: ParamsNLGSSM,
    model_params,
    emissions: Float[Array, "ntime emission_dim"],
    conditions,
    output_fields: Optional[List[str]] = ["filtered_means", "filtered_covariances", "predicted_means",
                                              "predicted_covariances"],
    num_particles = 100,
    resampling_criterion = ess_criterion,
    resampling_fn = multinomial_resampling,
    trial_masks = None,
):

    r"""Run an (iterated) extended Kalman filter to produce the
    marginal likelihood and filtered velocity estimates.

    Args:
        params: model parameters.
        model_params: model parameters.
        emissions: observation sequence.
        conditions: condition indices.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".
        trial_masks: trial masks.
        num_iters: number of linearizations around posterior for update step (default 1).

    Returns:
        posterior: posterior object.

    """
    num_trials, num_timesteps, emissions_dim = emissions.shape
    T = num_timesteps
    dim_x = model_params.initial.mean.shape[-1]
    dim_v = params.initial_mean.shape[-1]
    num_steps = num_trials * num_timesteps
    
    # Dynamics and emission functions and their Jacobians
    h = params.emission_function
    H = jacfwd(h)

    initial_velocity_mean = params.initial_mean
    initial_velocity_cov = params.initial_covariance

    initial_state_means = model_params.initial.mean
    initial_state_covs = model_params.initial.cov

    tau = params.dynamics_covariance

    def transition_fn(key, state, y, params, across_trial, aux):
        """Transition function for the augmented state across trials."""
        x_prev, P_prev = state

        A_prime, Q_prime, b_prime, R = params

        # Predict the next state
        pred_mean = A_prime @ x_prev + b_prime
        pred_cov = A_prime @ P_prev @ A_prime.T + Q_prime
        pred_cov = symmetrize(pred_cov)

        # Get the Jacobian of the emission function
        H_u = H(pred_mean)  # (N x (V+D))

        # Get the predicted emission
        y_pred = h(pred_mean)  # N

        # Get the innovation covariance
        S = H_u @ pred_cov @ H_u.T + R
        S = symmetrize(S)

        # Get the Kalman gain
        K = psd_solve(S, H_u @ pred_cov).T

        # Get the filtered mean
        x_filtered = pred_mean + K @ (y - y_pred)

        # Get the filtered covariance
        P_filtered = pred_cov - K @ S @ K.T
        P_filtered = symmetrize(P_filtered)

        x_new = MVN(x_filtered, P_filtered).sample(seed=key)
        y_new = h(x_new)

        # jax.debug.print('x_filtered: {x_filtered}', x_filtered=x_filtered)
        # jax.debug.print('P_filtered: {P_filtered}', P_filtered=P_filtered)
        # jax.debug.print('pred_mean: {pred_mean}', pred_mean=pred_mean)
        # jax.debug.print('Q_prime: {Q_prime}', Q_prime=Q_prime)
        # jax.debug.print('x_new: {x_new}', x_new=x_new)

        def true_fun(inputs):
            dim_x, pred_mean, Q_prime, x_new, aux = inputs
            log_p = tfd.MultivariateNormalFullCovariance(loc=pred_mean, covariance_matrix=Q_prime).log_prob(x_new)
            return log_p
        
        def false_fun(inputs):
            dim_x, pred_mean, Q_prime, x_new, aux = inputs
            
            Q_inv = jscipy.linalg.block_diag(inv_via_cholesky(aux[0]), aux[1])
            quad_term = jnp.einsum('ij,i,j->', Q_inv, x_new - pred_mean, x_new - pred_mean)
            logdet = jnp.linalg.slogdet(aux[0])[1]
            log_p = -0.5 * (quad_term + logdet + dim_x * jnp.log(2 * jnp.pi))
            
            return log_p
            
        inputs = (dim_x, pred_mean, Q_prime, x_new, aux)
        log_p = jax.lax.cond(across_trial, true_fun, false_fun, inputs)
        log_q = tfd.MultivariateNormalFullCovariance(loc=x_filtered, covariance_matrix=P_filtered).log_prob(x_new)
        log_lik = tfd.MultivariateNormalFullCovariance(loc=y_new, covariance_matrix=R).log_prob(y)

        jax.debug.print('log_p: {log_p}', log_p=log_p)
        jax.debug.print('log_q: {log_q}', log_q=log_q)
        jax.debug.print('log_lik: {log_lik}', log_lik=log_lik)
        
        log_incr = log_lik + log_p - log_q

        return (x_new, P_filtered), log_incr

    def resample(args):
        key, log_weights, states = args
        states, inds = resampling_fn(key, log_weights, states)
        return states, inds, jnp.zeros_like(log_weights)

    def dont_resample(args):
        _, log_weights, states = args
        return states, jnp.arange(num_particles), log_weights

    def smc_step(carry, state_slice):
        key, states, log_ws = carry
        key, sk1, sk2 = jax.random.split(key, num=3)
        r, observation = state_slice

        # Get parameters
        A = model_params.dynamics.weights
        Q = model_params.dynamics.cov
        R = model_params.emissions.cov
        b = _zeros_if_none(model_params.dynamics.bias, (dim_x,))

        A_augmented = jscipy.linalg.block_diag(A, jnp.eye(dim_v))
        Q_augmented = jscipy.linalg.block_diag(Q, jnp.zeros((dim_v, dim_v)))
        b_augmented = jnp.concatenate([b, jnp.zeros(dim_v)])

        current_condition = conditions[r]

        A_augmented_across_trial = jscipy.linalg.block_diag(jnp.zeros_like(A), jnp.eye(dim_v))
        Q_augmented_across_trial = jscipy.linalg.block_diag(initial_state_covs[current_condition], tau)
        b_augmented_across_trial = jnp.concatenate([initial_state_means[current_condition], jnp.zeros((dim_v))])

        params = (A_augmented_across_trial, Q_augmented_across_trial, b_augmented_across_trial, R)
        # Propagate the particle states
        new_states, incr_log_ws = vmap(transition_fn, (0, 0, None, None, None, None))(
            jax.random.split(sk1, num=num_particles), states, observation[0], params, True, 
            (initial_state_covs[current_condition], tau))

        # Update the log weights.
        log_ws += incr_log_ws

        # Resample the particles if resampling_criterion returns True and we haven't
        # exceeded the supplied number of steps.
        should_resample = jnp.logical_and(resampling_criterion(log_ws, r*T), r*T < num_steps)

        resampled_states, parents, resampled_log_ws = jax.lax.cond(
            should_resample,
            resample,
            dont_resample,
            (sk2, log_ws, new_states)
        )

        def inner_step(carry, t):
            key, states, log_ws = carry
            key, sk1, sk2 = jax.random.split(key, num=3)
            y_t = observation[t]

            inner_params = (A_augmented, Q_augmented, b_augmented, R)
            # Propagate the particle states
            new_states, incr_log_ws = vmap(transition_fn, (0, 0, None, None, None, None))(
                jax.random.split(sk1, num=num_particles), states, y_t, inner_params, False, (Q, jnp.zeros((dim_v, dim_v))))

            # Update the log weights.
            log_ws += incr_log_ws

            # Resample the particles if resampling_criterion returns True and we haven't
            # exceeded the supplied number of steps.
            should_resample = jnp.logical_and(resampling_criterion(log_ws, r*T+t), r*T+t < num_steps)

            resampled_states, parents, resampled_log_ws = jax.lax.cond(
                should_resample,
                resample,
                dont_resample,
                (sk2, log_ws, new_states)
            )

            return (key, resampled_states, resampled_log_ws), (log_ws, should_resample)

        # Run the SMC loop
        (key, resampled_states, resampled_log_ws), (inner_log_ws, inner_should_resample) = jax.lax.scan(
            inner_step,
            (key, resampled_states, resampled_log_ws),
            jnp.arange(1, num_timesteps)
        )

        # Concatenate the log weights
        log_ws = jnp.concatenate([log_ws[jnp.newaxis], inner_log_ws])

        # Concatenate the should resample
        should_resample = jnp.concatenate([should_resample[jnp.newaxis], inner_should_resample])

        return ((key, resampled_states, resampled_log_ws),
                (log_ws, should_resample))

    initial_means = jnp.zeros((num_particles, dim_x + dim_v))
    initial_means = initial_means.at[:, dim_x:].set(initial_velocity_mean)
    initial_covs = jnp.zeros((num_particles, dim_x + dim_v, dim_x + dim_v))
    initial_covs = initial_covs.at[:, dim_x:, dim_x:].set(initial_velocity_cov - tau)

    initial_states = (initial_means, initial_covs)

    _, (log_weights, resampled) = jax.lax.scan(
        smc_step,
        (key, initial_states, jnp.zeros([num_particles])),
        (jnp.arange(num_trials), emissions))

    # Reshape log_weights and resampled to be [num_trials * num_timesteps, num_particles]
    log_weights = log_weights.reshape(num_trials * num_timesteps, num_particles)
    resampled = resampled.reshape(num_trials * num_timesteps)

    # Average along particle dimension
    log_p_hats = jscipy.special.logsumexp(log_weights, axis=1) - jnp.log(num_particles)
    jax.debug.print('log_p_hats: {log_p_hats}', log_p_hats=log_p_hats)
    jax.debug.print('resampled: {resampled}', resampled=resampled)
    # Sum in time dimension on resampling steps.
    # Note that this does not include any steps past num_steps because
    # the resampling criterion doesn't allow resampling past num_steps steps.
    log_Z_hat = jnp.sum(log_p_hats * resampled)
    # If we didn't resample on the last timestep, add in the missing log_p_hat
    log_Z_hat += jnp.where(resampled[num_steps - 1], 0., log_p_hats[num_steps - 1])

    outputs = {"marginal_loglik": log_Z_hat}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    
    return posterior_filtered


def extended_kalman_filter_augmented_state(
    params: ParamsNLGSSM,
    model_params,
    emissions: Float[Array, "ntime emission_dim"],
    conditions,
    output_fields: Optional[List[str]] = ["filtered_means", "filtered_covariances", "predicted_means",
                                              "predicted_covariances"],
    block_masks = None,
    num_iters = 1,
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
    num_blocks, num_trials_per_block, num_timesteps, emissions_dim = emissions.shape
    dim_x = model_params.initial.mean.shape[-1]
    dim_v = params.initial_mean.shape[-1]
    
    # Dynamics and emission functions and their Jacobians
    h = params.emission_function
    H = jacrev(h, argnums=0, has_aux=True)

    initial_velocity_mean = params.initial_mean
    initial_velocity_cov = params.initial_covariance

    initial_state_means = model_params.initial.mean
    initial_state_covs = model_params.initial.cov

    tau = params.dynamics_covariance

    initial_condition = conditions[0, 0]

    def _step(carry, block_id):
        ll, _pred_mean, _pred_cov = carry

        # Get parameters
        A = model_params.dynamics.weights
        Q = model_params.dynamics.cov
        R = model_params.emissions.cov
        b = _zeros_if_none(model_params.dynamics.bias, (dim_x,))

        y = emissions[block_id]
        next_block_condition = conditions[block_id+1, 0]
        block_mask = block_masks[block_id]

        def _inner_step(inner_carry, r):
            ll, _, _, _pred_mean, _pred_cov = inner_carry

            # Get parameters and inputs for time index t
            y_r = y[r]
            next_trial_condition = conditions[block_id, r+1]

            def _inner_inner_step(inner_inner_carry, t):
                ll, _, _, _pred_mean, _pred_cov = inner_inner_carry

                y_t = y_r[t]

                # Get the Jacobian of the emission function
                H_u, y_pred = H(_pred_mean)  # (N x (V+D)), N

                # Get the innovation covariance
                s_k = H_u @ _pred_cov @ H_u.T + R
                s_k = symmetrize(s_k)

                # Update the log likelihood
                ll += MVN(y_pred, s_k).log_prob(jnp.atleast_1d(y_t))
                # jax.debug.print('ll: {ll}', ll=ll)

                def update_step(carry, _):
                    prior_mean, prior_cov = carry
                    # Get the Jacobian of the emission function
                    H_u, y_pred = H(prior_mean)  # (N x (V+D)), N

                    # Get the innovation covariance
                    s_k = H_u @ prior_cov @ H_u.T + R
                    s_k = symmetrize(s_k)

                    # Get the Kalman gain
                    K = psd_solve(s_k, H_u @ prior_cov).T

                    # Get the filtered mean
                    filtered_mean = prior_mean + K @ (y_t - y_pred) 

                    # Get the filtered covariance
                    filtered_cov = prior_cov - K @ s_k @ K.T
                    filtered_cov = symmetrize(filtered_cov)

                    return (filtered_mean, filtered_cov), None
                
                (filtered_mean, filtered_cov), _ = jax.lax.scan(update_step, 
                                                           (_pred_mean, _pred_cov), 
                                                           jnp.arange(num_iters))

                pred_mean = filtered_mean.at[:dim_x].set(A @ filtered_mean[:dim_x] + b)
                pred_cov = filtered_cov.at[:dim_x].set(A @ filtered_cov[:dim_x])
                pred_cov = pred_cov.at[:, :dim_x].set(pred_cov[:, :dim_x] @ A.T)
                pred_cov = pred_cov.at[:dim_x, :dim_x].set(pred_cov[:dim_x, :dim_x] + Q)

                # normalize the eigenvalues of the predicted covariance by their maximum
                # L, U = jnp.linalg.eigh(filtered_cov)
                lambda_max, _ = power_iteration(pred_cov)
                threshold = 1e-3
                should_normalize = lambda_max > threshold
                # jax.debug.print('max_eigval: {max_eigval}', max_eigval=jnp.max(L))
                pred_cov = jnp.where(should_normalize, 
                                        threshold * pred_cov / lambda_max, 
                                        pred_cov)

                return (ll, filtered_mean, filtered_cov, pred_mean, pred_cov), None

            init_carry = (ll, jnp.zeros_like(_pred_mean), jnp.zeros_like(_pred_cov), _pred_mean, _pred_cov)
            # Scan over time steps
            (ll, filtered_mean, filtered_cov, pred_mean, pred_cov), _ = lax.scan(_inner_inner_step, 
                                                                                 init_carry, 
                                                                                 jnp.arange(num_timesteps))
            
            # Get the predicted mean and covariance across trials but within block
            pred_mean = filtered_mean.at[:dim_x].set(initial_state_means[next_trial_condition])
            pred_cov = filtered_cov.at[:dim_x].set(0.0)
            pred_cov = pred_cov.at[:,:dim_x].set(0.0)
            pred_cov = pred_cov.at[:dim_x, :dim_x].set(initial_state_covs[next_trial_condition])

            # normalize the eigenvalues of the predicted covariance by their maximum
            # L, U = jnp.linalg.eigh(filtered_cov)
            lambda_max, _ = power_iteration(pred_cov)
            threshold = 1e-3
            should_normalize = lambda_max > threshold
            # jax.debug.print('max_eigval: {max_eigval}', max_eigval=jnp.max(L))
            pred_cov = jnp.where(should_normalize, 
                                    threshold * pred_cov / lambda_max, 
                                    pred_cov)

            return (ll, filtered_mean, filtered_cov, pred_mean, pred_cov), None

        def true_fun(inputs):
            (ll, filtered_mean, filtered_cov, _, _), _ = lax.scan(_inner_step, inputs, jnp.arange(num_trials_per_block))
            return ll, filtered_mean, filtered_cov

        def false_fun(inputs):
            ll, _, _, _pred_mean, _pred_cov = inputs
            return ll, _pred_mean, _pred_cov

        inputs = (ll, jnp.zeros_like(_pred_mean), jnp.zeros_like(_pred_cov), _pred_mean, _pred_cov)
        ll, filtered_mean, filtered_cov = jax.lax.cond(block_mask, true_fun, false_fun, inputs)

        # Get the predicted mean and covariance across blocks
        pred_mean = filtered_mean.at[:dim_x].set(initial_state_means[next_block_condition])
        pred_cov = filtered_cov.at[:dim_x].set(0.0)
        pred_cov = pred_cov.at[:,:dim_x].set(0.0)
        pred_cov = pred_cov.at[:dim_x, :dim_x].set(initial_state_covs[next_block_condition])
        pred_cov = pred_cov.at[dim_x:, dim_x:].set(pred_cov[dim_x:, dim_x:] + tau)

        # normalize the eigenvalues of the predicted covariance by their maximum
        # L, U = jnp.linalg.eigh(filtered_cov)
        lambda_max, _ = power_iteration(pred_cov)
        threshold = 1e-3
        should_normalize = lambda_max > threshold
        # jax.debug.print('max_eigval: {max_eigval}', max_eigval=jnp.max(L))
        pred_cov = jnp.where(should_normalize, 
                                threshold * pred_cov / lambda_max, 
                                pred_cov)

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
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_blocks))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered


def smc_ekf_proposal_x_marginalized(
    key,
    params: ParamsNLGSSM,
    model_params,
    emissions: Float[Array, "ntime emission_dim"],
    conditions,
    output_fields: Optional[List[str]] = ["filtered_means", "filtered_covariances", "predicted_means",
                                              "predicted_covariances"],
    num_particles = 100,
    resampling_criterion = ess_criterion,
    resampling_fn = multinomial_resampling,
    block_masks = None,
):

    r"""Run an (iterated) extended Kalman filter to produce the
    marginal likelihood and filtered velocity estimates.

    Args:
        params: model parameters.
        model_params: model parameters.
        emissions: observation sequence.
        conditions: condition indices.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".
        trial_masks: trial masks.
        num_iters: number of linearizations around posterior for update step (default 1).

    Returns:
        posterior: posterior object.

    """
    num_blocks, num_trials_per_block, num_timesteps, emissions_dim = emissions.shape
    T, N = num_timesteps, emissions_dim
    V = params.initial_mean.shape[-1]
    
    # Dynamics and emission functions and their Jacobians
    h = params.emission_function
    H = jacfwd(h, argnums=0, has_aux=True)

    initial_velocity_mean = params.initial_mean
    initial_velocity_cov = params.initial_covariance

    tau = params.dynamics_covariance

    def efficient_filter(y_true, m, P, condition):
        """
        Efficiently computes the filtered mean and covariance:
        where s_k = H_x @ cov_matrix @ H_xᵀ + D,
        D = block_diag(S_blocks).

        It uses the Woodbury matrix identity:
        s_k⁻¹ = D⁻¹ - D⁻¹ H_x (cov_matrix⁻¹ + H_xᵀ D⁻¹ H_x)⁻¹ H_xᵀ D⁻¹,
        and the determinant lemma:
        log|s_k| = log|D| - log|cov_matrix| + log|cov_matrix⁻¹ + H_xᵀ D⁻¹ H_x|.
        """
        H_x, (y_pred, R) = vmap(H, (None, 0, 0))(m, y_true, condition)  # (T x N x V), (T x N x N)
        H_x = H_x.reshape(-1, N, V)
        R = R.reshape(-1, N, N)
        y_pred = y_pred.reshape(-1, N)

        residuals = y_true.reshape(-1, N) - y_pred
        R_inv = vmap(inv_via_cholesky)(R)
        P_inv = inv_via_cholesky(P)

        U = P_inv + jnp.einsum('tiv,tij,tju->vu', H_x, R_inv, H_x)
        U_inv = inv_via_cholesky(U)

        R_inv_H_x = jnp.einsum('tij,tjv->tiv', R_inv, H_x)
        L = jnp.einsum('tjv,tju,uk->vk', H_x, R_inv_H_x, P)
        K = jnp.einsum('tiv,vu->tiu', R_inv_H_x, P) - jnp.einsum('tiv,vu,uk->tik', R_inv_H_x, U_inv, L)
        filtered_mean = m + jnp.einsum('tiu,ti->u', K, residuals)
        filtered_cov = P - jnp.einsum('tiu,tiv->uv', K, H_x) @ P
        filtered_cov = symmetrize(filtered_cov)

        # jax.debug.print('m: {m}', m=m)
        # jax.debug.print('filtered_mean: {filtered_mean}', filtered_mean=filtered_mean)
        # jax.debug.print('filtered_cov: {filtered_cov}', filtered_cov=filtered_cov)

        return filtered_mean, filtered_cov
    
    def efficient_log_likelihood(y_true, m, condition):
        *_, (y_pred, R) = vmap(h, (None, 0, 0))(m, y_true, condition)  # T x N
        R = R.reshape(-1, N, N)
        y_pred = y_pred.reshape(-1, N)

        residuals = y_true.reshape(-1, N) - y_pred
        R_inv = vmap(inv_via_cholesky)(R)
        
        quad_term = jnp.einsum('ti,tij,tj->', residuals, R_inv, residuals)

        _, logdet_R = jnp.linalg.slogdet(R)
        logdet = jnp.sum(logdet_R)

        # jax.debug.print('quad_term: {quad_term}', quad_term=quad_term)
        # jax.debug.print('logdet: {logdet}', logdet=logdet)
        
        ll = -0.5 * (quad_term + logdet + num_trials_per_block * T * N * jnp.log(2 * jnp.pi))
        return ll
    
    def transition_fn(key, state, y, condition):
        """Transition function for the augmented state across trials."""
        x_prev, P_prev = state

        pred_mean, pred_cov = _predict(x_prev, P_prev, tau)

        x_filtered, P_filtered = efficient_filter(y, pred_mean, pred_cov, condition)

        x_new = MVN(x_filtered, P_filtered).sample(seed=key)
            
        log_p = tfd.MultivariateNormalFullCovariance(loc=pred_mean, covariance_matrix=tau).log_prob(x_new)
        log_q = tfd.MultivariateNormalFullCovariance(loc=x_filtered, covariance_matrix=P_filtered).log_prob(x_new)
        log_lik = efficient_log_likelihood(y, x_new, condition)

        # jax.debug.print('log_p: {log_p}', log_p=log_p)
        # jax.debug.print('log_q: {log_q}', log_q=log_q)
        # jax.debug.print('log_lik: {log_lik}', log_lik=log_lik)
        
        log_incr = log_lik + log_p - log_q

        return (x_new, P_filtered), log_incr

    def resample(args):
        key, log_weights, states = args
        states, inds = resampling_fn(key, log_weights, states)
        return states, inds, jnp.zeros_like(log_weights)

    def dont_resample(args):
        _, log_weights, states = args
        return states, jnp.arange(num_particles), log_weights

    def smc_step(carry, state_slice):
        key, states, log_ws = carry
        key, sk1, sk2 = jax.random.split(key, num=3)
        r, observation = state_slice

        current_condition = conditions[r]
        # Propagate the particle states
        new_states, incr_log_ws = vmap(transition_fn, (0, 0, None, None))(
            jax.random.split(sk1, num=num_particles), states, observation, current_condition)

        # Update the log weights.
        log_ws += incr_log_ws

        # Resample the particles if resampling_criterion returns True and we haven't
        # exceeded the supplied number of steps.
        should_resample = jnp.logical_and(resampling_criterion(log_ws, r), r < num_blocks)

        resampled_states, parents, resampled_log_ws = jax.lax.cond(
            should_resample,
            resample,
            dont_resample,
            (sk2, log_ws, new_states)
        )

        return ((key, resampled_states, resampled_log_ws),
                (log_ws, should_resample))

    initial_means = jnp.zeros((num_particles, V))
    initial_means = initial_means.at[:, :].set(initial_velocity_mean)
    initial_covs = jnp.zeros((num_particles, V, V))
    initial_covs = initial_covs.at[:, :, :].set(initial_velocity_cov-tau)

    initial_states = (initial_means, initial_covs)

    _, (log_weights, resampled) = jax.lax.scan(
        smc_step,
        (key, initial_states, jnp.zeros([num_particles])),
        (jnp.arange(num_blocks), emissions))

    # Average along particle dimension
    log_p_hats = jscipy.special.logsumexp(log_weights, axis=1) - jnp.log(num_particles)
    # Sum in time dimension on resampling steps.
    # Note that this does not include any steps past num_steps because
    # the resampling criterion doesn't allow resampling past num_steps steps.
    log_Z_hat = jnp.sum(log_p_hats * resampled)
    # If we didn't resample on the last timestep, add in the missing log_p_hat
    log_Z_hat += jnp.where(resampled[num_blocks - 1], 0., log_p_hats[num_blocks - 1])

    outputs = {"marginal_loglik": log_Z_hat}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    
    return posterior_filtered


def extended_kalman_filter_x_marginalized(
        params: ParamsNLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        conditions,
        model_params=None,
        output_fields: Optional[List[str]] = ["filtered_means", "filtered_covariances", "predicted_means",
                                              "predicted_covariances"],
        block_masks = None,
        num_iters = 1,
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
    num_blocks, num_trials_per_block, num_timesteps, emissions_dim = emissions.shape
    T, N = num_timesteps, emissions_dim
    V = params.initial_mean.shape[-1]

    # Dynamics and emission functions and their Jacobians
    h = params.emission_function
    H = jacfwd(h, argnums=0, has_aux=True)

    def compute_log_likelihood(ll, y_true, m, P, condition):
        """
        Efficiently computes the log likelihood and filtered mean and covariance:
        ll = -0.5*(yᵀ s_k⁻¹ y + log|s_k| + m log(2π))
        where s_k = H_x @ cov_matrix @ H_xᵀ + D,
        D = block_diag(S_blocks).

        It uses the Woodbury matrix identity:
        s_k⁻¹ = D⁻¹ - D⁻¹ H_x (cov_matrix⁻¹ + H_xᵀ D⁻¹ H_x)⁻¹ H_xᵀ D⁻¹,
        and the determinant lemma:
        log|s_k| = log|D| - log|cov_matrix| + log|cov_matrix⁻¹ + H_xᵀ D⁻¹ H_x|.
        """
        H_x, (y_pred, R) = vmap(H, (None, 0, 0))(m, y_true, condition)  # (B x T x N x V), (B x T x N x N)
        # y_pred, *_ = vmap(h, (None, 0, 0))(m, y_true, condition)  # B x T x N
        H_x = H_x.reshape(-1, N, V)
        R = R.reshape(-1, N, N)
        y_pred = y_pred.reshape(-1, N)

        residuals = y_true.reshape(-1, N) - y_pred
        R_inv = jnp.linalg.inv(R)
        P_inv = jnp.linalg.inv(P)

        U = P_inv + jnp.einsum('tiv,tij,tju->vu', H_x, R_inv, H_x)
        U_inv = jnp.linalg.inv(U)

        q = jnp.einsum('tiv,tij,tj->v', H_x, R_inv, residuals)
        quad_term = jnp.einsum('ti,tij,tj->', residuals, R_inv, residuals) - q @ U_inv @ q

        _, logdet_R = jnp.linalg.slogdet(R)
        _, logdet_P = jnp.linalg.slogdet(P)
        _, logdet_U = jnp.linalg.slogdet(U)
        logdet = jnp.sum(logdet_R) + logdet_P + logdet_U

        # jax.debug.print('quad_term: {quad_term}', quad_term=quad_term)
        # jax.debug.print('logdet: {logdet}', logdet=logdet)
        
        ll += -0.5 * (quad_term + logdet + num_trials_per_block * T * N * jnp.log(2 * jnp.pi))

        return ll
    
    def compute_filter(y_true, m, P, condition):
        def update_step(carry, _):
            prior_mean, prior_cov = carry

            H_x, (y_pred, R) = vmap(H, (None, 0, 0))(prior_mean, y_true, condition)  # (B x T x N x V), (B x T x N x N)
            # y_pred, *_ = vmap(h, (None, 0, 0))(prior_mean, y_true, condition)  # B x T x N
            H_x = H_x.reshape(-1, N, V)
            R = R.reshape(-1, N, N)
            y_pred = y_pred.reshape(-1, N)

            residuals = y_true.reshape(-1, N) - y_pred
            R_inv = jnp.linalg.inv(R)
            P_inv = jnp.linalg.inv(prior_cov)

            U = P_inv + jnp.einsum('tiv,tij,tju->vu', H_x, R_inv, H_x)
            U_inv = jnp.linalg.inv(U)

            R_inv_H_x = jnp.einsum('tij,tjv->tiv', R_inv, H_x)
            L = jnp.einsum('tjv,tju,uk->vk', H_x, R_inv_H_x, P)
            K = jnp.einsum('tiv,vu->tiu', R_inv_H_x, P) - jnp.einsum('tiv,vu,uk->tik', R_inv_H_x, U_inv, L)
            filtered_mean = prior_mean + jnp.einsum('tiu,ti->u', K, residuals)
            filtered_cov = prior_cov - jnp.einsum('tiu,tiv->uv', K, H_x) @ prior_cov
            filtered_cov = symmetrize(filtered_cov)

            return (filtered_mean, filtered_cov), None

        (filtered_mean, filtered_cov), _ = lax.scan(update_step, (m, P), jnp.arange(num_iters))

        return filtered_mean, filtered_cov

    def _step(carry, t):
        ll, _pred_mean, _pred_cov = carry

        # Get parameters and inputs for time index t
        Q = params.dynamics_covariance
        y = emissions[t]
        condition = conditions[t]
        block_mask = block_masks[t]

        def true_fun(inputs):
            ll, m, P = inputs
            ll = compute_log_likelihood(ll, y, m, P, condition)
            # jax.debug.print('t: {t}', t=t)
            # jax.debug.print('ll: {ll}', ll=ll)
            filtered_mean, filtered_cov = compute_filter(y, m, P, condition)
            # jax.debug.print('filtered_mean: {filtered_mean}', filtered_mean=filtered_mean)
            # jax.debug.print('filtered_cov: {filtered_cov}', filtered_cov=filtered_cov)
            return ll, filtered_mean, filtered_cov

        def false_fun(inputs):
            ll, m, P = inputs
            return ll, m, P

        inputs = (ll, _pred_mean, _pred_cov)
        ll, filtered_mean, filtered_cov = jax.lax.cond(block_mask, true_fun, false_fun, inputs)

        # Predict the next state
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, Q)

        # normalize the eigenvalues of the predicted covariance by their maximum
        # L, U = jnp.linalg.eigh(filtered_cov)
        lambda_max, _ = power_iteration(pred_cov)
        threshold = 1e-3
        should_normalize = lambda_max > threshold
        # jax.debug.print('max_eigval: {max_eigval}', max_eigval=jnp.max(L))
        pred_cov = jnp.where(should_normalize, 
                                threshold * pred_cov / lambda_max, 
                                pred_cov)

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
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_blocks))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered

def extended_kalman_filter(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances"], # "predicted_means", "predicted_covariances"],
    trial_masks = None,
    mode = 'hybrid',
    num_iters = 1,
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
    H = jacfwd(h, argnums=0, has_aux=True)
    # HH = hessian(h)

    Q = params.dynamics_covariance
    dv = Q.shape[-1]

    def _step(carry, t):
        ll, _pred_mean, _pred_cov = carry

        # Get parameters and inputs for time index t
        R = params.emission_covariance[t]
        y = emissions[t]
        trial_mask = trial_masks[t]

        if mode == 'hybrid':
            def true_fun(inputs):
                def _update_step(carry, _):
                    _pred_mean, _pred_cov = carry

                    # Get the Jacobian of the emission function
                    H_x, y_pred = H(_pred_mean)  # (ND x V), ND
                    # y_pred = h(_pred_mean)  # ND

                    _pred_pre = inv_via_cholesky(_pred_cov)
                    filtered_pre = _pred_pre + H_x.T @ jscipy.linalg.block_diag(*R) @ H_x
                    filtered_cov = inv_via_cholesky(filtered_pre)
                    filtered_mean = _pred_mean - filtered_cov @ H_x.T @ (jscipy.linalg.block_diag(*R) @ y_pred - y.flatten())
                    return (filtered_mean, filtered_cov), None
                
                (filtered_mean, filtered_cov), _ = lax.scan(_update_step, inputs, jnp.arange(num_iters))
                return filtered_mean, filtered_cov

            def false_fun(inputs):
                _pred_mean, _pred_cov = inputs
                return _pred_mean, _pred_cov

            inputs = (_pred_mean, _pred_cov)
            filtered_mean, filtered_cov = jax.lax.cond(trial_mask, true_fun, false_fun, inputs)

            pred_mean = filtered_mean
            pred_cov = Q + filtered_cov

        elif mode == 'cov':
            def true_fun(inputs):
                def _update_step(carry, _):
                    _pred_mean, _pred_cov = carry
                    # Get the Jacobian of the emission function
                    H_x, y_pred = H(_pred_mean)  # (ND x V), ND
                    # y_pred = h(_pred_mean)  # ND

                    s_k = H_x @ _pred_cov @ H_x.T + jscipy.linalg.block_diag(*R)

                    # Condition on this emission
                    K = psd_solve(s_k, H_x @ _pred_cov).T
                    filtered_cov = _pred_cov -  K @ s_k @ K.T
                    filtered_mean = _pred_mean + K @ (y.flatten() - y_pred)
                    filtered_cov = symmetrize(filtered_cov)
                    return (filtered_mean, filtered_cov), None
                
                (filtered_mean, filtered_cov), _ = lax.scan(_update_step, inputs, jnp.arange(num_iters))
                return filtered_mean, filtered_cov

            def false_fun(inputs):
                _pred_mean, _pred_cov = inputs
                return _pred_mean, _pred_cov

            inputs = (_pred_mean, _pred_cov)
            filtered_mean, filtered_cov = jax.lax.cond(trial_mask, true_fun, false_fun, inputs)

            # Predict the next state
            pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, Q)

            # normalize the eigenvalues of the predicted covariance by their maximum
            # L, U = jnp.linalg.eigh(filtered_cov)
            lambda_max, _ = power_iteration(pred_cov)
            threshold = 1e-3
            should_normalize = lambda_max > threshold
            # jax.debug.print('max_eigval: {max_eigval}', max_eigval=jnp.max(L))
            pred_cov = jnp.where(should_normalize, 
                                 threshold * pred_cov / lambda_max, 
                                 pred_cov)

        # Build carry and output states
        carry = (ll, pred_mean, pred_cov)
        outputs = {
            "filtered_means": filtered_mean,
            "filtered_covariances": filtered_cov,
            # "predicted_means": _pred_mean,
            # "predicted_covariances": _pred_cov,
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
    num_iters = 1,
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
        filtered_posterior = extended_kalman_filter(params, emissions, trial_masks=trial_masks, mode=mode, num_iters=num_iters)
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
        G = psd_solve(S_pred, filtered_cov).T

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
        G = psd_solve(S_pred, filtered_cov).T

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
