from abc import ABC
from abc import abstractmethod
from fastprogress.fastprogress import progress_bar
from functools import partial
import jax.numpy as jnp
import jax.random as jr
from jax import jit, lax, vmap
from jax.tree_util import tree_map
from jaxtyping import Float, Array, PyTree
import optax
from tensorflow_probability.substrates.jax import distributions as tfd
from typing import Optional, Union, Tuple, Any
from typing_extensions import Protocol

from dynamax.parameters import to_unconstrained, from_unconstrained
from dynamax.parameters import ParameterSet, PropertySet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.optimize import run_sgd
from dynamax.utils.utils import ensure_array_has_batch_dim
from dynamax.utils.utils import rotate_subspace

class Posterior(Protocol):
    """A :class:`NamedTuple` with parameters stored as :class:`jax.DeviceArray` in the leaf nodes."""
    pass

class SuffStatsSSM(Protocol):
    """A :class:`NamedTuple` with sufficient statics stored as :class:`jax.DeviceArray` in the leaf nodes."""
    pass

class SSM(ABC):
    r"""A base class for state space models. Such models consist of parameters, which
    we may learn, as well as hyperparameters, which specify static properties of the
    model. This base class allows parameters to be indicated a standardized way
    so that they can easily be converted to/from unconstrained form for optimization.

    **Abstract Methods**

    Models that inherit from `SSM` must implement a few key functions and properties:

    * :meth:`initial_distribution` returns the distribution over the initial state given parameters
    * :meth:`transition_distribution` returns the conditional distribution over the next state given the current state and parameters
    * :meth:`emission_distribution` returns the conditional distribution over the emission given the current state and parameters
    * :meth:`log_prior` (optional) returns the log prior probability of the parameters
    * :attr:`emission_shape` returns a tuple specification of the emission shape
    * :attr:`inputs_shape` returns a tuple specification of the input shape, or `None` if there are no inputs.

    The shape properties are required for properly handling batches of data.

    **Sampling and Computing Log Probabilities**

    Once these have been implemented, subclasses will inherit the ability to sample
    and compute log joint probabilities from the base class functions:

    * :meth:`sample` draws samples of the states and emissions for given parameters
    * :meth:`log_prob` computes the log joint probability of the states and emissions for given parameters

    **Inference**

    Many subclasses of SSMs expose basic functions for performing state inference.

    * :meth:`marginal_log_prob` computes the marginal log probability of the emissions, summing over latent states
    * :meth:`filter` computes the filtered posteriors
    * :meth:`smoother` computes the smoothed posteriors

    **Learning**

    Likewise, many SSMs will support learning with expectation-maximization (EM) or stochastic gradient descent (SGD).

    For expectation-maximization, subclasses must implement the E- and M-steps.

    * :meth:`e_step` computes the expected sufficient statistics for a sequence of emissions, given parameters
    * :meth:`m_step` finds new parameters that maximize the expected log joint probability

    Once these are implemented, the generic SSM class allows to fit the model with EM

    * :meth:`fit_em` run EM to find parameters that maximize the likelihood (or posterior) probability.

    For SGD, any subclass that implements :meth:`marginal_log_prob` inherits the base class fitting function

    * :meth:`fit_sgd` run SGD to minimize the *negative* marginal log probability.

    """

    @abstractmethod
    def initial_distribution(
        self,
        params: ParameterSet,
        inputs: Optional[Float[Array, "input_dim"]]
    ) -> tfd.Distribution:
        r"""Return an initial distribution over latent states.

        Args:
            params: model parameters $\theta$
            inputs: optional  inputs  $u_t$

        Returns:
            distribution over initial latent state, $p(z_1 \mid \theta)$.

        """
        raise NotImplementedError

    @abstractmethod
    def transition_distribution(
        self,
        params: ParameterSet,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]
    ) -> tfd.Distribution:
        r"""Return a distribution over next latent state given current state.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            conditional distribution of next latent state $p(z_{t+1} \mid z_t, u_t, \theta)$.

        """
        raise NotImplementedError

    @abstractmethod
    def emission_distribution(
        self,
        params: ParameterSet,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]=None
    ) -> tfd.Distribution:
        r"""Return a distribution over emissions given current state.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            conditional distribution of current emission $p(y_t \mid z_t, u_t, \theta)$

        """
        raise NotImplementedError

    def log_prior(
        self,
        params: ParameterSet
    ) -> Scalar:
        r"""Return the log prior probability of any model parameters.

        Returns:
            lp (Scalar): log prior probability.
        """
        return 0.0

    @property
    @abstractmethod
    def emission_shape(self) -> Tuple[int]:
        r"""Return a pytree matching the pytree of tuples specifying the shape of a single time step's emissions.

        For example, a `GaussianHMM` with $D$ dimensional emissions would return `(D,)`.

        """
        raise NotImplementedError

    @property
    def inputs_shape(self) -> Optional[Tuple[int]]:
        r"""Return a pytree matching the pytree of tuples specifying the shape of a single time step's inputs.

        """
        return None

    # All SSMs support sampling
    def sample(
        self,
        params: ParameterSet,
        key: PRNGKey,
        num_timesteps: int,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps state_dim"],
              Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\theta$ and (optionally) inputs $u_{1:T}$.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            inputs: inputs $u_{1:T}$

        Returns:
            latent states and emissions

        """
        def _step(prev_state, args):
            key, inpt = args
            key1, key2 = jr.split(key, 2)
            state = self.transition_distribution(params, prev_state, inpt).sample(seed=key2)
            emission = self.emission_distribution(params, state, inpt).sample(seed=key1)
            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_input = tree_map(lambda x: x[0], inputs)
        initial_state = self.initial_distribution(params, initial_input).sample(seed=key1)
        initial_emission = self.emission_distribution(params, initial_state, initial_input).sample(seed=key2)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, next_inputs))

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

    def log_prob(
        self,
        params: ParameterSet,
        states: Float[Array, "num_timesteps state_dim"],
        emissions: Float[Array, "num_timesteps emission_dim"],
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Scalar:
        r"""Compute the log joint probability of the states and observations"""

        def _step(carry, args):
            lp, prev_state = carry
            state, emission, inpt = args
            lp += self.transition_distribution(params, prev_state, inpt).log_prob(state)
            lp += self.emission_distribution(params, state, inpt).log_prob(emission)
            return (lp, state), None

        # Compute log prob of initial time step
        initial_state = tree_map(lambda x: x[0], states)
        initial_emission = tree_map(lambda x: x[0], emissions)
        initial_input = tree_map(lambda x: x[0], inputs)
        lp = self.initial_distribution(params, initial_input).log_prob(initial_state)
        lp += self.emission_distribution(params, initial_state, initial_input).log_prob(initial_emission)

        # Scan over remaining time steps
        next_states = tree_map(lambda x: x[1:], states)
        next_emissions = tree_map(lambda x: x[1:], emissions)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        (lp, _), _ = lax.scan(_step, (lp, initial_state), (next_states, next_emissions, next_inputs))
        return lp

    # Some SSMs will implement these inference functions.
    def marginal_log_prob(
        self,
        params: ParameterSet,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Scalar:
        r"""Compute log marginal likelihood of observations, $\log \sum_{z_{1:T}} p(y_{1:T}, z_{1:T} \mid \theta)$.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            marginal log probability

        """
        raise NotImplementedError

    def filter(
        self,
        params: ParameterSet,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Posterior:
        r"""Compute filtering distributions, $p(z_t \mid y_{1:t}, u_{1:t}, \theta)$ for $t=1,\ldots,T$.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            filtering distributions

        """
        raise NotImplementedError

    def smoother(
        self,
        params: ParameterSet,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Posterior:
        r"""Compute smoothing distribution, $p(z_t \mid y_{1:T}, u_{1:T}, \theta)$ for $t=1,\ldots,T$.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            smoothing distributions

        """
        raise NotImplementedError

    # Learning algorithms
    def e_step(
        self,
        params: ParameterSet,
        emissions: Float[Array, "num_timesteps emission_dim"],
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[SuffStatsSSM, Scalar]:
        r"""Perform an E-step to compute expected sufficient statistics under the posterior, $p(z_{1:T} \mid y_{1:T}, u_{1:T}, \theta)$.

        Args:
            params: model parameters $\theta$
            emissions: emissions $y_{1:T}$
            inputs: optional inputs $u_{1:T}$

        Returns:
            Expected sufficient statistics under the posterior.

        """
        raise NotImplementedError

    def m_step(
        self,
        params: ParameterSet,
        props: PropertySet,
        batch_stats: SuffStatsSSM,
        m_step_state: Any
    ) -> ParameterSet:
        r"""Perform an M-step to find parameters that maximize the expected log joint probability.

        Specifically, compute

        $$\theta^\star = \mathrm{argmax}_\theta \; \mathbb{E}_{p(z_{1:T} \mid y_{1:T}, u_{1:T}, \theta)} \big[\log p(y_{1:T}, z_{1:T}, \theta \mid u_{1:T}) \big]$$

        Args:
            params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            batch_stats: sufficient statistics from each sequence
            m_step_state: any required state for optimizing the model parameters.

        Returns:
            new parameters

        """
        raise NotImplementedError

    def fit_em(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Union[Float[Array, "num_timesteps emission_dim"],
                         Float[Array, "num_batches num_timesteps emission_dim"]],
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
                               Float[Array, "num_batches num_timesteps input_dim"]]]=None,
        conditions: Optional[Float[Array, "num_batches"]] = None,
        trial_masks: jnp.array = None,
        num_iters: int=50,
        block_ids: jnp.array = None,
        block_masks: jnp.array = None,
        verbose: bool=True,
        print_ll: bool=False,
        run_velocity_smoother: bool = False,
        velocity_smoother_method: int = 0,
        use_wandb: bool = False,
        wandb_run = None,
    ) -> Tuple[ParameterSet, Float[Array, "num_iters"]]:
        """Fit the model with expectation maximization (EM) given a batch of sequences.
        
        Args:
            params: initial parameters
            props: parameter property specifications
            emissions: one or more emission sequences
            inputs: optional inputs
            conditions: optional conditions for each sequence
            trial_masks: optional mask for each trial
            num_iters: number of EM iterations
            block_ids: optional block IDs for block-structured observations (used by SMDS)
            block_masks: optional masks for blocks
            verbose: whether or not to show a progress bar
            print_ll: whether to print log likelihood at each step
            run_velocity_smoother: whether to run velocity smoother (used by SMDS)
            use_wandb: whether to use wandb for logging
            wandb_run: optional existing wandb run object
            model_dir: directory to save models
            model_name: name for saved model
            eval_data: optional tuple of (test_emissions, test_inputs, test_conditions, test_masks) for evaluation
            
        Returns:
            tuple of new parameters and log likelihoods over the course of EM iterations.
        
        """
        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)
        conditions = jnp.zeros(len(batch_emissions), dtype=int) if conditions is None else conditions
        trial_masks = jnp.ones(len(batch_emissions), dtype=bool) if trial_masks is None else trial_masks
        trial_ids = jnp.arange(len(batch_emissions), dtype=int)
        block_ids = jnp.eye(len(batch_emissions)) if block_ids is None else block_ids
        block_masks = jnp.ones(block_ids.shape[0], dtype=bool) if block_masks is None else block_masks
        num_blocks = block_ids.shape[0]
        block_size = len(batch_emissions) // num_blocks
        T, N = batch_emissions.shape[1:]
        
        # Import wandb utils only if needed
        if use_wandb:
            from dynamax.utils.wandb_utils import log_training_step, save_model, log_evaluation_metrics
            if wandb_run is None:
                import wandb
                wandb_run = wandb.run
                if wandb_run is None:
                    raise ValueError("No active wandb run found. Initialize wandb before calling fit_em or provide a run object.")
        
        @jit
        def em_step(params, m_step_state):
            if run_velocity_smoother:
                velocity_smoother = self.smoother(params, batch_emissions.reshape(num_blocks, block_size, T, N), 
                                                  conditions.reshape(num_blocks, block_size), block_masks,
                                                  method=velocity_smoother_method)
                Ev = velocity_smoother.smoothed_means
                Hs = vmap(rotate_subspace, in_axes=(None, None, 0))(params.emissions.base_subspace, self.state_dim, Ev)
                Hs = jnp.einsum('bij,bk->kij', Hs, block_ids)
                velocity_smoother_marginal_loglik = velocity_smoother.marginal_loglik
            else:
                velocity_smoother = None
                Hs = None
                velocity_smoother_marginal_loglik = 0.0

            batch_stats, lls, posteriors = vmap(partial(self.e_step, params))(batch_emissions,
                                                                              batch_inputs,
                                                                              conditions,
                                                                              trial_masks,
                                                                              trial_ids,
                                                                              Hs)
            lp = self.log_prior(params)
            params, m_step_state = self.m_step(params, props, batch_stats,
                                               m_step_state, posteriors,
                                               emissions, conditions, trial_masks,
                                               velocity_smoother, block_ids, block_masks)
            # debug.print('e_step: {x}', x=(batch_stats, lls))
            # debug.print('m_step{y}', y=params)
            return params, m_step_state, lp, lls.sum(), velocity_smoother_marginal_loglik

        log_probs = []
        m_step_state = self.initialize_m_step_state(params, props)
        pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
        best_lp = -jnp.inf
        best_params = params
        
        for iter_num in pbar:
            params, m_step_state, lp, ll, vel_ll = em_step(params, m_step_state)
            if run_velocity_smoother:
                total_lp = lp + vel_ll
                log_probs.append(total_lp)
            else:
                total_lp = lp + ll
                log_probs.append(lp + ll)
            
            if print_ll:
                print(iter_num, total_lp, lp, ll, vel_ll, params.emissions.tau.min(), params.emissions.tau.max(),
                    jnp.diag(params.emissions.initial_velocity_cov).max(), jnp.diag(params.emissions.cov).min())
                print('-----------------------------------------------------------------------------')
            
            if total_lp > best_lp:
                best_lp = total_lp
                best_params = params
            
            # Log metrics to wandb
            if use_wandb:
                metrics = {
                    # 'iteration': iter_num,
                    'total_log_prob': float(total_lp),
                    'log_prior': float(lp),
                    'log_likelihood': float(ll),
                }
                
                if hasattr(params.emissions, 'tau'):
                    metrics['tau_min'] = float(params.emissions.tau.min())
                    metrics['tau_max'] = float(params.emissions.tau.max())
                if hasattr(params.emissions, 'initial_velocity_cov'):
                    metrics['initial_velocity_cov_max'] = float(jnp.diag(params.emissions.initial_velocity_cov).max())
                if hasattr(params.emissions, 'cov'):
                    metrics['emissions_cov_min'] = float(jnp.diag(params.emissions.cov).min())
                
                log_training_step(wandb_run, metrics, iter_num)
        
        return best_params, jnp.array(log_probs)

    def fit_sgd(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Union[Float[Array, "num_timesteps emission_dim"],
                         Float[Array, "num_batches num_timesteps emission_dim"]],
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
                               Float[Array, "num_batches num_timesteps input_dim"]]]=None,
        optimizer: optax.GradientTransformation=optax.adam(1e-3),
        batch_size: int=1,
        num_epochs: int=50,
        shuffle: bool=False,
        key: PRNGKey=jr.PRNGKey(0)
    ) -> Tuple[ParameterSet, Float[Array, "niter"]]:
        r"""Compute parameter MLE/ MAP estimate using Stochastic Gradient Descent (SGD).

        SGD aims to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        by minimizing the _negative_ of that quantity.

        *Note:* ``emissions`` *and* ``inputs`` *can either be single sequences or batches of sequences.*

        On each iteration, the algorithm grabs a *minibatch* of sequences and takes a gradient step.
        One pass through the entire set of sequences is called an *epoch*.

        Args:
            params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            emissions: one or more sequences of emissions
            inputs: one or more sequences of corresponding inputs
            optimizer: an `optax` optimizer for minimization
            batch_size: number of sequences per minibatch
            num_epochs: number of epochs of SGD to run
            key: a random number generator for selecting minibatches
            verbose: whether or not to show a progress bar

        Returns:
            tuple of new parameters and losses (negative scaled marginal log probs) over the course of SGD iterations.

        """
        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        unc_params = to_unconstrained(params, props)

        def _loss_fn(unc_params, minibatch):
            """Default objective function."""
            params = from_unconstrained(unc_params, props)
            minibatch_emissions, minibatch_inputs = minibatch
            scale = len(batch_emissions) / len(minibatch_emissions)
            minibatch_lls = vmap(partial(self.marginal_log_prob, params))(minibatch_emissions, minibatch_inputs)
            lp = self.log_prior(params) + minibatch_lls.sum() * scale
            return -lp / batch_emissions.size

        dataset = (batch_emissions, batch_inputs)
        unc_params, losses = run_sgd(_loss_fn,
                                     unc_params,
                                     dataset,
                                     optimizer=optimizer,
                                     batch_size=batch_size,
                                     num_epochs=num_epochs,
                                     shuffle=shuffle,
                                     key=key)

        params = from_unconstrained(unc_params, props)
        return params, losses