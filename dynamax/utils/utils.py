from functools import partial
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jscipy
from jax import jit
from jax import vmap
from jax.tree_util import tree_map, tree_leaves, tree_flatten, tree_unflatten
import jax
import jaxlib
from jaxtyping import Array, Int
from scipy.optimize import linear_sum_assignment
from typing import Optional
from jax.scipy.linalg import cho_factor, cho_solve
import itertools
import mpmath

def has_tpu():
    try:
        return isinstance(jax.devices()[0], jaxlib.xla_extension.TpuDevice)
    except:
        return False


@jit
def pad_sequences(observations, valid_lens, pad_val=0):
    """
    Pad ragged sequences to a fixed length.
    Parameters
    ----------
    observations : array(N, seq_len)
        All observation sequences
    valid_lens : array(N, seq_len)
        Consists of the valid length of each observation sequence
    pad_val : int
        Value that the invalid observable events of the observation sequence will be replaced
    Returns
    -------
    * array(n, max_len)
        Ragged dataset
    """

    def pad(seq, len):
        idx = jnp.arange(1, seq.shape[0] + 1)
        return jnp.where(idx <= len, seq, pad_val)

    dataset = vmap(pad, in_axes=(0, 0))(observations, valid_lens), valid_lens
    return dataset


def monotonically_increasing(x, atol=0, rtol=0):
    thresh = atol + rtol*jnp.abs(x[:-1])
    return jnp.all(jnp.diff(x) >= -thresh)


def pytree_len(pytree):
    if pytree is None:
        return 0
    else:
        return len(tree_leaves(pytree)[0])


def pytree_sum(pytree, axis=None, keepdims=None, where=None):
    return tree_map(partial(jnp.sum, axis=axis, keepdims=keepdims, where=where), pytree)


def pytree_slice(pytree, slc):
    return tree_map(lambda x: x[slc], pytree)


def pytree_stack(pytrees):
    _, treedef = tree_flatten(pytrees[0])
    leaves = [tree_leaves(tree) for tree in pytrees]
    return tree_unflatten(treedef, [jnp.stack(vals) for vals in zip(*leaves)])

def random_rotation(seed, n, theta=None):
    r"""Helper function to create a rotating linear system.

    Args:
        seed (jax.random.PRNGKey): JAX random seed.
        n (int): Dimension of the rotation matrix.
        theta (float, optional): If specified, this is the angle of the rotation, otherwise
            a random angle sampled from a standard Gaussian scaled by ::math::`\pi / 2`. Defaults to None.
    Returns:
        [type]: [description]
    """

    key1, key2 = jr.split(seed)

    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * jnp.pi * jr.uniform(key1)

    if n == 1:
        return jr.uniform(key1) * jnp.eye(1)

    rot = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    out = jnp.eye(n)
    out = out.at[:2, :2].set(rot)
    q = jnp.linalg.qr(jr.uniform(key2, shape=(n, n)))[0]
    return q.dot(out).dot(q.T)

def random_dynamics_weights(key, n, num_rotations):
    key, key_root = jr.split(key)
    dynamics_weights = vmap(random_rotation, in_axes=(0, None))(jr.split(key_root, num_rotations), n)
    dynamics = jnp.eye(n)
    for i in range(len(dynamics_weights)):
        dynamics = dynamics @ dynamics_weights[i]
    return dynamics


def ensure_array_has_batch_dim(tree, instance_shapes):
    """Add a batch dimension to a PyTree, if necessary.

    Example: If `tree` is an array of shape (T, D) where `T` is
    the number of time steps and `D` is the emission dimension,
    and if `instance_shapes` is a tuple (D,), then the return
    value is the array with an added batch dimension, with
    shape (1, T, D).

    Example: If `tree` is an array of shape (N,TD) and
    `instance_shapes` is a tuple (D,), then the return
    value is simply `tree`, since it already has a batch
    dimension (of length N).

    Example: If `tree = (A, B)` is a tuple of arrays with
    `A.shape = (100,2)` `B.shape = (100,4)`, and
    `instances_shapes = ((2,), (4,))`, then the return value
    is equivalent to `(jnp.expand_dims(A, 0), jnp.expand_dims(B, 0))`.

    Args:
        tree (_type_): PyTree whose leaves' shapes are either
            (batch, length) + instance_shape or (length,) + instance_shape.
            If the latter, this function adds a batch dimension of 1 to
            each leaf node.

        instance_shape (_type_): matching PyTree where the "leaves" are
            tuples of integers specifying the shape of one "instance" or
            entry in the array.
    """
    def _expand_dim(x, shp):
        ndim = len(shp)
        assert x.ndim > ndim, "array does not match expected shape!"
        assert all([(d1 == d2) for d1, d2 in zip(x.shape[-ndim:], shp)]), \
            "array does not match expected shape!"

        if x.ndim == ndim + 2:
            # x already has a batch dim
            return x
        elif x.ndim == ndim + 1:
            # x has a leading time dimension but no batch dim
            return jnp.expand_dims(x, 0)
        else:
            raise Exception("array has too many dimensions!")

    if tree is None:
        return None
    else:
        return tree_map(_expand_dim, tree, instance_shapes)


def compute_state_overlap(
    z1: Int[Array, "num_timesteps"],
    z2: Int[Array, "num_timesteps"]
):
    """
    Compute a matrix describing the state-wise overlap between two state vectors
    ``z1`` and ``z2``.

    The state vectors should both of shape ``(T,)`` and be integer typed.

    Args:
        z1: The first state vector.
        z2: The second state vector.

    Returns:
        overlap matrix: Matrix of cumulative overlap events.
    """
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K = max(z1.max(), z2.max()) + 1

    overlap = jnp.sum(
        (z1[:, None] == jnp.arange(K))[:, :, None]
        & (z2[:, None] == jnp.arange(K))[:, None, :],
        axis=0,
    )
    return overlap


def find_permutation(
    z1: Int[Array, "num_timesteps"],
    z2: Int[Array, "num_timesteps"]
):
    """
    Find the permutation of the state labels in sequence ``z1`` so that they
    best align with the labels in ``z2``.

    Args:
        z1: The first state vector.
        z2: The second state vector.

    Returns:
        permutation such that ``jnp.take(perm, z1)`` best aligns with ``z2``.
        Thus, ``len(perm) = min(z1.max(), z2.max()) + 1``.

    """
    overlap = compute_state_overlap(z1, z2)
    _, perm = linear_sum_assignment(-overlap)
    return perm


def psd_solve(A, b, diagonal_boost=1e-16):
    """A wrapper for coordinating the linalg solvers used in the library for psd matrices."""
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = cho_factor(A, lower=True)
    x = cho_solve((L, lower), b)
    return x

def symmetrize(A):
    """Symmetrize one or more matrices."""
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))

def inv_via_cholesky(A, diagonal_boost=1e-16):
    """
    Compute a robust inverse of a positive–definite matrix A via Cholesky factorization.
    """
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = cho_factor(A, lower=True)
    x = cho_solve((L, lower), jnp.eye(A.shape[-1]))
    return x

def cayley_map(A):
    """
    Compute the Cayley map of a matrix A.
    """
    N = A.shape[0]
    I = jnp.eye(N, dtype=A.dtype)
    return jnp.linalg.solve(I - A, I + A)

def rotate_subspace(base_subspace, D, v):
    N = base_subspace.shape[0]
    in_manifold_dof = D * (D - 1) // 2

    in_manifold_v, out_manifold_v = jnp.split(v, [in_manifold_dof])
    in_manifold_rotation = jnp.zeros((D, D))
    in_manifold_rotation = in_manifold_rotation.at[jnp.triu_indices(D, k=1)].set(in_manifold_v)
    in_manifold_rotation = in_manifold_rotation - in_manifold_rotation.T

    out_manifold_dof_shape = (D, (N - D))

    rotation = jnp.zeros((N, N))
    rotation = rotation.at[:D, D:].set(out_manifold_v.reshape(out_manifold_dof_shape))
    rotation -= rotation.T
    rotation = rotation.at[:D, :D].set(in_manifold_rotation)
    # rotation = jscipy.linalg.expm(rotation)
    rotation = cayley_map(rotation)
    new_subspace = base_subspace @ rotation

    return new_subspace[:, :D]

def proj(u, a):
    return ((u @ a) / (u @ u)) * u

def normalize(x):
    return x / jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)

def gram_schmidt(mat):
    output = jnp.zeros_like(mat)
    obs_dim, state_dim = mat.shape

    output = output.at[:, 0].set(normalize(mat[:, 0]))
    for d in range(1, state_dim):
        u2 = mat[:, d]
        for d_prime in range(d):
            u2 -= proj(output[:, d_prime], mat[:, d])
        output = output.at[:, d].set(normalize(u2))
    return output

def power_iteration(M, num_iters=1000):
    """
    Compute the largest eigenvalue/eigenvector of a matrix M (assumed real symmetric or PSD)
    using Power Iteration, unrolled with lax.scan for a fixed number of steps.
    
    Args:
      M: 2D array (N x N)
      num_iters: int, number of iteration steps.

    Returns:
      (lambda_max, v_final): largest eigenvalue (approx) and corresponding eigenvector (approx).
    """
    # Random init for the vector
    key = jr.PRNGKey(0)
    v0 = jr.normal(key, shape=(M.shape[0],))
    v0 = v0 / jnp.linalg.norm(v0)
    
    def iteration_func(v, _):
        # One step of power iteration
        v_new = M @ v
        v_new = v_new / jnp.linalg.norm(v_new)
        return v_new, v_new  # (new_carry, output_for_this_step)
    
    # Run 'num_iters' steps of iteration.
    # - carry is the vector 'v'
    # - we ignore the second output (scan outputs all intermediate vs)
    v_final, _ = jax.lax.scan(iteration_func, v0, None, length=num_iters)
    
    # Rayleigh quotient as approximate largest eigenvalue
    lambda_max = jnp.dot(v_final, M @ v_final)
    return lambda_max, v_final

def compute_rotation(observations, emissions):
    """
    Given:
      observations: jnp.ndarray of shape (num_trials, num_timesteps, emission_dim)
      emissions:    jnp.ndarray of shape (num_trials, emission_dim, latent_dim)
    Returns:
      R: jnp.ndarray of shape (latent_dim, latent_dim)
         A rotation matrix such that rotating each trial's emission matrix (i.e. computing O_i @ R)
         orders the latent columns in descending order of explained variance.
      explained_variances: jnp.ndarray of shape (latent_dim,)
         Variances explained by the rotated latent dimensions.
    """
    # Compute latent factors per trial: shape (num_trials, num_timesteps, latent_dim)
    # Using the fact that the emission matrices are orthogonal.
    latent_factors = jnp.einsum('...te,...el->...tl', observations, emissions)
    
    # Stack trials and timesteps together, so we have all latent factors in one 2D array.
    if latent_factors.ndim == 3:
        num_trials, num_timesteps, latent_dim = latent_factors.shape
    else:
        num_timesteps, latent_dim = latent_factors.shape
        num_trials = 1

    latent_factors_stacked = latent_factors.reshape(num_trials * num_timesteps, latent_dim)
    
    # Perform SVD on the aggregated latent factors.
    # latent_factors_stacked = U @ diag(S) @ Vt, with singular values S in descending order.
    U, S, Vt = jnp.linalg.svd(latent_factors_stacked, full_matrices=False)
    
    # The rotation matrix that aligns with the principal directions is given by V = Vt.T.
    # Rotating the latent factors as L_rot = latent_factors_stacked @ V will yield columns
    # with decreasing variance (S^2 are proportional to the variances).
    R = Vt.T
    
    # Compute the explained variances for each rotated latent dimension.
    explained_variances = (S ** 2) / (latent_factors_stacked.shape[0] - 1)
    
    return R, explained_variances

def squared_exponential_spectral_measure(m, sigma, kappa):
    C_inf = float(mpmath.jtheta(3, 0., mpmath.exp(-2 * mpmath.pi**2 * kappa**2)))
    return (sigma**2 / C_inf) * jnp.exp(- 2* jnp.pi**2 * kappa**2 * m**2)

def Tm_basis(N: int, M_conditions: int=1, sigma: float=1.0, kappa: float=1.0, period: jnp.ndarray | float = None) -> list:
    '''
    Regular Fourier Features sample approximation to GP over the M-dimensional torus.
    For M=1, this is equivalent to T1_basis.
    Args:
        N: number of basis functions for each dimension (so total number of basis functions is 2*(N**M_conditions - 1) + 1)
        M_conditions: number of conditions
        sigma: kernel parameter
        kappa: kernel parameter
        period: period of the torus. Provide `period >= data_interval + 6 * kappa` for non-periodic data.
    '''
    def coef(index_array):
        return jnp.sqrt(squared_exponential_spectral_measure(jnp.linalg.norm(index_array), sigma, kappa))
    
    if period is None:
        period = jnp.ones(M_conditions)
    if isinstance(period, float):
        period = period * jnp.ones(M_conditions)

    basis_funcs = []
    for index in itertools.product(jnp.arange(N), repeat=M_conditions):
        if index == (0,)*M_conditions:
            constant_func = lambda x: coef(jnp.zeros(M_conditions))
            basis_funcs.append(constant_func) # only one constant function
        else:
            def _f_sin(x, index=index): # use defaults to avoid late binding
                return coef(jnp.array(index)) * jnp.sin(2*jnp.pi * jnp.dot(jnp.array(index), jnp.divide(x, period)))
            def _f_cos(x, index=index):
                return coef(jnp.array(index)) * jnp.cos(2*jnp.pi * jnp.dot(jnp.array(index), jnp.divide(x, period)))
            basis_funcs.append(_f_sin)
            basis_funcs.append(_f_cos)

    assert len(basis_funcs) == 2*(N**M_conditions - 1) + 1
    return basis_funcs


def rbf_basis(N: int, M_conditions: int=1, sigma: float=1.0, kappa: float=1.0) -> list:
    '''
    Radial Basis Function (Gaussian bump) approximation to GP for non-periodic data.

    Args:
        N: number of basis functions (centers evenly spaced in [0, 1] per dimension)
        M_conditions: number of conditions (dimensions of input)
        sigma: kernel amplitude parameter (controls overall scale)
        kappa: kernel lengthscale parameter (larger = wider bumps = smoother)

    Returns:
        List of basis functions. Total count is N**M_conditions.
    '''
    # Generate centers as grid in [0, 1]^M_conditions
    centers_1d = jnp.linspace(0, 1, N)

    basis_funcs = []
    for center_idx in itertools.product(range(N), repeat=M_conditions):
        center = jnp.array([centers_1d[i] for i in center_idx])

        def _f(x, center=center, sigma=sigma, kappa=kappa):
            # x can be scalar or array of shape (M_conditions,)
            x = jnp.atleast_1d(x)
            dist_sq = jnp.sum((x - center) ** 2)
            return sigma * jnp.exp(-0.5 * dist_sq / (kappa ** 2))

        basis_funcs.append(_f)

    assert len(basis_funcs) == N**M_conditions
    return basis_funcs