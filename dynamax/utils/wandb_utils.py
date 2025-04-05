import os
import wandb
import yaml
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

def init_wandb(config_path: str, mode: str = "online", project: str = "smds", entity: Optional[str] = None):
    """Initialize wandb with configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        mode: wandb mode ('online', 'offline', 'disabled')
        project: wandb project name
        entity: wandb entity name (username or team name)
    
    Returns:
        wandb run object
    """
    # Load config from YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb
    run = wandb.init(
        project=project,
        entity=entity,
        config=config,
        mode=mode
    )
    
    return run, config

def log_training_step(run, metrics: Dict[str, Any], step: int):
    """Log training metrics to wandb.
    
    Args:
        run: wandb run object
        metrics: Dictionary of metrics to log
        step: Current training step
    """
    # Convert JAX arrays to numpy for wandb
    metrics_np = {}
    for k, v in metrics.items():
        if isinstance(v, jnp.ndarray):
            metrics_np[k] = np.array(v)
        else:
            metrics_np[k] = v
    
    run.log(metrics_np, step=step)

def save_model(run, params, model_dir: str, model_name: str):
    """Save model parameters and upload to wandb.
    
    Args:
        run: wandb run object
        params: Model parameters
        model_dir: Directory to save model
        model_name: Model file name
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.npz")
    
    # Convert JAX arrays to numpy for saving
    params_np = {}
    for k, v in params._asdict().items():
        if hasattr(v, '_asdict'):
            params_np[k] = {sk: np.array(sv) for sk, sv in v._asdict().items()}
        else:
            params_np[k] = np.array(v)
    
    np.savez(model_path, **params_np)
    
    # Log model to wandb
    artifact = wandb.Artifact(name=model_name, type='model')
    artifact.add_file(model_path)
    run.log_artifact(artifact)

def log_evaluation_metrics(run, metrics: Dict[str, Any]):
    """Log evaluation metrics to wandb.
    
    Args:
        run: wandb run object
        metrics: Dictionary of metrics to log
    """
    # Convert JAX arrays to numpy for wandb
    metrics_np = {}
    for k, v in metrics.items():
        if isinstance(v, jnp.ndarray):
            metrics_np[k] = np.array(v)
        else:
            metrics_np[k] = v
    
    run.summary.update(metrics_np)

def get_best_run(project: str, metric: str, mode: str = "max", entity: Optional[str] = None) -> Tuple[str, float]:
    """Get the best run from a wandb project based on a metric.
    
    Args:
        project: wandb project name
        metric: Metric to optimize
        mode: 'max' or 'min'
        entity: wandb entity name
    
    Returns:
        Tuple of (run_id, metric_value)
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}" if entity else project)
    
    best_value = float('-inf') if mode == 'max' else float('inf')
    best_run_id = None
    
    for run in runs:
        if metric in run.summary:
            value = run.summary[metric]
            if (mode == 'max' and value > best_value) or (mode == 'min' and value < best_value):
                best_value = value
                best_run_id = run.id
    
    return best_run_id, best_value

def load_model_from_run(run_id: str, entity: Optional[str] = None, project: str = "smds") -> Dict[str, Any]:
    """Load model parameters from a wandb run.
    
    Args:
        run_id: wandb run ID
        entity: wandb entity name
        project: wandb project name
    
    Returns:
        Dictionary of model parameters
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}")
    
    # Find the model artifact
    artifacts = run.logged_artifacts()
    model_artifact = None
    for artifact in artifacts:
        if artifact.type == 'model':
            model_artifact = artifact
            break
    
    if not model_artifact:
        raise ValueError(f"No model artifact found in run {run_id}")
    
    # Download the model
    model_dir = model_artifact.download()
    model_files = os.listdir(model_dir)
    model_file = None
    for f in model_files:
        if f.endswith('.npz'):
            model_file = f
            break
    
    if not model_file:
        raise ValueError(f"No model file found in artifact {model_artifact.name}")
    
    # Load the model
    model_path = os.path.join(model_dir, model_file)
    model = np.load(model_path, allow_pickle=True)
    
    return dict(model) 