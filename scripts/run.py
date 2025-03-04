import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="smds", version_base=None)
def main(args: DictConfig) -> None:
    train(args)
    return

def train(args):

    # Get hydra output directory
    try:
      output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except:
      output_dir = None

    # setup logger:
    if args.trainer_cfg.fast_dev_run:
      logger = None
      wandb.init(mode='disabled')
    else:
      if args.trainer_cfg.logger == "wandb":
        kwargs = dict()

        wandb.init(project="smds", name=args.experiment_name, config=args)

        args_as_dict = OmegaConf.to_container(args)
        wandb.config.update(args_as_dict)
      else:
        logger = None


    # get dataset:
    datamodule = SpeechDataModule(args.data_cfg)

    # get module and model:
    modelito = all_models[args.model_cfg.type](**args.model_cfg.configs, nDays=datamodule.nDays)
    if torch.cuda.is_available():
      try:
        modelito = torch.compile(modelito)
      except:
        print("did not compile model")

    if args.model_cfg.get("resume_ckpt_path", None):
      # Do not resume training simply load checkpoints
      weights = torch.load(args.model_cfg.resume_ckpt_path)['state_dict']
      weights = {k.replace('model.', ''): v for k, v in weights.items()}
      modelito.load_state_dict(weights)

    model = all_modules[args.module_cfg.type](args, modelito)

    # set trainer:
    trainer_config = OmegaConf.to_container(args.trainer_cfg)
    trainer_config['logger'] = logger

    # set callbacks
    local_callbacks = []
    if args.callbacks:
      if args.callbacks.get("lr_monitor", None):
        local_callbacks.append(LearningRateMonitor(**args.callbacks.lr_monitor))
      if args.callbacks.get("grad_norm") and args.callbacks.grad_norm.get("type", None):
        local_callbacks.append(all_callbacks[args.callbacks.grad_norm.type])
      if args.callbacks.get("early_stopping", None):
        local_callbacks.append(EarlyStopping(**args.callbacks.early_stopping))
      if args.callbacks.get("masking_scheduler", None):
        local_callbacks.append(all_callbacks["masking_scheduler"](**args.callbacks.masking_scheduler))
        trainer_config["reload_dataloaders_every_n_epochs"] = 1
      if args.callbacks.get("checkpoint_cfg", None):
        local_callbacks.append(ModelCheckpoint(**args.callbacks.checkpoint_cfg))
      if args.callbacks.get("eigen_track", None):
        local_callbacks.append(all_callbacks["eigen_track"])

    trainer = L.Trainer(**trainer_config, callbacks=local_callbacks)

    # Train model
    if not args.eval_cfg.get("eval_only", 0):
      trainer.fit(model=model, datamodule=datamodule)
      ckpt_path = "best" if not args.trainer_cfg.fast_dev_run else None
    else:
      ckpt_path = args.eval_cfg.get("ckpt_path", None)

    # Test model
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # End logging
    if args.trainer_cfg.logger == "wandb" and not (logger is None):
      wandb.run.summary["output_dir"] = output_dir
      wandb.run.summary["total_params"] = sum(p.numel() for p in model.parameters())

      wandb.finish()

    # Goodbye

if __name__ == "__main__":
    main()