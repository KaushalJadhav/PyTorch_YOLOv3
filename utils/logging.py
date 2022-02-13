try:
    import wandb 
except ModuleNotFoundError:
    pass

def init_wandb(cfg) -> None:
    """
    Initialize project on Weights & Biases
    Args:
        cfg   (Dict) : Configuration file
    """
    wandb.init(
        name=cfg["LOGGING"]["NAME"],
        config=cfg,
        project=cfg["LOGGING"]["PROJECT"]cfg.WANDB.PROJECT,
        resume="allow",
        id=cfg["LOGGING"]["ID"]
    )