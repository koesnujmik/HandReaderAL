from typing import Optional

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models import AbstractTrainer
from src.utils import Active_Learning_Sampler
from src.utils import LOGGER, seed_everything

if len(LOGGER.handlers) > 1:
    LOGGER.handlers = []


def train(
    cfg: DictConfig, model, stage, train_loader=None, unlabeled_loader=None, val_loader=None, lp_model=None
):
    """
    Train the model.
    Parameters
    ----------
    cfg: DictConfig
        The configuration to use for training.
    """
    LOGGER.info(f"Instantiating optimizer <{cfg.optimizer._target_}>")
    optimizer_partial = hydra.utils.instantiate(cfg.optimizer)
    params = list(model.parameters()) + (list(lp_model.parameters()) if lp_model else [])
    optimizer = optimizer_partial(params)

    scheduler = None
    if cfg.get("scheduler"):
        LOGGER.info(f"Instantiating scheduler <{cfg.scheduler._target_}>")
        scheduler_partial = hydra.utils.instantiate(cfg.scheduler)
        scheduler = scheduler_partial(optimizer)

    LOGGER.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer_partial = hydra.utils.instantiate(cfg.trainer, _recursive_=True)
    trainer: AbstractTrainer = trainer_partial(
        model=model, optimizer=optimizer, scheduler=scheduler, stage=stage,
        lp_model=lp_model, lp_weight=getattr(cfg.trainer, "lp_weight", 0.0), lp_feature_tap=getattr(cfg.trainer, "lp_feature_tap", "mlp"), lp_beta=getattr(cfg.trainer, "lp_beta", 1.0),
    )

    LOGGER.info("Starting training!")
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=cfg.base.epochs)

def test(cfg: DictConfig, stage, model, test_loader=None):
    """
    Test a model on a given dataset split.

    Parameters
    ----------
    cfg: DictConfig
        The configuration to use for testing.
    """

    LOGGER.info(f"Instantiating testers <{cfg.test.tester._target_}>")
    tester = hydra.utils.instantiate(cfg.test.tester)

    LOGGER.info("Starting test!")
    tester.run_test(model=model, loader=test_loader, device=torch.device(cfg.test.device), stage=stage)

@hydra.main(version_base="1.3", config_path="../configs", config_name=None)
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for training.
    Parameters
    ----------
    cfg: DictConfig
        The configuration composed by Hydra.

    Returns
    -------
    Optional[float] with optimized metric value.
    """
    save_dir = Path(cfg.trainer.path_to_save_weights)
    save_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("seed"):
        LOGGER.info(f"Seed set to {cfg.seed}")
        seed_everything(cfg.seed)
    
    if cfg.get("wandb") and cfg.wandb.enable:
        wandb.login(key=cfg.wandb.key)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.task_name,          # <-- run 이름
            group=cfg.wandb.group,
            tags=cfg.tags,
            notes=cfg.wandb.notes,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
        )
        wandb.define_metric("global_step")                       # 숫자 축 정의
        wandb.define_metric("Loss/*", step_metric="global_step")
        wandb.define_metric("Acc/*",  step_metric="global_step")
        wandb.define_metric("Test/*", step_metric="stage")
        wandb.define_metric("LP/*", step_metric="global_step")    
    
    LOGGER.info(f"Instantiating datamoudle <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_set = datamodule.get_dataset("train")
    val_loader = datamodule.get_dataloader("val")

    LOGGER.info(f"Instantiating test datamoudle <{cfg.test.datamodule._target_}>")
    test_datamodule = hydra.utils.instantiate(cfg.test.datamodule)

    if cfg.test.split == "val":
        test_loader = test_datamodule.get_dataloader("val")
    else:
        test_loader = test_datamodule.get_dataloader("test")
    
    LOGGER.info(f"Instantiating Sampler <{cfg.sampler._target_}>")
    sampler_partial = hydra.utils.instantiate(cfg.sampler, _recursive_=True)
    sampler: Active_Learning_Sampler = sampler_partial(train_set=train_set)

    labeled_pool, unlabeled_pool = sampler.initial()

    for stage in range(9):
        LOGGER.info(f"Starting stage {stage}")
        LOGGER.info(f"Instantiating model <{cfg.model._target_}>")
        model = hydra.utils.instantiate(cfg.model)

        lp_model = None
        if cfg.get("loss_prediction_model"):
            lp_model = hydra.utils.instantiate(cfg.loss_prediction_model)
            lp_model = lp_model.to(torch.device(cfg.trainer.device))
        
        decoder = hydra.utils.instantiate(cfg.trainer.decoder)
        
        train_loader = datamodule.get_dataloader_with_dataset("train", labeled_pool)
        unlabeled_loader = datamodule.get_dataloader_with_dataset("unlabeled", unlabeled_pool)

        train(cfg, model, stage, train_loader, unlabeled_loader, val_loader, lp_model=lp_model)
        test(cfg, stage, model, test_loader)

        labeled_pool, unlabeled_pool = sampler.sampling(model, train_loader, unlabeled_loader, lp_model=lp_model, decoder=decoder)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    import gc

    gc.collect()
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    main()