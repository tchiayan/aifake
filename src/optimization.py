from ray.train.lightning import (
    RayDDPStrategy,
    prepare_trainer,
    RayLightningEnvironment,
    RayTrainReportCallback
)
from model import FakeModelDetection
from dataset import FakeDataModule
from ray import tune
from ray.tune.schedulers import ASHAScheduler # Async Hyperband Scheduler
import lightning as pl
from ray.train import RunConfig , ScalingConfig , CheckpointConfig
from ray.train.torch import TorchTrainer

def train_func(config):
    dm = FakeDataModule(batch_size=config["batch_size"] , data_folder="/home/tchiayan/aifake/data/preprocessed")
    model = FakeModelDetection(lr=config["lr"] , weight_decay=config["weight_decay"])

    trainer = pl.Trainer(
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=RayLightningEnvironment(),
        enable_progress_bar=False
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model , datamodule=dm)

def main():
    # Configuring search space
    search_space = {
        "lr": tune.loguniform(1e-5 , 1e-2),
        "weight_decay": tune.loguniform(1e-6 , 1e-4),
        "batch_size": tune.choice([16 , 32])
    }

    # Define the asynchronus hyperband scheduler
    num_epochs = 30
    num_samples = 10
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    # Specify the number of workers and resources per worker
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 4, "GPU": 1}
    )
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_acc",
            checkpoint_score_order="max",
        ),
    )
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config
    )

    # Create Tuner object and start ray tune
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_acc",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )

    results = tuner.fit()
    print(results)


if __name__ == "__main__":
    main()
