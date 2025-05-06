import torch
import wandb
import os

import typing as tp
import torch.distributed as dist

from omegaconf import DictConfig, OmegaConf
from torchtune.training.metric_logging import MetricLoggerInterface, Scalar


class WandbLogger(MetricLoggerInterface):
    """
    Singleton class to log into W&B.
    """
    _instance: tp.Optional[tp.Self] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WandbLogger, cls).__new__(cls)

        return cls._instance

    def setup(self, config: DictConfig) -> None:
        """
        Initialize wandb itself with separate runs for each device.
        """
        if group := config.get("group"):
            group = f"{group}-{dist.get_world_size()}x"
        if name := config.get("name"):
            name = f"{name}-{dist.get_rank()}/{dist.get_world_size() - 1}"
        if additional_info := config.get("additional_info"):
            name = f"{name}-{additional_info[dist.get_rank()]}"

        dir = os.path.expanduser(config.dir)
        if not os.path.exists(dir):
            os.makedirs(dir)

        self._log_buffer: tp.Dict[str, list[torch.Tensor]] = {}

        self._completions = wandb.Table(
            columns=["completion", "score"]
        )
        self._references = wandb.Table(
            columns=["reference", "completion", "chosen"]
        )

        wandb.init(
            dir=dir,
            entity=config.entity,
            project=config.project,
            group=group,
            name=name,
        )
        # define default x-axis (for latest wandb versions)
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step", step_sync=True)


    def log(self, name: str, data: Scalar, step: int) -> None:
        wandb.log({name: data, "step": step})

    def log_dict(self, payload: tp.Mapping[str, Scalar], step: int) -> None:
        wandb.log({**payload, "step": step})

    def log_config(self, config: DictConfig) -> None:
        resolved = OmegaConf.to_container(config, resolve=True)
        wandb.config.update(resolved)

    def collect(self, name: str, data: torch.Tensor) -> None:
        """
        Collect log in logger buffer to aggregate and offload to wandb later.
        """
        data = data.detach()
        if name in self._log_buffer:
            self._log_buffer[name].append(data)
        else:
            self._log_buffer[name] = [data]

    def collect_dict(self, payload: tp.Mapping[str, torch.Tensor]) -> None:
        """
        Collect dict of logs.
        """
        for name in payload:
            self.collect(name, payload[name])

    def collect_completion(self, completion: str, score: torch.Tensor) -> None:
        """
        Collect completion and score.
        """
        self._completions.add_data(completion, score)

    def collect_reference(
        self,
        reference: str,
        completion: str,
        chosen: int
    ) -> None:
        """
        Collect reference completion, validated completion and chosen id.
        """
        self._references.add_data(reference, completion, chosen)

    def flush(self, step: int) -> None:
        """
        Flush the log buffer to wandb.
        """
        for name in self._log_buffer:
            self.log(name, torch.stack(self._log_buffer[name]).mean(), step)

        self._log_buffer = {}

        if len(self._completions.data) != 0:
            self.log("completions", self._completions, step)
            self._completions = wandb.Table(columns=[
                "completion", "score"
            ])

        if len(self._references.data) != 0:
            self.log("references", self._references, step)
            self._references = wandb.Table(columns=[
                "reference", "completion", "chosen"
            ])

    def close(self) -> None:
        wandb.finish()
