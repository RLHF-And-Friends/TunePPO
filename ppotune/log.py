import torch
import wandb
import os

import typing as tp
import torch.distributed as dist

from omegaconf import DictConfig, OmegaConf
from torchtune.training.metric_logging import MetricLoggerInterface, Scalar
from pathlib import Path


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

        self._completions = wandb.Table( # TODO: deprecate in favor of table reference
            columns=["completion", "score"]
        )
        self._table_reference: tp.Dict[str, wandb.Table] = {}

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

    # TODO: deprecate in favor of table refernce. add collect_table_row method instead
    def collect_completion(self, completion: str, score: torch.Tensor) -> None:
        """
        Collect completion and score.
        """
        self._completions.add_data(completion, score)

    def collect_table(
        self,
        name: str,
        columns: tp.Dict[str, tp.Iterable[tp.Any]]
    ) -> None:
        self._table_reference[name] = wandb.Table(columns=list(columns.keys()))
        for row in zip(*columns.values()):
            self._table_reference[name].add_data(*row)

    def flush(self, step: int) -> None:
        """
        Flush the log buffer to wandb.
        """
        for name in self._log_buffer:
            self.log(name, torch.stack(self._log_buffer[name]).mean(), step)

        self._log_buffer = {}

        for name, table in self._table_reference.items():
            self.log(name, table, step)

        self._table_reference = {}

        if len(self._completions.data) != 0:
            self.log("completions", self._completions, step)
            self._completions = wandb.Table(columns=[
                "completion", "score"
            ])


    def close(self) -> None:
        wandb.finish()

class DiskLogger(MetricLoggerInterface):
    """
    Singleton class to log into disk.
    """
    _instance: tp.Optional[tp.Self] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiskLogger, cls).__new__(cls)

        return cls._instance

    def setup(self, config: DictConfig) -> None:
        """
        Initialize wandb itself with separate runs for each device.
        """
        log_dir = config.get("log_dir")
        filename = config.get("filename")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file_name = self.log_dir / filename
        self._file = open(self._file_name, "a")
        print(f"Writing logs to {self._file_name}")

        self._completions: list[tp.Dict[str, str | torch.Tensor]] = []


    def log_dict(self, payload: tp.Mapping[str, Scalar], step: int) -> None:
        self._file.write(f"Step {step} | ")
        for name, data in payload.items():
            self._file.write(f"{name}:{data} ")
        self._file.write("\n")
        self._file.flush()

    def collect_completion(self, completion: str, score: torch.Tensor, final_answer: str, path: str) -> None:
        """
        Collect completion and score.
        """
        raw = {
            "completion": completion,
            "score": score,
            "final_answer": final_answer,
            "path": path
        }
        self._completions.append(raw)

    def flush(self, step: int) -> None:
        """
        Flush the log buffer to wandb.
        """
        for raw in self._completions:
            self.log_dict(raw, step)
        self._completions = []

    def close(self) -> None:
        self._file.close()
