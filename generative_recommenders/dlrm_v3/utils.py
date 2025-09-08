# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""
mlperf dlrm_v3 inference benchmarking tool.
"""

import contextlib
import logging
import os
from typing import Callable, Dict, List, Optional

import gin
import tensorboard  # @manual=//tensorboard:lib  # noqa: F401 - required implicit dep when using torch.utils.tensorboard

import torch
from generative_recommenders.dlrm_v3.datasets.dataset import DLRMv3RandomDataset
from generative_recommenders.dlrm_v3.datasets.kuairand import DLRMv3KuaiRandDataset
from generative_recommenders.dlrm_v3.datasets.movie_lens import DLRMv3MovieLensDataset
from generative_recommenders.dlrm_v3.datasets.synthetic_movie_lens import (
    DLRMv3SyntheticMovieLensDataset,
)

from generative_recommenders.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from torch.profiler import profile, profiler, ProfilerActivity  # pyre-ignore [21]
from torch.utils.tensorboard import SummaryWriter
from torchrec.metrics.auc import AUCMetricComputation
from torchrec.metrics.mae import MAEMetricComputation
from torchrec.metrics.mse import MSEMetricComputation
from torchrec.metrics.ne import NEMetricComputation

from torchrec.metrics.rec_metric import RecMetricComputation

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("utils")


def _on_trace_ready_fn(
    rank: Optional[int] = None,
    trace_dir: Optional[str] = None,
) -> Callable[[torch.profiler.profile], None]:
    def handle_fn(p: torch.profiler.profile) -> None:
        pid = os.getpid()
        rank_str = f"_rank_{rank}" if rank is not None else ""
        file_name = f"libkineto_activities_{pid}_{rank_str}.json"
        
        # Use configurable trace directory or default to temp
        if trace_dir is None:
            import tempfile
            output_dir = os.path.join(tempfile.gettempdir(), "torch_traces")
        else:
            output_dir = trace_dir
            
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, file_name)
        
        logger.warning(
            p.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total"
            )
        )
        logger.warning(
            f"Trace saved to local path: {path}"
        )
        p.export_chrome_trace(path)

    return handle_fn


def profiler_or_nullcontext(enabled: bool, with_stack: bool):
    return (
        profile(
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=_on_trace_ready_fn(),
            with_stack=with_stack,
        )
        if enabled
        else contextlib.nullcontext()
    )


class Profiler:
    def __init__(self, rank, active: int = 50) -> None:
        self.rank = rank
        self._profiler: profiler.profile = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=10,
                warmup=20,
                active=active,
                repeat=1,
            ),
            on_trace_ready=_on_trace_ready_fn(self.rank),
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
        )

    def step(self) -> None:
        self._profiler.step()


@gin.configurable
class MetricsLogger:
    def __init__(
        self,
        multitask_configs: List[TaskConfig],
        batch_size: int,
        window_size: int,
        device: torch.device,
        rank: int,
        tensorboard_log_path: str = "",
        # Wandb configuration
        use_wandb: bool = False,
        wandb_project: str = "generative-recommenders",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        wandb_config: Optional[Dict] = None,
    ) -> None:
        self.multitask_configs: List[TaskConfig] = multitask_configs
        all_classification_tasks: List[str] = [
            task.task_name
            for task in self.multitask_configs
            if task.task_type != MultitaskTaskType.REGRESSION
        ]
        all_regression_tasks: List[str] = [
            task.task_name
            for task in self.multitask_configs
            if task.task_type == MultitaskTaskType.REGRESSION
        ]
        assert all_classification_tasks + all_regression_tasks == [
            task.task_name for task in multitask_configs
        ]
        self.task_names: List[str] = all_classification_tasks + all_regression_tasks

        self.class_metrics: Dict[str, List[RecMetricComputation]] = {
            "train": [],
            "eval": [],
        }
        if all_classification_tasks:
            for mode in ["train", "eval"]:
                self.class_metrics[mode].append(
                    NEMetricComputation(
                        my_rank=rank,
                        batch_size=batch_size,
                        n_tasks=len(all_classification_tasks),
                        window_size=window_size,
                    ).to(device)
                )
                self.class_metrics[mode].append(
                    AUCMetricComputation(
                        my_rank=rank,
                        batch_size=batch_size,
                        n_tasks=len(all_classification_tasks),
                        window_size=window_size,
                    ).to(device)
                )

        self.regression_metrics: Dict[str, List[RecMetricComputation]] = {
            "train": [],
            "eval": [],
        }
        if all_regression_tasks:
            for mode in ["train", "eval"]:
                self.regression_metrics[mode].append(
                    MSEMetricComputation(
                        my_rank=rank,
                        batch_size=batch_size,
                        n_tasks=len(all_regression_tasks),
                        window_size=window_size,
                    ).to(device)
                )
                self.regression_metrics[mode].append(
                    MAEMetricComputation(
                        my_rank=rank,
                        batch_size=batch_size,
                        n_tasks=len(all_regression_tasks),
                        window_size=window_size,
                    ).to(device)
                )

        self.global_step: Dict[str, int] = {"train": 0, "eval": 0}
        
        # Initialize TensorBoard logger
        self.tb_logger: Optional[SummaryWriter] = None
        if tensorboard_log_path != "":
            self.tb_logger = SummaryWriter(log_dir=tensorboard_log_path, purge_step=0)
            self.tb_logger.flush()
        
        # Initialize Wandb logger
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb and rank == 0:  # Only log from rank 0 to avoid duplicate logs
            if not WANDB_AVAILABLE:
                logger.warning("wandb is not available. Install wandb to enable logging.")
                self.use_wandb = False
            else:
                try:
                    # Prepare wandb config
                    config = {
                        "batch_size": batch_size,
                        "window_size": window_size,
                        "num_tasks": len(self.task_names),
                        "task_names": self.task_names,
                        "multitask_configs": [
                            {
                                "task_name": task.task_name,
                                "task_type": task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type),
                            }
                            for task in multitask_configs
                        ],
                    }
                    if wandb_config:
                        config.update(wandb_config)
                    
                    wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        name=wandb_run_name,
                        tags=wandb_tags,
                        config=config,
                        reinit=True,
                    )
                    logger.info(f"Initialized wandb logging for project: {wandb_project}")
                except Exception as e:
                    logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb logging.")
                    self.use_wandb = False
        else:
            if self.use_wandb and rank != 0:
                logger.info(f"Rank {rank}: wandb logging disabled for non-zero ranks to avoid duplicates")
            self.use_wandb = False

    @property
    def all_metrics(self) -> Dict[str, List[RecMetricComputation]]:
        return {
            "train": self.class_metrics["train"] + self.regression_metrics["train"],
            "eval": self.class_metrics["eval"] + self.regression_metrics["eval"],
        }

    def update(
        self,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        labels: torch.Tensor,
        mode: str = "train",
    ) -> None:
        for metric in self.all_metrics[mode]:
            metric.update(
                predictions=predictions,
                labels=labels,
                weights=weights,
            )
        self.global_step[mode] += 1

    def compute(self, mode: str = "train") -> Dict[str, float]:
        all_computed_metrics = {}

        for metric in self.all_metrics[mode]:
            computed_metrics = metric.compute()
            for computed in computed_metrics:
                all_values = computed.value.cpu()
                for i, task_name in enumerate(self.task_names):
                    key = f"metric/{str(computed.metric_prefix) + str(computed.name)}/{task_name}"
                    all_computed_metrics[key] = all_values[i]

        logger.info(
            f"{mode} - Step {self.global_step[mode]} metrics: {all_computed_metrics}"
        )
        return all_computed_metrics

    def compute_and_log(
        self,
        mode: str = "train",
        additional_logs: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, float]:
        all_computed_metrics = self.compute(mode)
        
        # Log to TensorBoard
        if self.tb_logger is not None:
            for k, v in all_computed_metrics.items():
                self.tb_logger.add_scalar(  # pyre-ignore [16]
                    f"{mode}_{k}",
                    v,
                    global_step=self.global_step[mode],
                )

            if additional_logs is not None:
                for tag, data in additional_logs.items():
                    for data_name, data_value in data.items():
                        self.tb_logger.add_scalar(
                            f"{tag}/{mode}_{data_name}",
                            data_value.detach().clone().cpu(),
                            global_step=self.global_step[mode],
                        )
        
        # Log to Wandb
        if self.use_wandb and wandb is not None:
            wandb_logs = {}
            
            # Log metrics with mode prefix
            for k, v in all_computed_metrics.items():
                wandb_logs[f"{mode}_{k}"] = v
            
            # Log additional logs
            if additional_logs is not None:
                for tag, data in additional_logs.items():
                    for data_name, data_value in data.items():
                        wandb_logs[f"{tag}/{mode}_{data_name}"] = data_value.detach().clone().cpu().item()
            
            # Add step information
            wandb_logs["step"] = self.global_step[mode]
            wandb_logs[f"{mode}_step"] = self.global_step[mode]
            
            try:
                wandb.log(wandb_logs, step=self.global_step[mode])
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
                
        return all_computed_metrics

    def reset(self, mode: str = "train"):
        for metric in self.all_metrics[mode]:
            metric.reset()
    
    def finish_wandb(self):
        """Finish the wandb run"""
        if self.use_wandb and wandb is not None:
            try:
                wandb.finish()
                logger.info("Finished wandb run")
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")


# the datasets we support
SUPPORTED_DATASETS = [
    "debug",
    "movielens-1m",
    "movielens-20m",
    "movielens-13b",
    "kuairand-1k",
]


def get_dataset(name: str, new_path_prefix: str = ""):
    assert name in SUPPORTED_DATASETS, f"dataset {name} not supported"
    if name == "debug":
        return DLRMv3RandomDataset, {}
    if name == "movielens-1m":
        return (
            DLRMv3MovieLensDataset,
            {
                "ratings_file": os.path.join(
                    new_path_prefix, "data/ml-1m/sasrec_format.csv"
                ),
            },
        )
    if name == "movielens-20m":
        return (
            DLRMv3MovieLensDataset,
            {
                "ratings_file": os.path.join(
                    new_path_prefix, "data/ml-20m/sasrec_format.csv"
                ),
            },
        )
    if name == "movielens-13b":
        return (
            DLRMv3SyntheticMovieLensDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/ml-13b/16x16384"
                ),
            },
        )
    if name == "kuairand-1k":
        return (
            DLRMv3KuaiRandDataset,
            {
                "seq_logs_file": os.path.join(
                    new_path_prefix, "data/KuaiRand-1K/data/processed_seqs.csv"
                ),
            },
        )
