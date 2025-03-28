import os
import re
import shutil
from dataclasses import field
from pathlib import Path
from typing import Dict, List

import torch
from wandb import Audio


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)-epoch-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _RE_CHECKPOINT.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder,
        max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])),
    )


def sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint") -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [
        str(x)
        for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")
        if os.path.isdir(x)
    ]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(
    save_total_limit=None, output_dir=None, checkpoint_prefix="checkpoint", logger=None
) -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(
        output_dir=output_dir, checkpoint_prefix=checkpoint_prefix
    )
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
        )
        shutil.rmtree(checkpoint, ignore_errors=True)


def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    step: int,
    epoch: int,
    learning_rate: float = None,
    prefix: str = "train",
):
    """Helper function to log all training/evaluation metrics with the correct prefixes and styling."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    log_metrics[f"{prefix}/epoch"] = epoch
    if learning_rate is not None:
        log_metrics[f"{prefix}/learning_rate"] = learning_rate
    accelerator.log(log_metrics, step=step)


def log_pred(
    accelerator,
    audios: List[torch.Tensor],
    sampling_rate: int,
    step: int,
    prefix: str = "eval",
    num_lines: int = 200000,
):
    """Helper function to log target/predicted transcriptions to weights and biases (wandb)."""
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        # pretty name for current step: step 50000 -> step 50k
        cur_step_pretty = f"{int(step // 1000)}k" if step > 1000 else step
        prefix_pretty = prefix.replace("/", "-")

        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"predictions/{prefix_pretty}-step-{cur_step_pretty}",
            columns=["Target descriptions"],
            step=step,
            commit=False,
        )

        # wandb can only loads 100 audios per step
        wandb_tracker.log(
            {
                "Speech samples": [
                    Audio(
                        audio,
                        sample_rate=sampling_rate,
                    )
                    for (i, audio) in enumerate(audios[: min(len(audios), 100)])
                ]
            },
            step=step,
        )


def log_pred_with_asr(
    accelerator,
    texts: List[str],
    # transcriptions: List[str],
    audios: List[torch.Tensor],
    sampling_rate: int,
    step: int,
    prefix: str = "eval",
    num_lines: int = 200000,
):
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        # pretty name for current step: step 50000 -> step 50k
        cur_step_pretty = f"{int(step // 1000)}k" if step > 1000 else step
        prefix_pretty = prefix.replace("/", "-")

        # convert str data to a wandb compatible format
        str_data = [
            # [texts[i], transcriptions[i]] for i in range(len(transcriptions))
            [texts[i]]
            for i in range(len(texts))
        ]

        columns = ["Target prompts"]

        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"predictions/{prefix_pretty}-step-{cur_step_pretty}",
            columns=columns,
            data=str_data[:num_lines],
            step=step,
            commit=False,
        )

        # wandb can only loads 100 audios per step
        wandb_tracker.log(
            {
                "Speech samples": [
                    Audio(
                        audio,
                        caption=f"{texts[i]}",
                        sample_rate=sampling_rate,
                    )
                    for (i, audio) in enumerate(audios[: min(len(audios), 100)])
                ]
            },
            step=step,
        )
