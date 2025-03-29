import warnings

warnings.filterwarnings(action="ignore")

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import datasets
import torch
from torch.utils.data import Sampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer


@dataclass
class DataCollatorWithPadding:
    prompt_tokenizer: AutoTokenizer
    train: bool
    data_config: Dict[str, Any]
    linguistic_type: str = "text"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        texts = [feature[self.linguistic_type] for feature in features]

        if self.train:
            input_texts = texts
        else:
            input_texts = [f"{text} {text}" for text in texts]

        prompt_inputs = self.prompt_tokenizer(
            input_texts, return_tensors="pt", padding=True
        )

        speech_input_ids = [
            torch.tensor(feature["codes"])[0, :] for feature in features
        ]
        speech_lengths = [codes.shape[-1] for codes in speech_input_ids]
        speech_attention_mask = torch.stack(
            [torch.arange(max(speech_lengths)) < length for length in speech_lengths]
        ).long()
        speech_input_ids = torch.nn.utils.rnn.pad_sequence(
            speech_input_ids,
            batch_first=True,
            padding_value=self.data_config.codebook_size,
        )

        batch = {
            "speech_input_ids": speech_input_ids,
            "speech_attention_mask": speech_attention_mask,
            "prompt_input_ids": prompt_inputs.input_ids,
            "prompt_attention_mask": prompt_inputs.attention_mask,
        }

        if not self.train:
            texts = [feature["text"] for feature in features]
            batch.update({"texts": texts})

        return batch


@dataclass
class DataCollatorWithPaddingNAR:
    prompt_tokenizer: AutoTokenizer
    train: bool
    data_config: Dict[str, Any]

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # assert self.prompt_tokenizer.padding_side == "left"

        texts = [feature["text"] for feature in features]

        if self.train:
            input_texts = texts
        else:
            input_texts = [f"{text} {text}" for text in texts]

        prompt_inputs = self.prompt_tokenizer(
            input_texts, return_tensors="pt", padding=True
        )

        speech_input_ids = [
            torch.tensor(feature["codes"]).transpose(0, 1) for feature in features
        ]
        speech_lengths = [codes.shape[0] for codes in speech_input_ids]
        speech_attention_mask = torch.stack(
            [torch.arange(max(speech_lengths)) < length for length in speech_lengths]
        ).long()
        speech_input_ids = torch.nn.utils.rnn.pad_sequence(
            speech_input_ids,
            batch_first=True,
            padding_value=self.data_config.codebook_size,
        )

        batch = {
            "speech_input_ids": speech_input_ids,
            "speech_attention_mask": speech_attention_mask,
            "prompt_input_ids": prompt_inputs.input_ids,
            "prompt_attention_mask": prompt_inputs.attention_mask,
        }

        if not self.train:
            batch.update({"texts": texts})

        return batch


class DynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: List[int],
        frames_threshold: int,
        max_samples: int = 0,
        random_seed: Optional[int] = None,
        drop_last: bool = False,
        max_token_length: Union[int, float] = float("inf"),
    ):
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.max_token_length = max_token_length

        batches = []
        indices = [(i, length) for i, length in enumerate(lengths)]
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices,
            desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu",
        ):
            if frame_len > self.max_token_length:
                continue

            if batch_frames + frame_len <= self.frames_threshold and (
                max_samples == 0 or len(batch) < max_samples
            ):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        del indices

        random.seed(random_seed)
        random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def read_parquet_datasets(
    pattern: str,
    num_proc: Optional[int] = None,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    data_type: str = "parquet",
):
    dataset = datasets.load_dataset(
        data_type,
        data_files=pattern,
        num_proc=num_proc,
        streaming=streaming,
        cache_dir=cache_dir,
    )
    dataset = datasets.concatenate_datasets(list(dataset.values()))
    return dataset


def prepare_datasets(
    patterns: Union[List[str], str],
    mode: str,
    num_proc: int = min(int(os.cpu_count() * 0.4), 32),
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    seed: int = 42,
    max_length: Optional[int] = None,
    return_all: bool = False,
):
    if isinstance(patterns, str):
        patterns = [patterns]

    dataset = []
    for pattern in tqdm(patterns, total=len(patterns), desc=mode):
        print(pattern)
        if pattern.endswith(".parquet"):
            data_type = "parquet"
        elif pattern.endswith(".arrow"):
            data_type = "arrow"
        else:
            raise ValueError
        dataset_i = read_parquet_datasets(
            pattern,
            num_proc=num_proc,
            streaming=streaming,
            cache_dir=cache_dir,
            data_type=data_type,
            return_all=return_all,
        )
        if max_length is not None:
            dataset_i = dataset_i.select(range(min(max_length, dataset_i.num_rows)))
        dataset.append(dataset_i)
    dataset = datasets.concatenate_datasets(dataset)
    dataset = dataset.shuffle(seed=seed)
    return dataset
