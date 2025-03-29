import json
import logging
import os
import re
import sys
import time
from datetime import timedelta

import datasets
import easydict
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import (DistributedDataParallelKwargs,
                              InitProcessGroupKwargs, set_seed)
from accelerate.utils.memory import release_memory
from datasets import IterableDataset
from multiprocess import set_start_method
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import LengthGroupedSampler

from arguments import DataArguments, ModelArguments, TrainingArguments
from dac_wrapper.modeling_dac import DACModel
from data import (DataCollatorWithPaddingNAR, DynamicBatchSampler,
                  prepare_datasets)
from mask_gct_eval import wer
from maskgct_s2a import MaskGCT_S2A
from utils import (get_last_checkpoint, log_metric, log_pred_with_asr,
                   rotate_checkpoints)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.dtype == "float16":
        mixed_precision = "fp16"
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
    else:
        mixed_precision = "no"

    kwargs_handlers = [
        InitProcessGroupKwargs(timeout=timedelta(minutes=60)),
        DistributedDataParallelKwargs(find_unused_parameters=True),
    ]

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        kwargs_handlers=kwargs_handlers,
    )

    accelerator.init_trackers(
        project_name=data_args.wandb_project,
        config={
            "learning_rate": training_args.learning_rate,
            "num_train_epochs": training_args.num_train_epochs,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "global_batch_size": training_args.per_device_train_batch_size
            * accelerator.num_processes,
            "mixed_precision": mixed_precision,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "warmup_steps": training_args.warmup_steps,
            "weight_decay": training_args.weight_decay,
            "adam_beta1": training_args.adam_beta1,
            "adam_beta2": training_args.adam_beta2,
        },
        init_kwargs=(
            {
                "wandb": {
                    "name": data_args.wandb_run_name,
                    "dir": training_args.output_dir,
                }
            }
            if data_args.wandb_run_name
            else {}
        ),
    )

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARN)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    with open(os.path.join(data_args.data_config_path)) as j:
        data_config = json.loads(j.read())
        data_config = easydict.EasyDict(data_config)

    audio_encoder = DACModel.from_pretrained(data_config["codec_type"]).to(
        accelerator.device
    )
    sampling_rate = audio_encoder.config.sample_rate

    prompt_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=data_config["pretrained_tokenizer_repo_id"],
        cache_dir=model_args.cache_dir,
        trust_remote_code=data_args.trust_remote_code,
        use_fast=True,
        padding_side=data_config.get("padding_side", "left"),
    )

    train_dataset = prepare_datasets(
        data_config["training_files"],
        mode="train",
        cache_dir=data_config["dataset_cache_dir"],
        seed=training_args.seed + accelerator.process_index,
    )
    eval_dataset = prepare_datasets(
        data_config["validation_files"],
        mode="validation",
        cache_dir=data_config["dataset_cache_dir"],
        max_length=data_config["validation_length_per_dataset"],
    )
    test_dataset = prepare_datasets(
        data_config["test_files"],
        mode="test",
        cache_dir=data_config["dataset_cache_dir"],
        max_length=data_config["test_length_per_dataset"],
    )

    vectorized_datasets = {
        "train": train_dataset,
        "eval": eval_dataset,
        "test": test_dataset,
    }

    num_codebooks = audio_encoder.config.model["bottleneck"]["config"]["n_codebooks"]
    codebook_size = audio_encoder.config.model["bottleneck"]["config"]["codebook_size"]

    data_config.update(
        {
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
        }
    )

    pretrained_weight_repo_id = data_config.get("pretrained_weight_repo_id")
    if pretrained_weight_repo_id is not None and last_checkpoint is None:
        model = MaskGCT_S2A.from_pretrained(pretrained_weight_repo_id)
    else:
        model = MaskGCT_S2A(
            vocab_size=len(prompt_tokenizer),
            num_quantizer=num_codebooks,
            codebook_size=codebook_size,
        )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    test_tensor = torch.tensor([accelerator.process_index], device=accelerator.device)
    gathered_tensor = accelerator.gather(test_tensor)
    print("gathered_tensor", gathered_tensor)
    accelerator.wait_for_everyone()

    asr_model_name_or_path = model_args.asr_model_name_or_path

    def compute_metrics(audios, texts, device="cpu"):
        results = {}

        audios = [a.cpu().numpy() for a in audios]

        word_error, character_error, transcriptions = wer(
            asr_model_name_or_path=asr_model_name_or_path,
            texts=texts,
            audios=audios,
            device=device,
            sampling_rate=sampling_rate,
        )

        results["wer"] = word_error
        results["cer"] = character_error

        return results, audios

    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    if training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(vectorized_datasets["train"]) // (
            train_batch_size * gradient_accumulation_steps
        )
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        logger.info(
            "max_steps is given, it will override any value given in num_train_epochs"
        )
        total_train_steps = int(training_args.max_steps)
        num_epochs = sys.maxsize
        steps_per_epoch = total_train_steps

    if training_args.eval_steps is None:
        logger.info(f"eval_steps is not set, evaluating at the end of each epoch")
        eval_steps = steps_per_epoch
    else:
        eval_steps = training_args.eval_steps

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(total_train_steps)
        * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    train_data_collator = DataCollatorWithPaddingNAR(
        prompt_tokenizer=prompt_tokenizer, train=True, data_config=data_config
    )
    test_data_collator = DataCollatorWithPaddingNAR(
        prompt_tokenizer=prompt_tokenizer, train=False, data_config=data_config
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    logger.info("***** Running training *****")
    logger.info(
        f"  Num examples = {total_train_steps * train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(
        "  Instantaneous batch size per device =" f" {per_device_train_batch_size}"
    )
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps),
        desc="Train steps ... ",
        position=0,
        disable=not accelerator.is_local_main_process,
    )
    continue_training = True
    epochs_trained = 0
    cur_step = 0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    with accelerator.main_process_first():
        if accelerator.is_main_process:
            prompt_tokenizer.save_pretrained(training_args.output_dir)

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")

        steps_trained_progress_bar.update(cur_step)

        for epoch in range(0, epochs_trained):
            with accelerator.local_main_process_first():
                vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(
                    seed=training_args.seed
                )

        if training_args.max_steps < 0:
            resume_step = (
                cur_step - epochs_trained * steps_per_epoch
            ) * gradient_accumulation_steps
        else:
            resume_step = None
            with accelerator.local_main_process_first():
                vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(
                    seed=training_args.seed
                )
    else:
        resume_step = None

    def train_step(
        batch,
        accelerator,
    ):
        model.train()
        ce_loss = model(**batch)
        metrics = {"loss": ce_loss}
        return ce_loss, metrics

    def eval_step(
        batch,
        accelerator,
    ):
        eval_model = model if not training_args.torch_compile else model._orig_mod
        eval_model.eval()
        with torch.no_grad():
            ce_loss = eval_model(**batch)
        metrics = {"loss": ce_loss}
        return metrics

    def generate_step(batch, accelerator):
        texts = batch.pop("texts", None)
        inputs = {
            "text_prompt_input_ids": batch["prompt_input_ids"],
            "speech_prompt_input_ids": batch["speech_input_ids"][:, :, :],
            "speech_input_ids": batch["speech_input_ids"][:, :, :1],
        }
        eval_model = accelerator.unwrap_model(
            model, keep_fp32_wrapper=mixed_precision != "fp16"
        ).eval()
        if training_args.torch_compile:
            eval_model = model._orig_mod
        with torch.no_grad():
            generated_codes = eval_model.generate(**inputs)
            output_audios = audio_encoder.decode(generated_codes.transpose(1, 2))
            output_audios = output_audios.squeeze(1)
        output_audios = accelerator.pad_across_processes(
            output_audios, dim=1, pad_index=0
        )
        return output_audios, texts

    model.train()
    for epoch in range(epochs_trained, num_epochs):
        if training_args.group_by_length:
            with accelerator.local_main_process_first():
                vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(
                    seed=training_args.seed
                )
            sampler = LengthGroupedSampler(
                train_batch_size, lengths=vectorized_datasets["train"]["target_length"]
            )
            train_dataloader = DataLoader(
                vectorized_datasets["train"],
                collate_fn=train_data_collator,
                batch_size=per_device_train_batch_size,
                sampler=sampler,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory,
            )
        else:
            accelerator.even_batches = False
            batch_sampler = DynamicBatchSampler(
                lengths=vectorized_datasets["train"]["target_length"],
                frames_threshold=data_config.get(
                    "frames_threshold", per_device_train_batch_size * 2048
                ),
                max_samples=data_config.get("max_samples", per_device_train_batch_size),
                random_seed=training_args.seed + epoch,
                drop_last=False,
                max_token_length=data_config.get("max_token_length", float("inf")),
            )
            train_dataloader = DataLoader(
                vectorized_datasets["train"],
                collate_fn=train_data_collator,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory,
                batch_sampler=batch_sampler,
            )

        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(
            train_dataloader.dataset, IterableDataset
        ):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            train_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
            resume_step = None
            accelerator.wait_for_everyone()

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                loss, train_metric = train_step(batch, accelerator)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), training_args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                steps_trained_progress_bar.update(1)
                cur_step += 1

                if cur_step % training_args.logging_steps == 0:
                    steps_trained_progress_bar.write(
                        f"Step... ({cur_step} / {total_train_steps} | Loss:"
                        f" {train_metric['loss']}, Learning Rate:"
                        f" {lr_scheduler.get_last_lr()[0]})"
                    )
                    log_metric(
                        accelerator,
                        metrics=train_metric,
                        learning_rate=lr_scheduler.get_last_lr()[0],
                        train_time=train_time + time.time() - train_start,
                        step=cur_step,
                        epoch=epoch,
                        prefix="train",
                    )

                if (
                    cur_step % training_args.save_steps == 0
                ) or cur_step == total_train_steps:
                    intermediate_dir = os.path.join(
                        training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}"
                    )
                    accelerator.save_state(
                        output_dir=intermediate_dir, safe_serialization=False
                    )
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        rotate_checkpoints(
                            training_args.save_total_limit,
                            output_dir=training_args.output_dir,
                            logger=logger,
                        )

                        if cur_step == total_train_steps:
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(training_args.output_dir)

                    accelerator.wait_for_everyone()

                if training_args.do_eval and (
                    cur_step % eval_steps == 0 or cur_step == total_train_steps
                ):
                    train_time += time.time() - train_start
                    eval_metrics = []
                    eval_preds = []
                    eval_texts = []
                    eval_start = time.time()

                    batch = release_memory(batch)

                    validation_dataloader = DataLoader(
                        vectorized_datasets["eval"],
                        collate_fn=train_data_collator,
                        batch_size=per_device_eval_batch_size,
                        drop_last=False,
                        num_workers=training_args.dataloader_pin_memory,
                        pin_memory=training_args.dataloader_pin_memory,
                    )
                    validation_dataloader = accelerator.prepare(validation_dataloader)

                    for batch in tqdm(
                        validation_dataloader,
                        desc=f"Evaluating - Inference ...",
                        position=2,
                        disable=not accelerator.is_local_main_process,
                    ):
                        eval_metric = eval_step(batch, accelerator)
                        eval_metric = accelerator.gather_for_metrics(eval_metric)
                        eval_metrics.append(eval_metric)

                    if training_args.predict_with_generate:
                        validation_dataloader = DataLoader(
                            vectorized_datasets["test"],
                            collate_fn=test_data_collator,
                            batch_size=1,
                            drop_last=False,
                            num_workers=training_args.dataloader_pin_memory,
                            pin_memory=training_args.dataloader_pin_memory,
                        )
                        test_dataloader = accelerator.prepare(validation_dataloader)
                        for batch in tqdm(
                            test_dataloader,
                            desc=f"Evaluating - Generation ...",
                            position=2,
                            disable=not accelerator.is_local_main_process,
                        ):
                            generated_audios, texts = generate_step(batch, accelerator)
                            eval_preds.extend(generated_audios.to("cpu"))
                            eval_texts.extend(texts)

                    eval_time = time.time() - eval_start
                    eval_metrics = {
                        key: torch.mean(
                            torch.cat([d[key].unsqueeze(0) for d in eval_metrics])
                        )
                        for key in eval_metrics[0]
                    }

                    metrics_desc = ""
                    if training_args.predict_with_generate:
                        if accelerator.is_local_main_process:
                            gen_metric_results, audios = compute_metrics(
                                eval_preds, eval_texts, accelerator.device
                            )
                            eval_metrics.update(gen_metric_results)
                            metrics_desc = " ".join(
                                [
                                    f"Eval {key}: {value} |"
                                    for key, value in gen_metric_results.items()
                                ]
                            )
                            if "wandb" in training_args.report_to:
                                log_pred_with_asr(
                                    accelerator=accelerator,
                                    texts=eval_texts,
                                    # transcriptions=transcriptions,
                                    audios=audios,
                                    sampling_rate=sampling_rate,
                                    step=cur_step,
                                    prefix="eval",
                                )
                        accelerator.wait_for_everyone()

                    if accelerator.is_local_main_process:
                        steps_trained_progress_bar.write(
                            f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']} |"
                            f" {metrics_desc})"
                        )

                    log_metric(
                        accelerator,
                        metrics=eval_metrics,
                        train_time=eval_time,
                        step=cur_step,
                        epoch=epoch,
                        prefix="eval",
                    )

                    eval_metrics, eval_preds, batch, eval_metric = release_memory(
                        eval_metrics, eval_preds, batch, eval_metric
                    )
                    if training_args.predict_with_generate:
                        generated_audios = release_memory(generated_audios)

                    train_start = time.time()

                if cur_step == total_train_steps:
                    continue_training = False
                    break

        if not continue_training:
            break

    accelerator.end_training()


if __name__ == "__main__":
    set_start_method("spawn")
    main()
