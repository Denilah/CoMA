import shutil
import torch
from tqdm import tqdm
from transformers import (
    TrainerCallback,
    Trainer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    PreTrainedModel,
    GenerationConfig
)
from typing import Callable, Union
import weakref
from torch.utils.data import DataLoader, SequentialSampler
import pudb


class GenerationAndSaveBestModelsCallback(TrainerCallback):
    def __init__(
        self,
        trainer: Trainer,
        metric_fn: Callable[
            [
                Union[torch.Tensor, torch.cuda.FloatTensor],
                Union[torch.Tensor, torch.cuda.FloatTensor],
            ],
            float,
        ],
        eval_dataset,
        save_dir,
        num_best_models=3,
        eval_steps: int = 100,
    ):
        super().__init__()
        self.trainer_ref = weakref.ref(trainer)  # 创建Trainer的弱引用
        ignored_columns = list(
            set(eval_dataset.column_names) - {'input_ids', 'labels'})
        eval_dataset = eval_dataset.remove_columns(ignored_columns)
        self.eval_dataloader = DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=trainer.args.eval_batch_size,
            collate_fn=trainer.data_collator,
            drop_last=trainer.args.dataloader_drop_last,
            num_workers=trainer.args.dataloader_num_workers,
            pin_memory=trainer.args.dataloader_pin_memory,
        )
        self.save_dir = save_dir
        self.num_best_models = num_best_models
        self.eval_steps = eval_steps
        self.best_eval_results = []
        self.metric_fn = metric_fn

        self.generation_config = GenerationConfig(
            temperature=0.001,
            top_k=30,
            top_p=0.85,
            do_sample=True,
            num_beams=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            repetition_penalty=1.2,
            max_new_tokens=1024,
            min_new_tokens=1,
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % self.eval_steps == 0:
            trainer = self.trainer_ref()  # 获取Trainer对象的引用
            if trainer is None:
                return
            if args.local_rank != -1:
                torch.distributed.barrier()  # 同步所有进程
            # TODO: 只实现了单个rank的eval， 还没实现多个rank并行eval
            if state.is_local_process_zero:
                model: PreTrainedModel = trainer.model
                model.eval()

                with torch.no_grad():
                    metric_value = 0
                    total_samples = 0
                    pudb.set_trace()
                    for batch in tqdm(self.eval_dataloader, desc='Customize Eval'):
                        input_ids = batch["input_ids"].to(model.device)
                        generated_ids = model.generate(
                            input_ids=input_ids, generation_config=self.generation_config)
                        # 从generated_ids计算得到的指标值
                        metric = self.metric_fn(generated_ids, batch["labels"])
                        metric_value += metric * input_ids.size(0)
                        total_samples += input_ids.size(0)
                    metric_value /= total_samples
                    if len(self.best_eval_results) < self.num_best_models:
                        self.best_eval_results.append(
                            (metric_value, state.global_step))
                        # TODO: lora需要额外处理
                        model.save_pretrained(
                            f"{self.save_dir}/best_model_{state.global_step}"
                        )
                    else:
                        self.best_eval_results.sort()
                        worst_result, worst_step = self.best_eval_results[0]
                        if metric_value > worst_result:
                            self.best_eval_results[0] = (
                                metric_value,
                                state.global_step,
                            )
                            # TODO: lora需要额外处理
                            model.save_pretrained(
                                f"{self.save_dir}/best_model_{state.global_step}"
                            )
                            shutil.rmtree(
                                f"{self.save_dir}/best_model_{worst_step}")
            if args.local_rank != -1:
                torch.distributed.barrier()  # 同步所有进程


def metric_fn(generated_ids, label_ids):
    # 这是一个简单的示例，实际的metric_fn应该根据你的具体任务和评估标准来定义
    return generated_ids.mean().item()
