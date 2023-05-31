#!/usr/bin/env python
# coding=utf-8
""" Finetuning models on the MIMIC-III-50 dataset """
import csv
import logging
import os
import random
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional
from typing import List
import datasets
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score
from CLtrainer import MultilabelTrainer
from scipy.special import expit
from torch import nn
import glob
import shutil

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from HCBert import HierarchicalConvolutionalBert

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=4096,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_segments: List[int] = field(
        default_factory=lambda: [32, 16, 8],
        metadata={
            "help": "The maximum number of segments (paragraphs) to be considered. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seg_length: List[int] = field(
        default_factory=lambda: [128, 256, 512],
        metadata={
            "help": "The maximum segment (paragraph) length to be considered. Segments longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    task: Optional[str] = field(
        default='ecthr_a',
        metadata={
            "help": "Define downstream task"
        },
    )
    truncate_head: Optional[bool] = field(
    default=True,
    metadata={
        "help": "Whether to truncate tokens from the head (True) or tail (False) of the sequence."
    },
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    hierarchical: bool = field(
        default=True, metadata={"help": "Whether to use a hierarchical variant or not"}
    )
    convolutional: bool = field(
        default=False, metadata={"help": "Whether to use a hierarchical variant or not"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Fix boolean parameter
    if model_args.do_lower_case == 'False' or not model_args.do_lower_case:
        model_args.do_lower_case = False
    else:
        model_args.do_lower_case = True

    if model_args.hierarchical == 'False' or not model_args.hierarchical:
        model_args.hierarchical = False
    else:
        model_args.hierarchical = True

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(data_args.server_ip, data_args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)



    if training_args.do_train:
        train_dataset = load_dataset("json", data_files="/home/ghan/datasets/mimic_train50.json",name=data_args.task)
        train_dataset=train_dataset['train']

    if training_args.do_eval:
        eval_dataset = load_dataset("json", data_files="/home/ghan/datasets/mimic_dev50.json",name=data_args.task)
        eval_dataset=eval_dataset['train']
        #eval_dataset = eval_dataset.shuffle(seed=42).select(range(int(len(eval_dataset) * 0.1)))        
    if training_args.do_predict:
        predict_dataset = load_dataset("json", data_files="/home/ghan/datasets/mimic_test50.json",name=data_args.task)
        predict_dataset=predict_dataset['train']
        #predict_dataset = predict_dataset.shuffle(seed=42).select(range(int(len(predict_dataset) * 0.1)))
    # Labels
    with open("/home/ghan/caml-mimic-fixed-/mimicdata/mimic3/TOP_50_CODES.csv", 'r') as f:
        reader = csv.reader(f)
        label_list = []
        for row in reader:
            label_list.append(row[0])
    num_labels = len(label_list)
    #PROC labels
    with open("/home/ghan/caml-mimic-fixed-/mimicdata/mimic3/TOP_50_PROC_CODES.csv", 'r') as f:
        reader = csv.reader(f)
        proc_list = []
        for row in reader:
            proc_list.append(row[0])
    #DIAG labels
    with open("/home/ghan/caml-mimic-fixed-/mimicdata/mimic3/TOP_50_DIAG_CODES.csv", 'r') as f:
        reader = csv.reader(f)
        diag_list = []
        for row in reader:
            diag_list.append(row[0])
    

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=f"{data_args.task}",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

# 这里取bert 与源代码相比不再增加全连接层 转而从HierarchicalConvolutionalBert里增加全连接层
    segment_encoder = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if model_args.hierarchical:
        if config.model_type == 'bert':
                model = HierarchicalConvolutionalBert(encoder=segment_encoder,
                                             max_segments=data_args.max_segments,
                                             max_segment_length=data_args.max_seg_length,
                                             num_labels=num_labels)
        else:
                raise NotImplementedError(f"{config.model_type} is no supported yet!")

    #if training_args.do_predict and not training_args.do_eval and not training_args.do_train:
    #model.load_state_dict(torch.load('/home/ghan/lex-glue/logs/Convolutional/bert-base-uncased/seed_1/pytorch_model.bin'))
    #print('参数导入成功##########################################################################################################')

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):

        ehr_template_1 = [[0] * data_args.max_seg_length[0]]
        ehr_template_2 = [[0] * data_args.max_seg_length[1]]
        ehr_template_3 = [[0] * data_args.max_seg_length[2]]         
        batch = {'input_ids_1': [], 'attention_mask_1': [], 'token_type_ids_1': [],
                         'input_ids_2': [], 'attention_mask_2': [], 'token_type_ids_2': [],
                         'input_ids_3': [], 'attention_mask_3': [], 'token_type_ids_3': []}
        for ehr in examples['TEXT']:
            encoded_text = tokenizer(ehr, padding=padding,max_length=data_args.max_seq_length, truncation=True)
                    # cut text to 64 segments
            segments_ids = []
            for i in range(0, data_args.max_segments[0] * data_args.max_seg_length[0], data_args.max_seg_length[0]):
                segment_ids = encoded_text['input_ids'][i:i + data_args.max_seg_length[0]]
                if not segment_ids:
                    break
                segments_ids.append(segment_ids)
            decoded_segments = [tokenizer.decode(segment) for segment in segments_ids]
            ehr_encodings = tokenizer(decoded_segments[:data_args.max_segments[0]], padding=padding,
                                               max_length=data_args.max_seg_length[0], truncation=True)                    
            batch['input_ids_1'].append(ehr_encodings['input_ids'] + ehr_template_1 * (
                                                    data_args.max_segments[0] - len(ehr_encodings['input_ids'])))
            batch['attention_mask_1'].append(ehr_encodings['attention_mask'] + ehr_template_1 * (
                                                    data_args.max_segments[0] - len(ehr_encodings['attention_mask'])))
            batch['token_type_ids_1'].append(ehr_encodings['token_type_ids'] + ehr_template_1 * (
                                                    data_args.max_segments[0] - len(ehr_encodings['token_type_ids'])))   
                    
                    # cut text to 32 segments
            segments_ids = []
            for i in range(0, data_args.max_segments[1] * data_args.max_seg_length[1], data_args.max_seg_length[1]):
                segment_ids = encoded_text['input_ids'][i:i + data_args.max_seg_length[1]]
                if not segment_ids:
                    break
                segments_ids.append(segment_ids)
            decoded_segments = [tokenizer.decode(segment) for segment in segments_ids]
            ehr_encodings = tokenizer(decoded_segments[:data_args.max_segments[1]], padding=padding,
                                               max_length=data_args.max_seg_length[1], truncation=True)                    
            batch['input_ids_2'].append(ehr_encodings['input_ids'] + ehr_template_2 * (
                                                    data_args.max_segments[1] - len(ehr_encodings['input_ids'])))
            batch['attention_mask_2'].append(ehr_encodings['attention_mask'] + ehr_template_2 * (
                                                    data_args.max_segments[1] - len(ehr_encodings['attention_mask'])))
            batch['token_type_ids_2'].append(ehr_encodings['token_type_ids'] + ehr_template_2 * (
                                                    data_args.max_segments[1] - len(ehr_encodings['token_type_ids'])))  
                    # cut text to 16 segments
            segments_ids = []
            for i in range(0, data_args.max_segments[2] * data_args.max_seg_length[2], data_args.max_seg_length[2]):
                segment_ids = encoded_text['input_ids'][i:i + data_args.max_seg_length[2]]
                if not segment_ids:
                    break
                segments_ids.append(segment_ids)
            decoded_segments = [tokenizer.decode(segment) for segment in segments_ids]
            ehr_encodings = tokenizer(decoded_segments[:data_args.max_segments[2]], padding=padding,
                                               max_length=data_args.max_seg_length[2], truncation=True)                    
            batch['input_ids_3'].append(ehr_encodings['input_ids'] + ehr_template_3 * (
                                                    data_args.max_segments[2] - len(ehr_encodings['input_ids'])))
            batch['attention_mask_3'].append(ehr_encodings['attention_mask'] + ehr_template_3 * (
                                                    data_args.max_segments[2] - len(ehr_encodings['attention_mask'])))
            batch['token_type_ids_3'].append(ehr_encodings['token_type_ids'] + ehr_template_3 * (
                                                    data_args.max_segments[2] - len(ehr_encodings['token_type_ids'])))  


        batch["labels"] = [[1 if label in labels else 0 for label in label_list] for labels in (label_string.split(';') for label_string in examples["LABELS"])]
        return batch

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):

        y_true = p.label_ids
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        y_preds = (expit(logits) > 0.5).astype('int32')

        # Compute regular scores
        macro_f1 = f1_score(y_true=y_true, y_pred=y_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true=y_true, y_pred=y_preds, average='micro', zero_division=0)

        return {'macro-f1': macro_f1, 'micro-f1': micro_f1}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions=predictions[0]

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        y_preds = (expit(predictions) > 0.5).astype('int32')

        for group in [1, 2, 3,4]:
    # Get the indices of samples in the current group
            indices = [i for i, x in enumerate(predict_dataset['length_feature']) if x == group]
    # Compute the micro-F1 score for the current group
            micro_f1 = f1_score(y_true=labels[indices], y_pred=y_preds[indices], average='micro', zero_division=0)
    # Add the score to the metrics dictionary
            metrics[f'micro-f1_group_{group}'] = micro_f1        
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        output_labels_file = os.path.join(training_args.output_dir, "test_labels.csv")
        output_predict_file = os.path.join(training_args.output_dir, "test_predictions.csv")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as predict_writer, open(output_labels_file, "w") as labels_writer:
                predict_writer_csv = csv.writer(predict_writer, delimiter='\t')
                labels_writer_csv = csv.writer(labels_writer, delimiter='\t')
                for index, (pred_list, label_list) in enumerate(zip(predictions, labels)):
                    pred_line = [f'{pred:.5f}' for pred in pred_list]
                    label_line = [str(label) for label in label_list]

                    predict_writer_csv.writerow([index] + pred_line)
                    labels_writer_csv.writerow([index] + label_line)
    # Clean up checkpoints
    #checkpoints = [filepath for filepath in glob.glob(f'{training_args.output_dir}/*/') if '/checkpoint' in filepath]
    #for checkpoint in checkpoints:
    #    shutil.rmtree(checkpoint)


if __name__ == "__main__":
    main()
