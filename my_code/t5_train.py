import logging
import os
import sys
from datasets import load_metric, load_from_disk, Dataset, DatasetDict, load_dataset
from collections import defaultdict
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, Adafactor

from transformers import (
    DataCollatorForSeq2Seq,
    EvalPrediction,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
import wandb

from utils_qa import postprocess_qa_predictions, check_no_error, tokenize
from my_seq2seq_trainer import Seq2SeqTrainer
import torch
import nltk
import re

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

logger = logging.getLogger(__name__)

def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.model_name_or_path = 'KETI-AIR/ke-t5-large-newslike'
    data_args.preprocessing_num_workers = 12
    
    print(model_args)
    print(data_args)
    print(training_args)

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # datasets = load_from_disk(data_args.dataset_name)
    # datasets = load_dataset('squad_kor_v1')
    datasets = load_from_disk(data_args.dataset_name)

    # datasets = DatasetDict({'train':datasets['train'], 'validation':datasets2['validation']})
    
    total_dic = {}
    for train_or_valid, train_or_valid_data in datasets.items():
        dic = defaultdict(list)
        for i in range(len(train_or_valid_data)):
            for key, value in train_or_valid_data[i].items():
                if key == 'context':
                    value = re.sub(r'\\n+|날짜=[\d]+-[\d]+-[\d]+', ' ', value).strip()
                    value = re.sub(r'\([一-龥]+\)', '', value)
                elif key == 'answers':
                    text = re.sub(r'\([一-龥]+\)', '', value['text'][0])
                    value = {'answer_start':value['answer_start'], 'text':[text]}
                dic[key].append(value)
        total_dic[train_or_valid] = Dataset.from_dict(dic)
    datasets = DatasetDict(total_dic)

    print(datasets)
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        'KETI-AIR/ke-t5-large-newslike',
        cache_dir=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'KETI-AIR/ke-t5-large-newslike',
        use_fast=True,
        cache_dir=None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        'KETI-AIR/ke-t5-large-newslike',
        cache_dir=None,
        config=config,
    )

    model.load_state_dict(torch.load('/opt/ml/pytorch_model_t5_2.bin'))

    training_args.num_beams = model_args.num_beams
    training_args.max_source_length = data_args.max_source_length
    training_args.max_target_length = data_args.max_target_length

    training_args.num_train_epochs = 5
    training_args.learning_rate = 1e-4
    training_args.per_device_train_batch_size = 1
    training_args.per_device_eval_batch_size = 4
    training_args.gradient_accumulation_steps = 32
    training_args.dataloader_num_workers = 4
    # training_args.eval_steps = 10
    training_args.logging_steps = 100
    # training_args.lr_scheduler_type='constant'
    training_args.predict_with_generate=True,
    training_args.warmup_steps=4800

    # train or eval mrc model
    if training_args.do_train or training_args.do_eval:
        wandb.init(project='MRC', name = model_args.model_name_or_path, reinit = False)
        # wandb.watch(model)
        wandb.config.update(training_args)
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(data_args, training_args, model_args, datasets, tokenizer, model):
    # Training preprocessing
    def preprocess_function(examples):
        inputs = [f'question: {q}  context: {c} </s>' for q, c in zip(examples['question'], examples['context'])]
        targets = [f'{a["text"][0]} </s>' for a in examples['answers']]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=False, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=data_args.max_target_length, padding=False, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["example_id"] = []
        for i in range(len(model_inputs["labels"])):
            model_inputs["example_id"].append(examples["id"][i])
        return model_inputs
    column_names = datasets['train'].column_names


    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=False,
                )

    if training_args.do_eval:
        eval_dataset = datasets["validation"]
        eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=False,
                )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data collator.
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None
    )

    # Post-processing:
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    metric = load_metric("squad")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels is for rouge metric, not used for f1/em metric
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        formatted_predictions = [{"id": ex['id'], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"])]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]

        result = metric.compute(predictions=formatted_predictions, references=references)
        return result

    # optimizer = Adafactor(model.parameters(), lr=training_args.learning_rate, scale_parameter=False, relative_step=False)

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # optimizers = (optimizer, None),
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        # trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        # trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        # trainer.state.save_to_json(
            # os.path.join(training_args.output_dir, "trainer_state.json")
        # )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        print(metrics)
        metrics["eval_samples"] = len(eval_dataset)

        # trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
