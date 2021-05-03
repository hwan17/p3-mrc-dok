from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig, AdamW, get_scheduler, BertForQuestionAnswering
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from training import Trainers
from utils import *
from loss import *
from retrieval import *
from process_data import *

from importlib import import_module
import time
import wandb
import pandas as pd
import copy


import logging
import os
import sys
from datasets import load_metric, load_from_disk

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, BertTokenizerFast

from transformers import (
    AdamW,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

def engine(cfg, args):
    seed_everything(cfg.values.seed)
    
    ########## 모델 정의
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(cfg.values.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.values.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.values.model_name, config=config)

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    ####### dataset 정의
    if args.mode == 'train':
        datasets = load_from_disk(cfg.values.dataset.train_dataset)
        ex_train_dataset = datasets["train"]
        ex_valid_dataset = datasets['validation']

        if cfg.values.train_args.retrieval: 
            retriever = SparseRetrieval(tokenize_fn=tokenize, data_path=cfg.values.inference.data_path, context_path=cfg.values.inference.context_path)
            retriever.get_sparse_embedding()
            #retriever.build_faiss()

        column_names = ex_train_dataset.column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        #### preprocess train/validation dataset
        kwargs_dict = {'tokenizer': tokenizer, 'max_length': cfg.values.model_args.max_seq_length, 
                            'stride':cfg.values.model_args.doc_stride, 'pad_on_right' : pad_on_right,
                            'pad_to_max_length':cfg.values.model_args.pad_to_max_length, 
                            'question_column_name': question_column_name, 'context_column_name': context_column_name}

        features_dataset = ex_train_dataset.map(prepare_train_features, batched=True, 
                                                num_proc=cfg.values.dataset.preprocessing_num_workers, 
                                                remove_columns=column_names, fn_kwargs = kwargs_dict)

        fe_valid_dataset = ex_valid_dataset.map(prepare_validation_features, batched=True, 
                                                num_proc=cfg.values.dataset.preprocessing_num_workers, 
                                                remove_columns=column_names, fn_kwargs = kwargs_dict)
        features = copy.deepcopy(fe_valid_dataset)

        ##Look at the columns your tokenizer is returning. You might wanna limit it to only the required columns.
        columns_to_return = set(fe_valid_dataset.features.keys()).intersection(set(features_dataset.features.keys()))
        fe_valid_dataset.set_format(type='torch', columns=list(columns_to_return))
    
    elif args.mode == 'final_train':
        datasets = load_from_disk(cfg.values.dataset.train_dataset)
        final_train_dataset = datasets['validation']

        column_names = final_train_dataset.column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        #### preprocess train/validation dataset
        kwargs_dict = {'tokenizer': tokenizer, 'max_length': cfg.values.model_args.max_seq_length, 
                            'stride':cfg.values.model_args.doc_stride, 'pad_on_right' : pad_on_right,
                            'pad_to_max_length':cfg.values.model_args.pad_to_max_length, 
                            'question_column_name': question_column_name, 'context_column_name': context_column_name}

        features_dataset = final_train_dataset.map(prepare_train_features, batched=True, 
                                          num_proc=cfg.values.dataset.preprocessing_num_workers, 
                                          remove_columns=column_names, fn_kwargs = kwargs_dict)
    
    elif args.mode == 'inference':
        datasets = load_from_disk(cfg.values.dataset.test_dataset)
        # run passage retrieval if true
        if cfg.values.inference.retrieval:
            datasets = run_sparse_retrieval(datasets, tokenize, cfg.values.inference.data_path, cfg.values.inference.context_path)

        # only for eval or predict
        datasets = datasets['validation']
        column_names = datasets.column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        kwargs_dict = {'tokenizer': tokenizer, 'max_length': cfg.values.model_args.max_seq_length, 
                        'stride':cfg.values.model_args.doc_stride, 'pad_on_right' : pad_on_right,
                        'pad_to_max_length':cfg.values.model_args.pad_to_max_length, 
                        'question_column_name': question_column_name, 'context_column_name': context_column_name}

        inference_datasets = datasets.map(prepare_validation_features, batched=True, 
                               num_proc=cfg.values.dataset.preprocessing_num_workers, 
                               remove_columns=column_names, fn_kwargs = kwargs_dict)
        features_dataset = copy.deepcopy(inference_datasets)

        required_col = [i for i in inference_datasets.features.keys() if i in ['attention_mask', 'input_ids', 'token_type_ids']]
        inference_datasets.set_format(type='torch', columns=required_col)

    ##### criterion 정의
    if cfg.values.train_args.criterion == 'label_smoothing':
        kwargs_criterion_dict = {'classes': cfg.values.model_args.max_seq_length, 'smoothing': cfg.values.train_args.smoothing}
    else:
        kwargs_criterion_dict = {}
    criterion = create_criterion(cfg.values.train_args.criterion, **kwargs_criterion_dict)

    ##### optimizer 정의
    optimizer_fct = getattr(import_module('transformers'), cfg.values.train_args.optimizer)
    optimizer = optimizer_fct(model.parameters(), lr=cfg.values.train_args.lr, eps=cfg.values.train_args.adam_epsilon)

    ##### scheduler 정의
    if cfg.values.train_args.scheduler_name == 'steplr':
        scheduler = StepLR(
            optimizer, 
            step_size = (len(features_dataset) // cfg.values.train_args.train_batch_size) * cfg.values.train_args.warmup_epoch, 
            gamma = cfg.values.train_args.gamma
            )

    elif cfg.values.train_args.scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor = cfg.values.train_args.gamma, patience = 2, cooldown = 0)

    elif cfg.values.train_args.scheduler_name == 'CosineAnnealing' :
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                        first_cycle_steps=int((len(features_dataset) // cfg.values.train_args.train_batch_size) * cfg.values.train_args.first_cycle_epoch),
                        cycle_mult=1.0,
                        max_lr=cfg.values.train_args.lr,
                        min_lr=cfg.values.train_args.min_lr,
                        warmup_steps=int((len(features_dataset) // cfg.values.train_args.train_batch_size) * cfg.values.train_args.warmup_epoch),
                        gamma=cfg.values.train_args.gamma)
    else:
        scheduler = get_scheduler(
            cfg.values.train_args.scheduler_name, optimizer, 
            num_warmup_steps = int((len(features_dataset) // cfg.values.train_args.train_batch_size) * cfg.values.train_args.warmup_epoch),
            num_training_steps = len(features_dataset) * cfg.values.train_args.num_epochs
            )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data collator.
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of = None if cfg.values.model_args.pad_to_max_length else 8)

    if args.mode == 'train':
        best_stats = {'f1' : 0, 'exact_match' : 0, 'best_saved_epoch' : 1}
        for epoch in range(cfg.values.train_args.num_epochs):
            start_time = time.time() # 시작 시간 기록
            trainer = Trainers(cfg, model, epoch, criterion, optimizer, scheduler, train_dataset = features_dataset, 
                                valid_dataset = fe_valid_dataset, collate_fn = data_collator)
            train_loss = trainer.train()
            metrics = trainer.evaluate(args.mode, examples = ex_valid_dataset, features = features,  
                                       required_col = columns_to_return, answer_column = answer_column_name)

            if cfg.values.train_args.scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(metrics['exact_match'])

            end_time = time.time() # 종료 시간 기록
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

            print(f'Time Spent : {elapsed_mins}m {elapsed_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} ')
            print(f"\tEval EM: {metrics['exact_match']} | Eval F1: {metrics['f1']}%")

            if metrics['exact_match'] > best_stats['exact_match']:
                best_stats['exact_match'], best_stats['best_saved_epoch'] = metrics['exact_match'], epoch + 1
                torch.save(model.state_dict(), cfg.values.model_args.model_save_dir + f'{args.config}.pt')
                print('\tBetter model found!! saving the model')
        print()
        print('='*50 + f" Model Last saved from Epoch : {best_stats['best_saved_epoch']} " + '='*50)
        print('='*50 + ' Training finished ' + '='*50)

    elif args.mode == 'inference':
        ### load my model
        model.load_state_dict(torch.load(cfg.values.model_args.model_save_dir + f'{args.config}.pt'))
        trainer = Trainers(cfg, model, test_dataset = inference_datasets, collate_fn = data_collator)
        inference = trainer.evaluate(args.mode, examples = datasets, features = features_dataset,  
                                   required_col = required_col)
        print()
        print('='*50 + ' Prediction finished ' + '='*50)

    elif args.mode == 'final_train':
        for epoch in range(cfg.values.train_args.num_epochs):
            start_time = time.time() # 시작 시간 기록
            
            trainer = Trainers(cfg, model, epoch, criterion, optimizer, scheduler, 
                               train_dataset = dataset, collate_fn = data_collator)
            train_loss = trainer.train()

            end_time = time.time() # 종료 시간 기록
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

            print(f'Time Spent : {elapsed_mins}m {elapsed_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            current_lr = get_lr(optimizer)
        torch.save(model.state_dict(), f'/opt/ml/my_code/results/{args.config}.pt')
        print()
        print('='*50 + ' Final Training finished ' + '='*50)






