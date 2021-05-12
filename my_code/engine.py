from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig, AdamW, get_scheduler, AutoModel
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from training_mrc import Trainers
from training_dpr import *
from utils import *
from loss import *
from retrieval import *
from process_data import *
from model import *
import multiprocessing
from functools import partial
import parmap

from importlib import import_module
import time
import wandb
import pandas as pd
import copy
import pickle

import json
import logging
import os
import sys
from datasets import load_metric, load_from_disk, load_dataset, concatenate_datasets

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, BertTokenizerFast
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
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

    ######### DPR 모델 정의
    if cfg.values.train.retrieval == 'dpr' or cfg.values.inference.retrieval == 'dpr' or cfg.values.inference.retrieval == 'bm25_dpr':
        model_name = cfg.values.dpr.model_name
        config_dpr = AutoConfig.from_pretrained(model_name)
        tokenizer_dpr = AutoTokenizer.from_pretrained(model_name)
        p_encoder = BertEncoder.from_pretrained(model_name)
        q_encoder = BertEncoder.from_pretrained(model_name)

    if args.mode == 'train' and cfg.values.train.retrieval == 'dpr':
        squad_datasets = load_dataset("squad_kor_v1")
        datasets = load_from_disk('/opt/ml/input/data/data/train_dataset')

        if os.path.isfile('/opt/ml/my_code/dpr_process_contexts/our_passages.txt') and os.path.isfile('/opt/ml/my_code/dpr_process_contexts/v1_passages.txt'):
            with open('/opt/ml/my_code/dpr_process_contexts/our_passages.txt', 'rb') as our_p:
                our_passages = pickle.load(our_p)
            with open('/opt/ml/my_code/dpr_process_contexts/v1_passages.txt', 'rb') as v1_p:
                v1_passages = pickle.load(v1_p)
        else:
            our_passages =  [i for i in range(datasets['train'].num_rows)]      
            our_passages = parmap.map(partial(get_train_contexts_multiprocessing, dataset = datasets['train']), 
                                      our_passages, pm_pbar = True, pm_processes = 4)
            our_passages = sum(our_passages, [])
            with open('/opt/ml/my_code/dpr_process_contexts/our_passages.txt', 'wb') as our_p:
                pickle.dump(our_passages, our_p)

            v1_passages =  [i for i in range(squad_datasets['train'].num_rows)]      
            v1_passages = parmap.map(partial(get_train_contexts_multiprocessing, dataset = squad_datasets['train']), 
                                      v1_passages, pm_pbar = True, pm_processes = 4)
            v1_passages = sum(v1_passages, [])
            with open('/opt/ml/my_code/dpr_process_contexts/v1_passages.txt', 'wb') as v1_p:
                pickle.dump(v1_passages, v1_p)

        q_seqs = tokenizer_dpr(datasets['train']['question']+squad_datasets['train']['question'], padding='max_length', truncation=True, return_tensors='pt')
        p_seqs = tokenizer_dpr(our_passages + v1_passages, padding='max_length', truncation=True, return_tensors='pt')
 
        with open(cfg.values.dataset.context_path, "r") as f:
            wiki = json.load(f)

        wiki_contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로

        val_passages, val_passage_indices = get_eval_contexts_indices(datasets['validation']['context']+wiki_contexts)

        ground_truth = datasets['validation']['context']
        val_q_seqs = tokenizer_dpr(datasets['validation']['question'], padding='max_length', truncation=True, return_tensors='pt')
        val_p_seqs = tokenizer_dpr(val_passages, padding='max_length', truncation=True, return_tensors='pt')

        # 6개의 input(벡터) 값들을 학습할 때 편리하게 access 하기 위해 붙인다
        train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
                                      q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])
        valid_p_seqs = TensorDataset(val_p_seqs['input_ids'], val_p_seqs['attention_mask'], val_p_seqs['token_type_ids'])
        valid_q_seqs = TensorDataset(val_q_seqs['input_ids'], val_q_seqs['attention_mask'], val_q_seqs['token_type_ids'])

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params' : [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.values.dpr.weight_decay},
            {'params' : [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay':0.0},
            {'params' : [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.values.dpr.weight_decay},
            {'params' : [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.values.dpr.lr, eps = cfg.values.dpr.adam_epsilon)
        scheduler = get_scheduler(
            cfg.values.dpr.scheduler, optimizer,
            num_warmup_steps = int((len(train_dataset)//cfg.values.dpr.batch_size)*cfg.values.dpr.warmup_epoch),
            num_training_steps = len(train_dataset) * cfg.values.dpr.num_epochs
            )
        
        best_acc = 0.0
        for epoch in range(cfg.values.dpr.num_epochs):
            start_time = time.time() # 시작 시간 기록
            dpr_trainer = DprTrainers(cfg, p_encoder, q_encoder, epoch, scheduler=scheduler, optimizer=optimizer, train_dataset = train_dataset)
            train_loss = dpr_trainer.train()
            evaluation = dpr_trainer.evaluate(mode = args.mode, q_seqs = valid_q_seqs, p_seqs = valid_p_seqs, topk = cfg.values.dpr.topk, 
                                              contexts = datasets['validation']['context']+wiki_contexts,  indices = val_passage_indices,
                                              ground_truth = ground_truth)                
            
            if cfg.values.train_args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(train_loss)

            end_time = time.time() # 종료 시간 기록
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

            print(f'Time Spent : {elapsed_mins}m {elapsed_secs}s')
            for key, value in evaluation.items():
                print(f'\tEvaluation | {key} : {value}')
                
            if best_acc < evaluation['acc']:
                best_acc = evaluation['acc']
                b_epoch = epoch + 1
                torch.save(p_encoder.state_dict(), cfg.values.dpr.model_save_dir + f'/Pencoder_{cfg.values.dpr.model_name}.pt')
                torch.save(q_encoder.state_dict(), cfg.values.dpr.model_save_dir + f'/Qencoder_{cfg.values.dpr.model_name}.pt')
                print('\tbetter model found! saving models')
        print()
        print('='*50 + f' DPR Training finished / best models found in epoch : {b_epoch} ' + '='*50)
    
    elif args.mode == 'inference' and (cfg.values.inference.retrieval =='dpr' or cfg.values.inference.retrieval == 'bm25_dpr'):
        p_encoder.load_state_dict(torch.load(cfg.values.dpr.model_save_dir + f'/Pencoder_{cfg.values.dpr.model_name}.pt'))
        q_encoder.load_state_dict(torch.load(cfg.values.dpr.model_save_dir + f'/Qencoder_{cfg.values.dpr.model_name}.pt'))

        datasets = load_from_disk(cfg.values.dataset.test_dataset)
        with open(cfg.values.dataset.context_path, "r") as f:
            wiki = json.load(f)
        contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로
        contexts, context_indices = get_eval_contexts_indices(contexts)

        q_seqs = tokenizer_dpr(datasets['validation']['question'], padding='max_length', truncation=True, return_tensors='pt')
        p_seqs = tokenizer_dpr(contexts, padding='max_length', truncation=True, return_tensors='pt')
        
        p_datasets = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'])
        q_datasets = TensorDataset(q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])

        dpr_trainer = DprTrainers(cfg, p_model = p_encoder, q_model = q_encoder)
        contexts = dpr_trainer.evaluate(mode = args.mode, q_seqs = q_datasets, p_seqs = p_datasets, topk = cfg.values.dpr.topk, contexts = contexts,
                                        indices = context_indices, querys = datasets['validation']['question'])



# ================================================================================================================================================= #



    ########## MRC 모델 정의
    if cfg.values.train.mrc == 'mrc' or cfg.values.inference.mrc == 'mrc':
        # Load pretrained model and tokenizer
        config = AutoConfig.from_pretrained(cfg.values.mrc.model_name)
        tokenizer = AutoTokenizer.from_pretrained(cfg.values.mrc.model_name, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(cfg.values.mrc.model_name, config=config)

        # Padding side determines if we do (question|context) or (context|question).
        pad_on_right = tokenizer.padding_side == "right"

    ####### dataset 정의
    if args.mode == 'train' and cfg.values.train.mrc == 'mrc':
        datasets = load_from_disk(cfg.values.dataset.train_dataset)
        kor_quad_v1 = load_dataset("squad_kor_v1")
        kor_quad = kor_quad_v1['train']

        ex_train_dataset = datasets["train"]
        ex_valid_dataset = datasets['validation']

        column_names = ex_train_dataset.column_names
        kor_column_names = kor_quad.column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        #### preprocess train/validation dataset
        kwargs_dict = {'tokenizer': tokenizer, 'max_length': cfg.values.mrc.max_seq_length, 
                            'stride':cfg.values.mrc.doc_stride, 'pad_on_right' : pad_on_right,
                            'pad_to_max_length':cfg.values.mrc.pad_to_max_length, 
                            'question_column_name': question_column_name, 'context_column_name': context_column_name}

        features_dataset = ex_train_dataset.map(prepare_train_features, batched=True, 
                                                num_proc=cfg.values.dataset.num_workers,
                                                writer_batch_size=0, 
                                                remove_columns=column_names, fn_kwargs = kwargs_dict)

        kor_features_dataset = kor_quad.map(prepare_train_features, batched=True, 
                                                num_proc=cfg.values.dataset.num_workers,
                                                writer_batch_size=0, 
                                                remove_columns=kor_column_names, fn_kwargs = kwargs_dict)
                                                
        fe_valid_dataset = ex_valid_dataset.map(prepare_validation_features, batched=True, 
                                                num_proc=cfg.values.dataset.num_workers, 
                                                writer_batch_size=0,
                                                remove_columns=column_names, fn_kwargs = kwargs_dict)
        features = copy.deepcopy(fe_valid_dataset)

        ##Look at the columns your tokenizer is returning. You might wanna limit it to only the required columns.
        columns_to_return = set(fe_valid_dataset.features.keys()).intersection(set(features_dataset.features.keys()))
        fe_valid_dataset.set_format(type='torch', columns=list(columns_to_return))

        features_dataset = features_dataset.map(features= kor_features_dataset.features)
        features_dataset = concatenate_datasets([features_dataset,kor_features_dataset])
    
    elif args.mode == 'inference' and cfg.values.inference.mrc == 'mrc':
        datasets = load_from_disk(cfg.values.dataset.test_dataset)
        
        ## geting data from DPR
        if cfg.values.inference.retrieval == 'dpr':
            datasets = retrieve_datasets_for_mrc(datasets, contexts)
        elif cfg.values.inference.retrieval == 'bm25' or cfg.values.inference.retrieval == 'tfidf':
            datasets = run_sparse_retrieval(datasets, cfg.values.dataset.context_path, type_ = cfg.values.inference.retrieval,
                                            topk = cfg.values.dpr.topk, tokenize = tokenize)

        # only for eval or predict
        datasets = datasets['validation']
        column_names = datasets.column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        kwargs_dict = {'tokenizer': tokenizer, 'max_length': cfg.values.mrc.max_seq_length, 
                        'stride':cfg.values.mrc.doc_stride, 'pad_on_right' : pad_on_right,
                        'pad_to_max_length':cfg.values.mrc.pad_to_max_length, 
                        'question_column_name': question_column_name, 'context_column_name': context_column_name}

        inference_datasets = datasets.map(prepare_validation_features, batched=True, 
                               num_proc=cfg.values.dataset.num_workers, 
                               remove_columns=column_names, fn_kwargs = kwargs_dict)
        features_dataset = copy.deepcopy(inference_datasets)

        required_col = [i for i in inference_datasets.features.keys() if i in ['attention_mask', 'input_ids', 'token_type_ids']]
        inference_datasets.set_format(type='torch', columns=required_col)

    if cfg.values.train.mrc =='mrc' or cfg.values.inference.mrc == 'mrc':
        ##### criterion 정의
        if cfg.values.mrc.criterion == 'label_smoothing':
            kwargs_criterion_dict = {'classes': cfg.values.mrc.max_seq_length, 'smoothing': cfg.values.mrc.smoothing}
        else:
            kwargs_criterion_dict = {}
        criterion = create_criterion(cfg.values.mrc.criterion, **kwargs_criterion_dict)

        ##### optimizer 정의
        optimizer_fct = getattr(import_module('transformers'), cfg.values.mrc.optimizer)
        optimizer = optimizer_fct(model.parameters(), lr=cfg.values.mrc.lr, eps=cfg.values.mrc.adam_epsilon)

        ##### scheduler 정의
        if cfg.values.mrc.scheduler == 'steplr':
            scheduler = StepLR(
                optimizer, 
                step_size = (len(features_dataset) // cfg.values.mrc.batch_size) * cfg.values.mrc.warmup_epoch, 
                gamma = cfg.values.mrc.gamma
                )

        elif cfg.values.mrc.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, 'max', factor = cfg.values.mrc.gamma, patience = 2, cooldown = 0)

        elif cfg.values.mrc.scheduler == 'CosineAnnealing' :
            scheduler = CosineAnnealingWarmupRestarts(optimizer,
                            first_cycle_steps=int((len(features_dataset) // cfg.values.mrc.batch_size) * cfg.values.mrc.first_cycle_epoch),
                            cycle_mult=1.0,
                            max_lr=cfg.values.mrc.lr,
                            min_lr=cfg.values.mrc.min_lr,
                            warmup_steps=int((len(features_dataset) // cfg.values.mrc.batch_size) * cfg.values.mrc.warmup_epoch),
                            gamma=cfg.values.mrc.gamma)
        else:
            scheduler = get_scheduler(
                cfg.values.mrc.scheduler, optimizer, 
                num_warmup_steps = int((len(features_dataset) // cfg.values.mrc.batch_size) * cfg.values.mrc.warmup_epoch),
                num_training_steps = len(features_dataset) * cfg.values.mrc.num_epochs
                )

        # Data collator
        # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data collator.
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of = None if cfg.values.mrc.pad_to_max_length else 8)

        if args.mode == 'train':
            best_stats = {'f1' : 0, 'exact_match' : 0, 'best_saved_epoch' : 1}
            for epoch in range(cfg.values.mrc.num_epochs):
                start_time = time.time() # 시작 시간 기록
                trainer = Trainers(cfg, model, epoch, criterion, optimizer, scheduler, train_dataset = features_dataset, 
                                    valid_dataset = fe_valid_dataset, collate_fn = data_collator)
                train_loss = trainer.train()
                metrics = trainer.evaluate(args.mode, examples = ex_valid_dataset, features = features,  
                                        required_col = columns_to_return, answer_column = answer_column_name)

                if cfg.values.mrc.scheduler == 'ReduceLROnPlateau':
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
                    torch.save(model.state_dict(), cfg.values.mrc.model_save_dir + f'/{cfg.values.mrc.save_name}.pt')
                    print('\tBetter model found!! saving the model')
            print()
            print('='*50 + f" MRC Model Last saved from Epoch : {best_stats['best_saved_epoch']} " + '='*50)
            print('='*50 + ' MRC Training finished ' + '='*50)

        elif args.mode == 'inference':
            ### load my model
            model.load_state_dict(torch.load(cfg.values.mrc.model_save_dir + f'/{cfg.values.mrc.save_name}.pt'))
            trainer = Trainers(cfg, model, test_dataset = inference_datasets, collate_fn = data_collator)
            inference = trainer.evaluate(args.mode, examples = datasets, features = features_dataset,  
                                    required_col = required_col)
            print()
            print('='*50 + ' Final Inference finished ' + '='*50)





