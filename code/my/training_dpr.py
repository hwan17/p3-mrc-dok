import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import wandb
from process_data import *
from utils import *
import torch.nn.functional as F
from collections import defaultdict

from dpr_bm25 import *

class DprTrainers(object):
    def __init__(self, cfg, p_model, q_model, epoch=None, criterion = None, optimizer=None, scheduler=None, train_dataset=None):
        self.cfg = cfg
        self.epoch = epoch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.p_model = p_model
        self.q_model = q_model

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.p_model.to(self.device)
        self.q_model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.cfg.values.dpr.batch_size)

        global_step, epoch_loss = 0, 0.0
        self.p_model.train()
        self.q_model.train()
        with tqdm(train_dataloader, total = len(train_dataloader), unit = 'batch') as train_bar:
            for step, batch in enumerate(train_bar):
                batch = tuple(t.to(self.device) for t in batch)
                self.optimizer.zero_grad()

                p_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]
                        }

                q_inputs = {'input_ids': batch[3],
                            'attention_mask': batch[4],
                            'token_type_ids': batch[5]
                            }

                p_outputs = self.p_model(**p_inputs)
                q_outputs = self.q_model(**q_inputs)

                # 각 question에 대응되는 passage 존재 (batch_size개)
                # in_batch negative : question과 대응되는 score는 높이고, 대응되지 않는 passage의 similarity score는 최소화 시키는

                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                # target : position of positive samples : diagonal 즉 대응되는 쌍 학습
                targets = torch.arange(0, sim_scores.shape[0]).long()
                targets = targets.to(self.device)

                sim_scores = F.log_softmax(sim_scores, dim=1)
                loss = F.nll_loss(sim_scores, targets)
                loss.backward()

                epoch_loss += loss.item()
                if (step + 1) % self.cfg.values.dpr.gradient_accumulation_steps == 0:
                    global_step += 1
                    self.optimizer.step()
                    if self.cfg.values.dpr.scheduler != 'ReduceLROnPlateau':
                       self.scheduler.step()  # Update learning rate schedule
                
                # # visualize wandb
                current_lr = get_lr(self.optimizer)
                # wandb.log({"Train Loss": total_loss.item(), "Learning Rate": current_lr})
                
                # update progress bar
                train_bar.set_description(f'DPR Training Epoch [{self.epoch + 1} / {self.cfg.values.dpr.num_epochs}]')
                train_bar.set_postfix(loss = loss.item(), current_lr = current_lr)

        return epoch_loss / global_step

    def evaluate(self, mode, q_seqs=None, p_seqs=None, topk = 1, contexts = None, indices = None, ground_truth = None, querys = None):
        self.p_model.eval()
        self.q_model.eval()
        p_dataloader = DataLoader(p_seqs, batch_size=self.cfg.values.dpr.batch_size, shuffle = False)
        q_dataloader = DataLoader(q_seqs, batch_size=self.cfg.values.dpr.batch_size, shuffle = False)
        querys = DataLoader(querys, batch_size = self.cfg.values.dpr.batch_size, shuffle = False)

        p_embs, states = torch.Tensor().to(self.device), defaultdict(int)
        states['idx'], topk_contexts = 0, []

        if self.cfg.values.reference.retrieval == 'bm25_dpr':
            p_bm25 = RetrieveBM25(context_path=self.cfg.values.dataset.context_path)

        with tqdm(p_dataloader, total = len(p_dataloader), unit = ' passage embedding ') as passage_bar:
            with torch.no_grad():
                for batch in passage_bar:
                    batch = tuple(t.to(self.device) for t in batch)
                    p_inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'token_type_ids': batch[2]
                                }
                    p_emb = self.p_model(**p_inputs)
                    p_embs = torch.cat((p_embs, p_emb), dim=0)

                with tqdm(zip(q_dataloader,querys), total = len(q_dataloader), unit = ' query embedding ') as query_emb:
                    for batch, query in query_emb:
                        batch = tuple(t.to(self.device) for t in batch)
                        q_inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2]
                                }
                        q_emb = self.q_model(**q_inputs)
                        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))

                        if self.cfg.values.reference.retrieval == 'bm25_dpr':
                            bm25_score = p_bm25.get_bm25_score(queries = query)
                            dot_prod_scores = dot_prod_scores*(1.1) + bm25_score.to(self.device)
                            
                        ranks = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
                        
                        for rank in ranks[:,:topk]:
                            if mode == 'train':
                                for k in range(topk):
                                    if ground_truth[states['idx']] == contexts[indices[rank[k]]]:
                                        states[f'# of correct passage from rank {k+1} '] += 1
                                states['idx'] += 1
                            
                            elif mode == 'inference':
                                concate_contexts = ''
                                for k in range(topk):
                                    concate_contexts += contexts[indices[rank[k]]] + ' '
                                topk_contexts.append(concate_contexts)
                                    
        if mode == 'train':
            states['total correct'] = sum([value for key, value in states.items() if key != 'idx'])
            states['acc'] = (states['total correct'] / states['idx']) * 100
            return states
        elif mode == 'inference':
            return topk_contexts




