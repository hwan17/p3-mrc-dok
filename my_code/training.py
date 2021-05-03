import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from process_data import *
from utils import *

class Trainers(object):
    def __init__(self, cfg, model, epoch=None, criterion = None, optimizer=None, scheduler=None, train_dataset=None, 
                valid_dataset = None, test_dataset=None, collate_fn = None):
        self.cfg = cfg
        self.epoch = epoch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn
        self.model = model

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self):
        train_dataset = DataLoader(self.train_dataset, collate_fn=self.collate_fn, batch_size=self.cfg.values.train_args.train_batch_size, shuffle = True)
        global_step = 0
        epoch_loss, epoch_acc = 0.0, 0.0
        self.model.train()
        with tqdm(train_dataset, total = len(train_dataset), unit = 'batch') as train_bar:
            for step, batch in enumerate(train_bar):
                inputs = {key : value.to(self.device) for key, value in batch.items()}
                self.optimizer.zero_grad()
                pred = self.model(**inputs)

                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = pred.start_logits.size(1)
                pred.start_logits.clamp_(0, ignored_index)
                pred.end_logits.clamp_(0, ignored_index)
                
                start_loss = self.criterion(pred.start_logits, inputs['start_positions'])
                end_loss = self.criterion(pred.end_logits, inputs['end_positions'])
                total_loss = (start_loss + end_loss) / 2
                total_loss.backward()

                epoch_loss += total_loss.item()
                if (step + 1) % self.cfg.values.train_args.gradient_accumulation_steps == 0:
                    global_step += 1
                    self.optimizer.step()
                    if self.cfg.values.train_args.scheduler_name != 'ReduceLROnPlateau':
                       self.scheduler.step()  # Update learning rate schedule
                
                # visualize wandb
                current_lr = get_lr(self.optimizer)
                wandb.log({"Train Loss": total_loss.item(), "Learning Rate": current_lr})
                
                # update progress bar
                train_bar.set_description(f'Training Epoch [{self.epoch + 1} / {self.cfg.values.train_args.num_epochs}]')
                train_bar.set_postfix(loss = total_loss.item(), current_lr = current_lr)

        return epoch_loss / global_step

    def evaluate(self, mode, examples, features, answer_column=None, required_col = None):
        self.model.eval()
        start, end = [], []
        if mode == 'train':
            valid_dataset = DataLoader(self.valid_dataset, collate_fn=self.collate_fn, batch_size=self.cfg.values.train_args.eval_batch_size, shuffle=False)
            with tqdm(valid_dataset, total = len(valid_dataset), unit = "Evaluating") as eval_bar:
                with torch.no_grad():
                    for batch in eval_bar:
                        inputs = {key : value.to(self.device) for key, value in batch.items() if key in required_col}
                        pred = self.model(**inputs)
                        start.extend(pred.start_logits.tolist())
                        end.extend(pred.end_logits.tolist())

                        # update progress bar
                        eval_bar.set_description(f'Evaluating [{self.epoch + 1} / {self.cfg.values.train_args.num_epochs}]')

            start, end = np.array(start), np.array(end)
            predictions = post_processing_function(examples, features, (start,end), self.cfg.values.post_process.n_best_size, 
                                                   self.cfg.values.post_process.max_answer_length,  mode = mode, 
                                                   answer_column_name=answer_column)
            metrics = compute_metrics(predictions)
            
            # visualize wandb
            wandb.log({"Eval exact match": metrics['exact_match'], "Eval F1": metrics['f1']})
            
            return metrics
        
        elif mode == 'inference':
            inference_dataset = DataLoader(self.test_dataset, collate_fn = self.collate_fn, batch_size=1, shuffle=False)
            with tqdm(inference_dataset, total = len(inference_dataset), unit = 'inference') as inference_bar:
                with torch.no_grad():
                    for batch in inference_bar:
                        inputs = {key : value.to(self.device) for key, value in batch.items() if key in required_col}
                        pred = self.model(**inputs)
                        start.extend(pred.start_logits.tolist())
                        end.extend(pred.end_logits.tolist())

                        # update progress bar
                        inference_bar.set_description(f'Inference')
            
            start, end = np.array(start), np.array(end)
            predictions = post_processing_function(examples, features, (start,end), self.cfg.values.post_process.n_best_size, 
                                                   self.cfg.values.post_process.max_answer_length, mode = mode,
                                                   output_dir=self.cfg.values.inference.output_dir)
            return predictions

