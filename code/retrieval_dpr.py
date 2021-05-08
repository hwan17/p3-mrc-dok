import json
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup,ElectraModel, ElectraPreTrainedModel
from datasets import Dataset
import time
from contextlib import contextmanager
import pandas as pd



@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs[1]

        return pooled_output

class ElectraEncoder(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraEncoder, self).__init__(config)

        self.electra = ElectraModel(config)
        self.init_weights()
      
    def forward(self, input_ids, 
              attention_mask=None, token_type_ids=None): 

        outputs = self.electra(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        return pooled_output
    

class DenseRetrieval:
    def __init__(self):
        self.initialize()
        
    def initialize(self):
        model_name = "bert-base-multilingual-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dump_path = "/opt/ml/input/data/data/wikipedia_documents.json"
        with open(dump_path, 'r') as f:
            wiki = json.load(f)

        self.wiki_list = []
        for wiki_key in wiki.keys():
            self.wiki_list.append(wiki[wiki_key]['text'])

        self.p_encoder = BertEncoder.from_pretrained(model_name)
        self.q_encoder = BertEncoder.from_pretrained(model_name)
        self.p_encoder.load_state_dict(torch.load("/opt/ml/input/dpr/p_encoder.pth"))
        self.q_encoder.load_state_dict(torch.load("/opt/ml/input/dpr/q_encoder.pth"))
        self.p_encoder.to('cuda')  
        self.q_encoder.to('cuda') 
        
        
        self.wiki_embs = []
        with torch.no_grad():
            self.p_encoder.eval()

            for p in tqdm(self.wiki_list):
                p = self.tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                p_emb = self.p_encoder(**p).to('cpu').numpy()
                self.wiki_embs.append(p_emb)

        self.wiki_embs = torch.Tensor(self.wiki_embs).squeeze()  # (num_passage, emb_dim)

        print("wiki embedding done", self.wiki_embs.size())


        
    def get_relevant_doc_bulk(self, queries, k=1):
      
        tokenized_queries = []
        doc_scores = []
        doc_indices = []
        for query in tqdm(queries):
            with torch.no_grad():
                self.q_encoder.eval()
                q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                q_emb = self.q_encoder(**q_seqs_val).to('cpu')  # (num_query, emb_dim)

            result = torch.matmul(q_emb, torch.transpose(self.wiki_embs, 0, 1))
            rank = torch.argsort(result, dim=1, descending=True).squeeze()

            score = []
            indice = []
            for i in range(k):
                score.append(result.squeeze()[rank[i]])
                indice.append(rank[i])
            
         
            doc_scores.append(score)
            doc_indices.append(indice)
            
        return doc_scores, doc_indices

    def retrieve(self, query_or_dataset, topk=1):
       if isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices =self.get_relevant_doc_bulk(query_or_dataset['question'], k=topk)
            
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                
                all_contexts =''
                for i in range(topk):
                    all_contexts =all_contexts + " "+ self.wiki_list[doc_indices[idx][i]] 
                
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    "context_id": doc_indices[idx][0],  # retrieved id
                    "context": all_contexts #self.wiki_list[doc_indices[idx][0]]
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)

            cqas = pd.DataFrame(total)
            
            return cqas



