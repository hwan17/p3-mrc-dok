import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from konlpy.tag import Mecab
# from rank_bm25 import BM25Okapi,BM25Plus,BM25L
from datasets import Dataset, load_from_disk
import torch
from contextlib import contextmanager

from transformers import AutoTokenizer,BertPreTrainedModel, BertModel

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        return pooled_output

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

class DenseRetrieval:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased") ##
        self.p_encoder = BertEncoder.from_pretrained('/opt/ml/code/0506/dpr/p_encoder/')
        self.q_encoder = BertEncoder.from_pretrained('/opt/ml/code/0506/dpr/q_encoder/')
#         valid_corpus = load_from_disk('/opt/ml/input/data/data/train_dataset')['validation']['context']
        with open("/opt/ml/input/data/data/wikipedia_documents.json", "r") as f:
            wiki = json.load(f)
            
        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) 
        self.ids = list(range(len(self.contexts)))
        
        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()
            
        self.p_embs = []
        
#         for p in valid_corpus:
        for p in tqdm(self.contexts):
            p = self.tokenizer(p, padding='max_length', truncation=True, return_tensors='pt').to('cuda')
            p_emb = self.p_encoder(**p).to('cpu').detach().numpy()
            self.p_embs.append(p_emb)
        
        self.p_embs = torch.Tensor(self.p_embs).squeeze()



    def get_relevant_doc_dpr(self, query, k=1):
        tokenzied_query = self.tokenizer(query, padding='max_length', truncation=True, return_tensors='pt').to('cuda')
        q_emb = self.q_encoder(**tokenzied_query).to('cpu')
        dot_prod_scores = torch.matml(q_emb, torch.transpose(self.p_embs,0,1))

        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        return dot_prod_scores.squeeze()[rank].tolist()[:k], rank.tolist()[:k]

    def get_relevant_doc_bulk_dpr(self, queries, k=1):
        tokenized_queries = []
        doc_scores = []
        doc_indices = []
        for query in tqdm(queries):
            tokenized_query = self.tokenizer(query, padding='max_length', truncation=True, return_tensors='pt').to('cuda')
            q_emb = self.q_encoder(**tokenized_query).to('cpu')
            dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.p_embs,0,1))
            rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

            doc_scores.append(dot_prod_scores.squeeze()[rank].tolist()[:k])
            doc_indices.append(rank.tolist()[:k])

        return doc_scores, doc_indices


    def retrieve(self, query_or_dataset, topk=1):
        print('topk',topk)

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_dpr(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(contexts[doc_indices[i]])
            return doc_scores, [contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            total = []
            doc_scores, doc_indices = self.get_relevant_doc_bulk_dpr(query_or_dataset['question'], k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc='Dense retrieval: ')):
                all_contexts = ''
                for i in range(topk):
                    all_contexts = all_contexts + ' '+ self.contexts[doc_indices[idx][i]]

                tmp = {
                    'question': example['question'],
                    'id': example['id'],
                    'context_id': doc_indices[idx][0], #retrieved id
                    'context': all_contexts # self.contexts[doc_indices[idx][0]] # retrieved documnent
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp['original_context'] = example['context']
                    tmp['answers'] = example['answers']
                total.append(tmp)

            cqas = pd.DataFrame(total)

            return cqas





