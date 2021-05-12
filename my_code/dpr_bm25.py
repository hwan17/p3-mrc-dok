import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Value, Features, Dataset, load_from_disk, concatenate_datasets, DatasetDict

from tqdm.auto import tqdm
import pandas as pd
import pickle
import json
import os
import numpy as np
import torch

from konlpy.tag import Mecab

import time
from contextlib import contextmanager
from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi,BM25Plus,BM25L

#from utils_qa import *

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

class RetrieveBM25:
    def __init__(self, context_path):
        with open(context_path, "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()]))
        self.ids = list(range(len(self.contexts)))

        self.mecab = Mecab()
        tokenized_corpus = []
        for context in self.contexts:
            tokenized_context = self.mecab.morphs(context)
            tokenized_corpus.append(tokenized_context)
        self.bm25 = BM25Plus(tokenized_corpus, k1=1, b=0.4, delta=0.4) #self.bm25 = BM25Okapi(tokenized_corpus)

    def get_bm25_score(self, queries):
        doc_scores = torch.Tensor()
        for query in tqdm(queries):
            tokenized_query= self.mecab.morphs(query)
            doc_score = self.bm25.get_scores(tokenized_query)
            doc_scores = torch.cat((doc_scores,torch.from_numpy(doc_score).unsqueeze(0)), dim=0)
        return doc_scores