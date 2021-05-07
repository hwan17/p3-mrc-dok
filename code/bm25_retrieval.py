import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi,BM25Plus,BM25L
from datasets import Dataset

from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')


class SparseRetrievalBM25:
    def __init__(self):
        self.initialize()
        
    def initialize(self):
        with open("/opt/ml/input/data/data/wikipedia_documents.json", "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로
        self.ids = list(range(len(self.contexts)))

        self.mecab = Mecab()
        
        tokenized_corpus = []
        for context in tqdm(self.contexts):
            tokenized_context = self.mecab.morphs(context)
            tokenized_corpus.append(tokenized_context)

        self.bm25 = BM25Plus(tokenized_corpus, k1=1, b=0.4, delta=0.4) #self.bm25 = BM25Okapi(tokenized_corpus)
        
        
    def get_relevant_doc_bm25(self, query, k=1):
        tokenized_query  = self.mecab.morphs(query)
        doc_score = self.bm25.get_scores(tokenized_query)
        sorted_result = np.argsort(doc_score.squeeze())[::-1]

        return doc_score.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k]

    def get_relevant_doc_bulk_bm25(self, queries, k=1):
        tokenized_queries = []
        doc_scores = []
        doc_indices = []
        for query in tqdm(queries):
            tokenized_query= self.mecab.morphs(query)
            doc_score = self.bm25.get_scores(tokenized_query)
            sorted_result = np.argsort(doc_score.squeeze())[::-1]
            doc_scores.append(doc_score.squeeze()[sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve(self, query_or_dataset, topk=1):
        print("topk",topk)

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_bm25(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(contexts[doc_indices[i]])
            return doc_scores, [contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices =self.get_relevant_doc_bulk_bm25(query_or_dataset['question'], k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
                all_contexts =''
                for i in range(topk):
                    all_contexts =all_contexts + " "+ self.contexts[doc_indices[idx][i]]

                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    "context_id": doc_indices[idx][0],  # retrieved id
                    "context": all_contexts #self.contexts[doc_indices[idx][0]]  # retrieved doument
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)

            cqas = pd.DataFrame(total)
            
            return cqas



