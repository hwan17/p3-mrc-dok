import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm.auto import tqdm
import pandas as pd
import pickle
import json
import os
import numpy as np

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from konlpy.tag import Mecab

import time
from contextlib import contextmanager
from my_bm25 import *

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

class BM25Retrieval:
    def __init__(self, tokenize_fn, data_path="/opt/ml/input/data/data", context_path="wikipedia_documents.json"):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        bm25_name = f"BM25.bin"
        bm25_path = os.path.join(self.data_path, bm25_name)

        if os.path.isfile(bm25_path):
            with open(bm25_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("BM25 pickle load.")
        else:
            print("Build BM25")
            self.bm25 = BM25(
                self.contexts,
                tokenizer=tokenize_fn,
                k1=1.5,
                b=0.75,
                delta=1,
                n_gram=(1,1),
                max_features=None,
                tf_limit=None,
                )
            with open(bm25_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("BM25 pickle saved.")

    def retrieve(self, query_or_dataset, topk=1):
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            topk_name = f"BM25_{topk}"
            topk_path = os.path.join(self.data_path, topk_name)
            if os.path.isfile(topk_path):
                with open(topk_path, "rb") as file:
                    cqas, doc_scores = pickle.load(file)
                print(f"BM25_{topk} pickle load.")
            else:
                # make retrieved result as dataframe
                print(f'retrieve_top{topk} start')
                total = []
                with timer("query exhaustive search"):
                    doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset['question'], k=topk)
                for idx, example in enumerate(tqdm(query_or_dataset, desc="BM25 retrieval: ")):
                    # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
                    
                    for i in range(topk):
                        tmp = {
                            "question": example["question"],
                            "id": example['id'],
                            "context_id": doc_indices[idx][i],  # retrieved id
                            "context": self.contexts[doc_indices[idx][i]],  # retrieved doument
                            "context_score": doc_scores[idx][:topk]
                        }
                        if 'context' in example.keys() and 'answers' in example.keys():
                            tmp["original_context"] = example['context']  # original document
                            tmp["answers"] = example['answers']           # original answer
                        total.append(tmp)

                cqas = pd.DataFrame(total)
                with open(topk_path, "wb") as file:
                    pickle.dump((cqas, doc_scores), file)
            return cqas, doc_scores

    def get_relevant_doc(self, query, k=1):
        with timer("query ex search"):
            result = self.bm25.get_scores(query)
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        sorted_result = np.argsort(result)[::-1][:k]
        return result[sorted_result].tolist()[:k], sorted_result.tolist()[:k]

    def get_relevant_doc_bulk(self, queries, k=1):
        with timer("query ex search"):
            result = self.bm25.get_scores(queries)
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

if __name__ == "__main__":
    # Test sparse
    org_dataset = load_from_disk("/opt/ml/input/data/data/train_dataset")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    ) # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*"*40, "query dataset", "*"*40)
    print(full_ds)

    ### Mecab 이 가장 높은 성능을 보였기에 mecab 으로 선택 했습니다 ###
    mecab = Mecab()
    def tokenize(text):
        # return text.split(" ")
        return mecab.morphs(text)

    # from transformers import AutoTokenizer
    #
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "bert-base-multilingual-cased",
    #     use_fast=True,
    # )
    ###############################################################

    wiki_path = "wikipedia_documents.json"
    retriever = BM25Retrieval(
        # tokenize_fn=tokenizer.tokenize,
        tokenize_fn=tokenize,
        data_path="/opt/ml/input/data/data",
        context_path=wiki_path)

    # test single query
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    with timer("single query by exhaustive search"):
        scores, indices = retriever.retrieve(query)

    # test bulk
    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve(full_ds)
        df['correct'] = df['original_context'] == df['context']
        print("correct retrieval result by exhaustive search", df['correct'].sum() / len(df))