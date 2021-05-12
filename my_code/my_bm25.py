import math
import parmap
import numpy as np
from tqdm.auto import tqdm
from functools import partial
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time

class n_gram_BM25:
    def __init__(self, corpus, tokenizer=None, n_gram=(1,1), max_features=None, tf_limit=None):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer
        self.n_gram = n_gram
        self.max_features = max_features
        self.tf_limit = tf_limit

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        start, end = self.n_gram
        total_frequencies = defaultdict(int)
        
        for document in tqdm(corpus):
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            
            for n in range(start, end + 1):
                for idx in range(len(document) - n + 1):
                    word = ' '.join(document[idx:idx+n])
                    if word not in frequencies:
                        frequencies[word] = 0
                    frequencies[word] += 1
                    total_frequencies[word] += 1
                    
            self.doc_freqs.append(frequencies)
            
        if self.max_features is not None:
            freq = set(sorted(total_frequencies, key=lambda x:(-total_frequencies[x], -len(tokenize(x))))[:self.max_features])
            
            for doc in self.doc_freqs:
                for i in set(doc.keys()) - freq:
                    if i in doc:
                        doc.pop(i)
            
        if self.tf_limit is not None:
            freq = set(key for key in total_frequencies if total_frequencies[key] > self.tf_limit)
            
            for doc in self.doc_freqs:
                for i in set(doc.keys()) - freq:
                    if i in doc:
                        doc.pop(i)
        
        for frequencies in self.doc_freqs:
            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

        self.avgdl = num_doc / self.corpus_size
        return nd
    
    def _tokenize_corpus(self, corpus):
        print('tokenize start')
        tokenized_corpus = parmap.map(self.tokenizer, corpus, pm_pbar=True, pm_processes=cpu_count())
        print('tokenize finish')
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, queries):
        raise NotImplementedError()

    def get_top_n(self, queries, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(queries)
        
        if isinstance(queries, str):
            score = self.get_scores(queries)
            top_n = np.argsort(score)[::-1][:n]
            return [documents[i] for i in top_n]
        else:
            top_ns = np.flip(np.argsort(scores), axis=1)[:, :n]
            return [[documents[i] for i in top_n] for top_n in top_ns]

class BM25(n_gram_BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1, n_gram=(1,1), max_features=None, tf_limit=None):
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer, n_gram, max_features, tf_limit)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf
            
    def get_scores(self, queries):
        if isinstance(queries, str):
            score = np.zeros(self.corpus_size)
            doc_len = np.array(self.doc_len)
            queries = self.tokenizer(queries)
            
            for q in queries:
                q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
                score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
            return score
        else:
            score = np.zeros((len(queries), self.corpus_size))
            doc_len = np.array(self.doc_len)

            queries = self._tokenize_corpus2(queries)

            for idx, query in enumerate(tqdm(queries)):
                for q in query:
                    q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
                    score[idx] += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
            return score

    def _tokenize_corpus2(self, corpus):
        print('tokenize start')
        tokenized_corpus = parmap.map(self.tokenizer, corpus, pm_pbar=True, pm_processes=cpu_count())
        print('tokenize finish')
        return tokenized_corpus