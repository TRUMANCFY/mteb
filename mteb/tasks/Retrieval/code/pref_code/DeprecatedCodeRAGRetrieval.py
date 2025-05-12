from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata
import os
import json
from tqdm import tqdm
import pandas as pd

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    return data

# Example doc-id: "someprefix-python-123"
# This will break if the format changes
def get_lang(_doc_str):
    return _doc_str.split("-")[1]

class DeprecatedCodeRAGRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    metadata = TaskMetadata(
        name="DeprecatedCodeRAGRetrieval",
        description="The dataset is derived from CodeNet to benchmark retrievers' performance on cvefixed.",
        reference="https://huggingface.co/datasets/code_search_net/",
        dataset={
            "path": "code-search-net/code_search_net",
            "revision": "fdc6a9e39575768c27eb8a2a5f702bf846eb4759",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="@article{husain2019codesearchnet, title={{CodeSearchNet} challenge: Evaluating the state of semantic code search}, author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc}, journal={arXiv preprint arXiv:1909.09436}, year={2019} }",
        descriptive_stats={
            "n_samples": {
                _EVAL_SPLIT: 3979,
            },
            "avg_character_length": {
                "test": {
                    "numpy": {
                        "average_document_length": 862.842,
                        "average_query_length": 466.546,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "pandas": {
                        "average_document_length": 862.842,
                        "average_query_length": 466.546,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "pytorch": {
                        "average_document_length": 862.842,
                        "average_query_length": 466.546,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "scipy": {
                        "average_document_length": 862.842,
                        "average_query_length": 466.546,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "seaborn": {
                        "average_document_length": 862.842,
                        "average_query_length": 466.546,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "tensorflow": {
                        "average_document_length": 862.842,
                        "average_query_length": 466.546,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "transformers": {
                        "average_document_length": 862.842,
                        "average_query_length": 466.546,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                },
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        # read the dataset
        DATASET_DIR = os.getenv("COQUIR_DATASET_PATH", 'datasets/')

        if not os.path.exists(DATASET_DIR):
            raise ValueError(f"Dataset directory {DATASET_DIR} does not exist. Please set the COQUIR_DATASET_PATH environment variable.")

        DEPRECATEDCODE_DIR = os.path.join(DATASET_DIR, 'deprecated')

        query_deprecated_file = os.path.join(DEPRECATEDCODE_DIR, 'deprecated_sampled.xlsx')
        corpus_deprecated_file = os.path.join(DEPRECATEDCODE_DIR, 'corpus.jsonl')

        query_df = pd.read_excel(query_deprecated_file)
        query_deprecated_lines = query_df.to_dict(orient='records')
        corpus_deprecated_lines = load_jsonl(corpus_deprecated_file)

        self.queries = {self._EVAL_SPLIT: {}}
        self.corpus = {self._EVAL_SPLIT: {}}
        self.relevant_docs = {self._EVAL_SPLIT: {}}    

        # filter queries based on languages
        for _line in tqdm(corpus_deprecated_lines):
            _doc_id = _line['doc-id']
            self.corpus[self._EVAL_SPLIT][_doc_id] = {
                "title": str(_line.get('title', '')),
                "text": str(_line['text'])
            }
            
        # insert queries and relevant docs
        for _query_idx, _line in tqdm(enumerate(query_deprecated_lines)):
            qid = "query-" + str(_query_idx)
            # insert relevant docs
            if qid not in self.queries[self._EVAL_SPLIT]:
                self.queries[self._EVAL_SPLIT][qid] = _line['queries']
            
            if qid not in self.relevant_docs[self._EVAL_SPLIT]:
                self.relevant_docs[self._EVAL_SPLIT][qid] = {}
            
        self.data_loaded = True
