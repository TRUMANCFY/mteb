from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata
import os
import json
from tqdm import tqdm
from datasets import load_dataset

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    return data

# no pandas and transformers because of the short corpus (shorter than 1000)
# update: let take them back and set top_k as 100
_LANGS = ["numpy", "pandas", "pytorch", "scipy", "seaborn", "sklearn", "tensorflow", "transformers"]

# Example doc-id: "someprefix-python-123"
# This will break if the format changes
def get_lang(_doc_str):
    return _doc_str.split("-")[1]

class DeprecatedCodePreferenceRetrieval(MultilingualTask, AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    metadata = TaskMetadata(
        name="DeprecatedCodePreferenceRetrieval",
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
        eval_langs={lang: [lang + "-Code"] for lang in _LANGS},
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
        # DATASET_DIR = os.getenv("COQUIR_DATASET_PATH", 'datasets/')

        # if not os.path.exists(DATASET_DIR):
        #     raise ValueError(f"Dataset directory {DATASET_DIR} does not exist. Please set the COQUIR_DATASET_PATH environment variable.")

        # DEPRECATEDCODE_DIR = os.path.join(DATASET_DIR, 'DeprecatedCode')

        dataset = load_dataset("CoQuIR/DepreAPI")
        qrels_deprecated_lines = list(dataset['test'])

        corpus = load_dataset("CoQuIR/DepreAPI", "corpus")
        corpus_deprecated_lines = list(corpus['corpus'])

        query = load_dataset("CoQuIR/DepreAPI", "query")
        query_deprecated_lines = list(query['query'])

        query_deprecated_dict = {_line['id']: _line for _line in query_deprecated_lines}


        self.queries = {self._EVAL_SPLIT: {}}
        self.corpus = {self._EVAL_SPLIT: {}}
        self.relevant_docs = {self._EVAL_SPLIT: {}}

        for _line in tqdm(corpus_deprecated_lines):
            _lang = get_lang(_line['id'])
            assert _lang in _LANGS, f"Language {_lang} not supported. Supported languages are: {_LANGS}"
            if _lang not in self.queries:
                self.queries[_lang] = {self._EVAL_SPLIT: {}}
                self.corpus[_lang] = {self._EVAL_SPLIT: {}}
                self.relevant_docs[_lang] = {self._EVAL_SPLIT: {}}
            
            _doc_id = _line['id']
            self.corpus[_lang][self._EVAL_SPLIT][_doc_id] = {
                "title": str(_line.get('title', '')),
                "text": str(_line['text'])
            }

        # insert queries and relevant docs
        for _line in tqdm(qrels_deprecated_lines):
            qid = _line['qid']
            pos_doc_ids = _line['pos-docids']

            for pos_doc_id in pos_doc_ids:
                _lang = get_lang(pos_doc_id)
                # insert query
                # insert relevant docs
                if qid not in self.queries[_lang][self._EVAL_SPLIT]:
                    self.queries[_lang][self._EVAL_SPLIT][qid] = str(query_deprecated_dict[qid]['text'])
            
                if qid not in self.relevant_docs[_lang][self._EVAL_SPLIT]:
                    self.relevant_docs[_lang][self._EVAL_SPLIT][qid] = {}
                self.relevant_docs[_lang][self._EVAL_SPLIT][qid][pos_doc_id] = 1
        
        self.data_loaded = True



        
        
        # for _lang in _LANGS:
        #     self.queries[_lang] = {self._EVAL_SPLIT: {}}
        #     self.corpus[_lang] = {self._EVAL_SPLIT: {}}
        #     self.relevant_docs[_lang] = {self._EVAL_SPLIT: {}}
            
        #     corpus_deprecated_file = os.path.join(DEPRECATEDCODE_DIR, f'corpus-{_lang}.jsonl')
        #     qrels_deprecated_file = os.path.join(DEPRECATEDCODE_DIR, f'qrels-{_lang}.jsonl')
        #     query_deprecated_file = os.path.join(DEPRECATEDCODE_DIR, f'query-{_lang}.jsonl')

        #     corpus_deprecated_lines = load_jsonl(corpus_deprecated_file)
        #     qrels_deprecated_lines = load_jsonl(qrels_deprecated_file)
        #     query_deprecated_lines = load_jsonl(query_deprecated_file)

        #     # convert query_bug_lines to dict
        #     query_deprecated_dict = {_line['query-id']: _line for _line in query_deprecated_lines}

        #     for _line in tqdm(corpus_deprecated_lines):
        #         _doc_id = _line['doc-id']
        #         self.corpus[_lang][self._EVAL_SPLIT][_doc_id] = {
        #             "title": str(_line.get('title', '')),
        #             "text": str(_line['text'])
        #         }

        #     # insert queries and relevant docs
        #     for _line in tqdm(qrels_deprecated_lines):
        #         qid = _line['qid']
        #         pos_doc_ids = _line['pos-docids']

        #         for pos_doc_id in pos_doc_ids:
        #             # insert query
        #             # insert relevant docs
        #             if qid not in self.queries[_lang][self._EVAL_SPLIT]:
        #                 self.queries[_lang][self._EVAL_SPLIT][qid] = str(query_deprecated_dict[qid]['text'])
                
        #             if qid not in self.relevant_docs[_lang][self._EVAL_SPLIT]:
        #                 self.relevant_docs[_lang][self._EVAL_SPLIT][qid] = {}
        #             self.relevant_docs[_lang][self._EVAL_SPLIT][qid][pos_doc_id] = 1

        # self.data_loaded = True
