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


_LANGS = ["c",
          "go",
          "java",
          "python",
          "ruby"]

# Example doc-id: "someprefix-python-123"
# This will break if the format changes
def get_lang(_doc_str):
    return _doc_str.split("-")[1]

class CVEFixesPreferenceRetrieval(MultilingualTask, AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    metadata = TaskMetadata(
        name="CVEFixesPreferenceRetrieval",
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
                    "python": {
                        "average_document_length": 862.842,
                        "average_query_length": 466.546,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "javascript": {
                        "average_document_length": 1415.632,
                        "average_query_length": 186.018,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "go": {
                        "average_document_length": 563.729,
                        "average_query_length": 125.213,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ruby": {
                        "average_document_length": 577.634,
                        "average_query_length": 313.818,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "java": {
                        "average_document_length": 420.287,
                        "average_query_length": 690.36,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "c": {
                        "average_document_length": 712.129,
                        "average_query_length": 162.119,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "cpp": {
                        "average_document_length": 712.129,
                        "average_query_length": 162.119,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "rust": {
                        "average_document_length": 712.129,
                        "average_query_length": 162.119,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "swift": {
                        "average_document_length": 712.129,
                        "average_query_length": 162.119,
                        "num_documents": 1000,
                        "num_queries": 1000,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "typescript": {
                        "average_document_length": 712.129,
                        "average_query_length": 162.119,
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

        # CVEFIXES_DIR = os.path.join(DATASET_DIR, 'CVEFixes')
        # corpus_cvefixes_file = os.path.join(CVEFIXES_DIR, 'corpus.jsonl')
        # qrels_cvefixes_file = os.path.join(CVEFIXES_DIR, 'qrels.jsonl')
        # query_cvefixes_file = os.path.join(CVEFIXES_DIR, 'query.jsonl')

        # corpus_cvefixes_lines = load_jsonl(corpus_cvefixes_file)
        # qrels_cvefixes_lines = load_jsonl(qrels_cvefixes_file)
        # query_cvefixes_lines = load_jsonl(query_cvefixes_file)
        
        dataset = load_dataset("CoQuIR/CVEFixes")
        qrels_cvefixes_lines = list(dataset['test'])

        corpus = load_dataset("CoQuIR/CVEFixes", "corpus")
        corpus_cvefixes_lines = list(corpus['corpus'])

        query = load_dataset("CoQuIR/CVEFixes", "query")
        query_cvefixes_lines = list(query['query'])

        # convert query_cvefixes_lines to dict
        query_cvefixes_dict = {_line['id']: _line for _line in query_cvefixes_lines}

        # queries[lang][split][query_id] = text
        self.queries = {}
        # corpus[lang][split][doc_id] = {"title": doc_title, "text": doc_text}
        self.corpus = {}
        self.relevant_docs = {}

        # insert corpus
        for _line in tqdm(corpus_cvefixes_lines):
            _lang = _line['lang']
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
        for _line in tqdm(qrels_cvefixes_lines):
            qid = _line['qid']
            pos_doc_ids = _line['pos-docids']

            for pos_doc_id in pos_doc_ids:
                _lang = get_lang(pos_doc_id)
                # insert query
                # insert relevant docs
                if qid not in self.queries[_lang][self._EVAL_SPLIT]:
                    self.queries[_lang][self._EVAL_SPLIT][qid] = str(query_cvefixes_dict[qid]['text'])
                
                if qid not in self.relevant_docs[_lang][self._EVAL_SPLIT]:
                    self.relevant_docs[_lang][self._EVAL_SPLIT][qid] = {}
                self.relevant_docs[_lang][self._EVAL_SPLIT][qid][pos_doc_id] = 1

        self.data_loaded = True
