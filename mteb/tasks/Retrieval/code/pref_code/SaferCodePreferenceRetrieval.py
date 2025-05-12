from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata
import os
import json
from tqdm import tqdm

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    return data

def get_lang(_doc_str):
    # Example doc-id: safecoder-python-train-new-9-pos0
    return _doc_str.split("-")[1]

_LANGS = ["c", "cpp", "python", "java", "javascript", "go", "ruby"]

class SaferCodePreferenceRetrieval(MultilingualTask, AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    metadata = TaskMetadata(
        name="SaferCodePreferenceRetrieval",
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
            "test": {
                "average_document_length": 862.842,
                "average_query_length": 466.546,
                "num_documents": 1000,
                "num_queries": 1000,
                "average_relevant_docs_per_query": 1.0,
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

        SAFECODE_DIR = os.path.join(DATASET_DIR, 'SafeCoder')
        corpus_safecode_file = os.path.join(SAFECODE_DIR, 'corpus.jsonl')
        qrels_safecode_file = os.path.join(SAFECODE_DIR, 'qrels.jsonl')
        query_safecode_file = os.path.join(SAFECODE_DIR, 'query.jsonl')

        corpus_safecode_lines = load_jsonl(corpus_safecode_file)
        qrels_safecode_lines = load_jsonl(qrels_safecode_file)
        query_safecode_lines = load_jsonl(query_safecode_file)

        # convert query_safecode_lines to dict
        query_safecode_dict = {_line['query-id']: _line for _line in query_safecode_lines}

        self.queries = {}
        self.corpus = {}
        self.relevant_docs = {}

        # insert corpus
        for _line in tqdm(corpus_safecode_lines):
            _lang = _line['lang']
            assert _lang in _LANGS, f"Language {_lang} not supported. Supported languages are: {_LANGS}"
            if _lang not in self.queries:
                self.queries[_lang] = {self._EVAL_SPLIT: {}}
                self.corpus[_lang] = {self._EVAL_SPLIT: {}}
                self.relevant_docs[_lang] = {self._EVAL_SPLIT: {}}
            
            _doc_id = _line['doc-id']
            self.corpus[_lang][self._EVAL_SPLIT][_doc_id] = {
                "title": _line.get('title', ''),
                "text": _line['text']
            }
        
        # insert queries and relevant docs
        for _line in tqdm(qrels_safecode_lines):
            qid = _line['qid']
            pos_doc_ids = _line['pos-docids']

            for pos_doc_id in pos_doc_ids:
                _lang = get_lang(pos_doc_id)
                assert _lang in _LANGS, f"Language {_lang} not supported. Supported languages are: {_LANGS}"
                
                if qid not in self.queries[_lang][self._EVAL_SPLIT]:
                    self.queries[_lang][self._EVAL_SPLIT][qid] = str(query_safecode_dict[qid]['text'])
                
                if qid not in self.relevant_docs[_lang][self._EVAL_SPLIT]:
                    self.relevant_docs[_lang][self._EVAL_SPLIT][qid] = {}
                self.relevant_docs[_lang][self._EVAL_SPLIT][qid][pos_doc_id] = 1

        self.data_loaded = True
