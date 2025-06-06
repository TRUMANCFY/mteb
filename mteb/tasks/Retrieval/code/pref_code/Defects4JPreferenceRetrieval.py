from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
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


class Defects4JPreferenceRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    metadata = TaskMetadata(
        name="Defects4JPreferenceRetrieval",
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
        # DATASET_DIR = os.getenv("COQUIR_DATASET_PATH", 'datasets/')

        # if not os.path.exists(DATASET_DIR):
        #     raise ValueError(f"Dataset directory {DATASET_DIR} does not exist. Please set the COQUIR_DATASET_PATH environment variable.")

        # DEFECTS4J_DIR = os.path.join(DATASET_DIR, 'Defects4J')
        # corpus_defects4j_file = os.path.join(DEFECTS4J_DIR, 'corpus.jsonl')
        # qrels_defects4j_file = os.path.join(DEFECTS4J_DIR, 'qrels.jsonl')
        # query_defects4j_file = os.path.join(DEFECTS4J_DIR, 'query.jsonl')

        # corpus_defects4j_lines = load_jsonl(corpus_defects4j_file)
        # qrels_defects4j_lines = load_jsonl(qrels_defects4j_file)
        # query_defects4j_lines = load_jsonl(query_defects4j_file)

        dataset = load_dataset("CoQuIR/Defects4J")
        qrels_defects4j_lines = list(dataset['test'])

        corpus = load_dataset("CoQuIR/Defects4J", "corpus")
        corpus_defects4j_lines = list(corpus['corpus'])

        query = load_dataset("CoQuIR/Defects4J", "query")
        query_defects4j_lines = list(query['query'])

        # convert query_defects_lines to dict
        query_defects4j_dict = {_line['id']: _line for _line in query_defects4j_lines}

        self.queries = {self._EVAL_SPLIT: {}}
        self.corpus = {self._EVAL_SPLIT: {}}
        self.relevant_docs = {self._EVAL_SPLIT: {}}

        # insert corpus
        for _line in tqdm(corpus_defects4j_lines):
            _doc_id = _line['id']
            self.corpus[self._EVAL_SPLIT][_doc_id] = {
                "title": str(_line.get('title', '')),
                "text": str(_line['text'])
            }
        # insert queries and relevant docs
        for _line in tqdm(qrels_defects4j_lines):
            qid = _line['qid']
            pos_doc_ids = _line['pos-docids']

            for pos_doc_id in pos_doc_ids:
                # insert query
                # insert relevant docs
                if qid not in self.queries[self._EVAL_SPLIT]:
                    self.queries[self._EVAL_SPLIT][qid] = str(query_defects4j_dict[qid]['text'])
                
                if qid not in self.relevant_docs[self._EVAL_SPLIT]:
                    self.relevant_docs[self._EVAL_SPLIT][qid] = {}
                self.relevant_docs[self._EVAL_SPLIT][qid][pos_doc_id] = 1

        self.data_loaded = True
