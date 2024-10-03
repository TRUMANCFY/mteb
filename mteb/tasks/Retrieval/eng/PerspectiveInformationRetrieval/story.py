from __future__ import annotations

import csv
import os
import json

from mteb.abstasks.TaskMetadata import TaskMetadata

from .....abstasks.AbsTaskRetrieval import AbsTaskRetrieval



class StoryInformationRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="StoryInformationRetrieval",
        reference="https://github.com/colinzhaoust/pir",
        description=(
            "Story"
        ),
        dataset={
            "path": "RAR-b/alphanli",
            "revision": "303f40ef3d50918d3dc43577d33f2f7344ad72c1",
            "trust_remote_code": True,
        }, # just for placeholder
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_5",
        date=(
            "2024-05-04",
            "2024-05-05",
        ),  # The period here is for both wiki articles and queries
        domains=["Encyclopaedic", "Academic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 1998},
            "avg_character_length": {
                "test": {
                    "average_document_length": 16.216,
                    "average_query_length": 26.00750750750751,
                    "num_documents": 2000,
                    "num_queries": 1998,
                    "average_relevant_docs_per_query": 1.001,
                },
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}

        # optional each of them will have two additional keys as 

        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            corpus, queries, qrels = self._load_data_for_split()
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

        self.data_loaded = True

    def _load_data_for_split(self):
        corpus, queries, qrels = {}, {}, {}

        root_dir = os.getenv("ROOT_DIR", "/storage/ukp/work/cai_e/instruction_pir/retrieval")
        data_dir = os.path.join(root_dir, "data/pir/story")
        
        corpus_file = os.path.join(data_dir, "corpus.jsonl")
        query_file = os.path.join(data_dir, "queries.jsonl")
        qrels_file = os.path.join(data_dir, 'qrels.tsv')

        corpus_lines = self._readjsonls(corpus_file)
        query_lines = self._readjsonls(query_file)

        for _line in corpus_lines:
            _id = _line['id']
            _title, _text = _line['contents'].split('\n')
            corpus[_id] = {
                'id': _id,
                "title": _title,
                "text": _text,
            }
        
        for _line in query_lines:
            queries[_line['id']] = _line['title']

        qrels = self._readqrels(qrels_file)
              
        return corpus, queries, qrels
    
    @classmethod
    def _readjsonls(cls, file):
        with open(file, 'r') as f:
            return [json.loads(l) for l in f.readlines()]
        
    @classmethod
    def _readqrels(cls, qrels_file):
        reader = csv.reader(open(qrels_file, encoding="utf-8"),
                            delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        
        next(reader)
        
        qrels = {}
        
        for _id, _row in enumerate(reader):
            query_id, corpus_id, score = _row[0], _row[1], int(float(_row[2]))
            
            if query_id not in qrels:
                qrels[query_id] = {corpus_id: score}
            else:
                qrels[query_id][corpus_id] = score
                
        return qrels