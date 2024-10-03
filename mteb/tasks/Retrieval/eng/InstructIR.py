from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

from datasets import load_dataset

class InstructIR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="InstructIR",
        description="InstructIR",
        reference="https://arxiv.org/abs/2402.14334",
        dataset={
            "path": "kaist-ai/InstructIR",
            "revision": "1615ed0ee07d6d33b0082362f008bfe62041a54b",
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=None,
        sample_creation="found",
        bibtex_citation="""
        @article{oh2024instructir,
      title={INSTRUCTIR: A Benchmark for Instruction Following of Information Retrieval Models}, 
      author={Hanseok Oh and Hyunji Lee and Seonghyeon Ye and Haebin Shin and Hansol Jang and Changwook Jun and Minjoon Seo},
      year={2024},
      eprint={2402.14334},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 26948.344086021505,
                    "average_query_length": 3038.42,
                    "num_documents": 186,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 3.9,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}

        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            corpus, queries, qrels = self._load_data_for_split()
            self.corpus[split], self.queries[split], self.relevant_docs[split] = corpus, queries, qrels

        self.data_loaded = True

    def _load_data_for_split(self):
        corpus, queries, qrels = {}, {}, {}

        corpus_lines = list(load_dataset('kaist-ai/InstructIR', 'corpus')['corpus'])
        queries_lines = list(load_dataset('kaist-ai/InstructIR', 'queries')['queries'])
        qrels_lines = list(load_dataset('kaist-ai/InstructIR', 'qrels')['test'])

        # build corpus
        for line in corpus_lines:
            corpus[line['_id']] = line

        # build queries
        for line in queries_lines:
            queries[line['_id']] = line['text']

        # build qrels
        for line in qrels_lines:
            qrels[line['query-id']] = qrels.get(line['query-id'], {})
            qrels[line['query-id']].update({line['corpus-id']: int(line['score'])})


        return corpus, queries, qrels


