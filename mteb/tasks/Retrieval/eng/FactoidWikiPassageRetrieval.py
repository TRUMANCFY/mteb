from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

from datasets import load_dataset

class FactoidWikiPassageRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FactoidWikiPassage",
        description="FactoidWikiPassage",
        reference="https://arxiv.org/abs/2312.06648",
        dataset={
            "path": "https://huggingface.co/datasets/trumancai/factoid-wiki",
            "revision": "3c0160ea7d5768b606b36d82efe3ffc7e37c085a",
        },
        type="Retrieval",
        category="p2p",
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
        bibtex_citation="""@article{chen2023densex,
  title={Dense X Retrieval: What Retrieval Granularity Should We Use?},
  author={Tong Chen and Hongwei Wang and Sihao Chen and Wenhao Yu and Kaixin Ma and Xinran Zhao and Hongming Zhang and Dong Yu},
  journal={arXiv preprint arXiv:2312.06648},
  year={2023},
  URL = {https://arxiv.org/pdf/2312.06648.pdf}
}
        """,
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 80.02917113032743,
                    "average_query_length": 80.02917113032743,
                    "num_documents": 41393528,
                    "num_queries": 41393528,
                    "average_relevant_docs_per_query": 1,
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
        # load training for test
        corpus_lines = list(load_dataset('trumancai/factoid-wiki', split='train'))
        
        for line in corpus_lines:
            corpus[line['id']] = {
                '_id': line['_id'],
                'title': line['title'],
                'text': line['text'],
            }
            
            # distinguish between queries and corpus
            queries['query-' + line['_id']] = line['text']
            qrels['query-' + line['_id']] = {line['_id']: 1}

        return corpus, queries, qrels
