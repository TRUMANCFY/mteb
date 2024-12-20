from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

from datasets import load_dataset

class MsmarcoWInstructionRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MsmarcoWInstructionRetrieval",
        description="MsmarcoWInstructionRetrieval",
        reference="https://arxiv.org/abs/2409.11136",
        dataset={
            "path": "trumancai/msmarco-w-instruction",
            "revision": "b1c058b8037cf599ade72749a8782b47d8a2cd14",
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
        @article{weller2024promptriever,
  title={Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models},
  author={Weller, Orion and Van Durme, Benjamin and Lawrie, Dawn and Paranjape, Ashwin and Zhang, Yuhao and Hessel, Jack},
  journal={arXiv preprint arXiv:2409.11136},
  year={2024}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 61.293623801412586,
                    "average_query_length": 109.157956271219,
                    "num_documents": 7907329,
                    "num_queries": 489243,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )


