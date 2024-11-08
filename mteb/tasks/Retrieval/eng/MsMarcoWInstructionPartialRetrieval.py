from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

from datasets import load_dataset

class MsmarcoWInstructionPartialRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MsmarcoWInstructionPartialRetrieval",
        description="MsmarcoWInstructionPartialRetrieval",
        reference="https://arxiv.org/abs/2409.11136",
        dataset={
            "path": "trumancai/msmarco-w-instruction-partial",
            "revision": "5217256a3f26991a5ccb2fbc8f503399a5775c58",
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
                    "num_documents": 131274,
                    "num_queries": 10000,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )


