from __future__ import annotations

import csv
import os
import json

from mteb.abstasks.TaskMetadata import TaskMetadata

from .....abstasks.AbsTaskRetrieval import AbsTaskRetrieval



class AmbigQAInformationRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="AmbigQAInformationRetrieval",
        reference="https://github.com/colinzhaoust/pir",
        description=(
            "AmbigQA"
        ),
        dataset={
            "path": "trumancai/perspective-information-retrieval-ambigqa",
            "revision": "6395e0dc121189a771f7c643819d92ea6a05b813",
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
            "n_samples": {"test": 2295},
            "avg_character_length": {
                "test": {
                    "average_document_length": 27.43118218161051,
                    "average_query_length": 12.473202614379085,
                    "num_documents": 1751,
                    "num_queries": 2295,
                    "average_relevant_docs_per_query": 1.003,
                },
            },
        },
    )