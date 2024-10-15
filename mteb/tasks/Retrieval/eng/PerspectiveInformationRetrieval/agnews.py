from __future__ import annotations

import csv
import os
import json

from mteb.abstasks.TaskMetadata import TaskMetadata

from .....abstasks.AbsTaskRetrieval import AbsTaskRetrieval



class AGNewsInformationRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="AGNewsInformationRetrieval",
        reference="https://github.com/colinzhaoust/pir",
        description=(
            "AGNews"
        ),
        dataset={
            "path": "trumancai/perspective-information-retrieval-agnews",
            "revision": "d314401515c7c68621ccf8b5baa8372a3d0fd4f8",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_5",
        date=(
            "2024-05-04",
            "2024-05-05",
        ),  # The period here is for both wiki articles and queries - just for placeholder
        domains=["Encyclopaedic", "Academic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        descriptive_stats={
            "n_samples": {"test": 624},
            "avg_character_length": {
                "test": {
                    "average_document_length": 169.64509990485251,
                    "average_query_length": 88.86390854654327,
                    "num_documents": 1051,
                    "num_queries": 3674,
                    "average_relevant_docs_per_query": 1.0,
                },
            },
        },
    )