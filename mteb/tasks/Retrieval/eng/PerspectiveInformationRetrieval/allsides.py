from __future__ import annotations

import csv
import os
import json

from mteb.abstasks.TaskMetadata import TaskMetadata

from .....abstasks.AbsTaskRetrieval import AbsTaskRetrieval



class AllsidesInformationRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="AllsidesInformationRetrieval",
        reference="https://github.com/colinzhaoust/pir",
        description=(
            "Allsides"
        ),
        dataset={
            "path": "trumancai/perspective-information-retrieval-allsides",
            "revision": "cb004a74fa915b277a181681149ae14139fbb562",
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
            "n_samples": {"test": 324},
            "avg_character_length": {
                "test": {
                    "average_document_length": 1089.2201550387597,
                    "average_query_length": 12.407407407407407,
                    "num_documents": 645,
                    "num_queries": 324,
                    "average_relevant_docs_per_query": 1.0,
                },
            },
        },
    )
