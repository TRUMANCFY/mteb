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
            "path": "trumancai/perspective-information-retrieval-story",
            "revision": "b3fb76b6af6dac9d2d93e5f74318ce0f8a716028",
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