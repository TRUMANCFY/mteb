from __future__ import annotations

import csv
import os
import json

from mteb.abstasks.TaskMetadata import TaskMetadata

from .....abstasks.AbsTaskRetrieval import AbsTaskRetrieval



class PerspectrumInformationRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="PerspectrumInformationRetrieval",
        reference="https://github.com/colinzhaoust/pir",
        description=(
            "Perspectrum"
        ),
        dataset={
            "path": "trumancai/perspective-information-retrieval-perspectrum",
            "revision": "113d1f7be476d140d366f063c4a7bdda158ed2a7",
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
            "n_samples": {"test": 1104},
            "avg_character_length": {
                "test": {
                    "average_document_length": 10.862082362082361,
                    "average_query_length": 15.219551282051283,
                    "num_documents": 2574,
                    "num_queries": 1104,
                    "average_relevant_docs_per_query": 5.024456521739131,
                },
            },
        },
    )
