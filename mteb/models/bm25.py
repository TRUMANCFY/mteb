# from __future__ import annotations

# import logging
# from functools import partial
# from typing import Any
# import os
# import queue
# import multiprocessing as mp

# import bm25s  # Ensure this is available within the worker processes
# import Stemmer

# from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel
# from mteb.model_meta import ModelMeta
# from mteb.models.text_formatting_utils import corpus_to_texts

# logger = logging.getLogger(__name__)


# ##################################
# # Worker + Helper for Multiprocess
# ##################################

# def _build_or_load_bm25_retriever(
#     corpus_with_ids,
#     stopwords: str,
#     stemmer_language: str | None,
#     save_index_path: str | None = None,
# ):
#     """Each process will call this to load/build its own BM25 index."""
#     if save_index_path and os.path.exists(save_index_path):
#         logger.info("Worker: Loading BM25 index from disk...")
#         retriever = bm25s.BM25.load(save_index_path, load_corpus=True)
#     else:
#         logger.info("Worker: Building a fresh BM25 index in this process...")
#         # Build from scratch in the worker
#         corpus_texts = [
#             "\n".join([doc.get("title", ""), doc["text"]]) for doc in corpus_with_ids
#         ]
#         tokenized = bm25s.tokenize(
#             corpus_texts,
#             stopwords=stopwords,
#             stemmer=Stemmer.Stemmer(stemmer_language) if stemmer_language else None,
#         )
#         retriever = bm25s.BM25()
#         retriever.index(tokenized)
#     return retriever


# def _bm25_worker(
#     input_queue: mp.Queue,
#     output_queue: mp.Queue,
#     corpus_with_ids: list[dict[str, Any]],
#     top_k: int,
#     stopwords: str,
#     stemmer_language: str | None,
#     save_index_path: str | None,
# ):
#     """
#     Each worker builds or loads its own BM25 retriever. 
#     Then, for each chunk of raw queries from input_queue:
#       - Encodes queries (tokenizes them)
#       - Retrieves top_k results
#       - Puts (qids_chunk, results_chunk, scores_chunk) in output_queue
#     """
#     # 1) Build or load the retriever locally
#     local_retriever = _build_or_load_bm25_retriever(
#         corpus_with_ids, stopwords, stemmer_language, save_index_path
#     )

#     # 2) Start processing chunks of raw queries
#     while True:
#         try:
#             chunk_data = input_queue.get(timeout=30)  # 30s timeout
#             if chunk_data is None:
#                 # Signal to terminate
#                 break

#             # chunk_data is a list of (qid, raw_query)
#             local_qids, local_raw_queries = zip(*chunk_data)

#             # Encode queries *inside* this worker process
#             tokenized_queries = bm25s.tokenize(
#                 local_raw_queries,
#                 stopwords=stopwords,
#                 stemmer=Stemmer.Stemmer(stemmer_language) if stemmer_language else None,
#             )

#             # Retrieve top-k results
#             local_results, local_scores = local_retriever.retrieve(
#                 tokenized_queries, corpus=corpus_with_ids, k=top_k
#             )

#             output_queue.put((local_qids, local_results, local_scores))

#         except queue.Empty:
#             # If no data arrives for a while, just exit
#             break


# #########################################
# # Loader function that returns the class
# #########################################

# def bm25_loader(**kwargs):
#     """
#     This loader instantiates the BM25Search class
#     with the given kwargs, unchanged from your original code.
#     """
#     try:
#         import bm25s
#         import Stemmer
#     except ImportError:
#         raise ImportError(
#             "bm25s or Stemmer is not installed. Please install it with `pip install bm25s Stemmer`."
#         )

#     class BM25Search(DRESModel):
#         """BM25 search"""

#         def __init__(
#             self,
#             previous_results: str = None,
#             stopwords: str = "en",
#             stemmer_language: str | None = "english",
#             **kwargs,
#         ):
#             super().__init__(
#                 model=None,
#                 batch_size=1,
#                 corpus_chunk_size=1,
#                 previous_results=previous_results,
#                 **kwargs,
#             )

#             self.stopwords = stopwords
#             self.stemmer_language = stemmer_language
#             self.stemmer = (
#                 Stemmer.Stemmer(stemmer_language) if stemmer_language else None
#             )

#         @classmethod
#         def name(self):
#             return "bm25s"

#         def search(
#             self,
#             corpus: dict[str, dict[str, str]],
#             queries: dict[str, str | list[str]],
#             top_k: int,
#             return_sorted: bool = False,
#             **kwargs,
#         ) -> dict[str, dict[str, float]]:
#             """
#             If n_jobs=1 -> single-process as before.
#             If n_jobs>1 -> Each worker:
#                1) Builds/loads its own BM25 retriever
#                2) Encodes queries inside the worker
#                3) Retrieves top-k
#             """
#             save_index_path = kwargs.get("save_index_path", None)
#             n_jobs = kwargs.get("n_jobs", 1)
#             chunk_size = kwargs.get("chunk_size", 500)

#             # Prepare corpus data
#             corpus_ids = list(corpus.keys())
#             corpus_with_ids = [{"doc_id": cid, **corpus[cid]} for cid in corpus_ids]

#             # Convert to raw query list
#             query_ids = list(queries.keys())
#             queries_texts = [queries[qid] for qid in queries]
#             logger.info(f"Retrieving {len(queries)} queries with top_k={top_k}")

#             # For final result
#             final_results = {qid: {} for qid in query_ids}

#             if n_jobs == 1:
#                 # --- Single-process fallback (unchanged from original) ---
#                 logger.info("Single-process retrieval...")

#                 # Load or build retriever in main process
#                 if save_index_path and os.path.exists(save_index_path):
#                     logger.info("Loading precomputed BM25 index in single-process...")
#                     retriever = bm25s.BM25.load(save_index_path, load_corpus=True)
#                 else:
#                     logger.info("Building BM25 index in single-process...")
#                     corpus_texts = [
#                         "\n".join([doc.get("title", ""), doc["text"]])
#                         for doc in corpus_with_ids
#                     ]
#                     encoded_corpus = self.encode(corpus_texts)
#                     retriever = bm25s.BM25()
#                     retriever.index(encoded_corpus)
#                     if save_index_path:
#                         retriever.save(save_index_path)

#                 # Encode queries in main process
#                 tokenized_queries = self.encode(queries_texts, return_ids=False)

#                 # Iterate chunks
#                 for start_idx in range(0, len(query_ids), chunk_size):
#                     qids_chunk = query_ids[start_idx : start_idx + chunk_size]
#                     qtok_chunk = tokenized_queries[start_idx : start_idx + chunk_size]

#                     local_results, local_scores = retriever.retrieve(
#                         qtok_chunk, corpus=corpus_with_ids, k=top_k
#                     )
#                     for i, qid in enumerate(qids_chunk):
#                         for doc_dict, score in zip(local_results[i], local_scores[i]):
#                             final_results[qid][doc_dict["doc_id"]] = float(score)

#             else:
#                 # --- Multi-process retrieval, each worker has its own retriever ---
#                 logger.info(f"Multi-process retrieval: n_jobs={n_jobs}")

#                 # Prepare chunked raw queries (no encoding here; worker does it)
#                 id_query_pairs = list(zip(query_ids, queries_texts))
#                 chunks = [
#                     id_query_pairs[i : i + chunk_size]
#                     for i in range(0, len(id_query_pairs), chunk_size)
#                 ]
#                 ctx = mp.get_context("spawn")
#                 input_queue = ctx.Queue()
#                 output_queue = ctx.Queue()
#                 processes = []

#                 # Spawn worker processes; each builds/loads its own retriever
#                 for _ in range(n_jobs):
#                     p = ctx.Process(
#                         target=_bm25_worker,
#                         args=(
#                             input_queue,
#                             output_queue,
#                             corpus_with_ids,
#                             top_k,
#                             self.stopwords,
#                             self.stemmer_language,
#                             save_index_path,
#                         ),
#                     )
#                     p.start()
#                     processes.append(p)

#                 # Dispatch chunks
#                 for chunk_data in chunks:
#                     input_queue.put(chunk_data)

#                 # Collect results
#                 results_received = 0
#                 total_chunks = len(chunks)
#                 while results_received < total_chunks:
#                     qids_chunk, results_chunk, scores_chunk = output_queue.get()
#                     results_received += 1
#                     for i, qid in enumerate(qids_chunk):
#                         for doc_dict, score in zip(results_chunk[i], scores_chunk[i]):
#                             final_results[qid][doc_dict["doc_id"]] = float(score)

#                 # Send None to signal workers to shut down
#                 for _ in range(n_jobs):
#                     input_queue.put(None)

#                 # Wait for processes to finish
#                 for p in processes:
#                     p.join()

#                 input_queue.close()
#                 output_queue.close()

#             # Store final results in self.results
#             self.results = final_results
#             return self.results

#         def encode(self, texts: list[str], **kwargs):
#             """Encode input text as BM25 term vectors."""
#             return bm25s.tokenize(
#                 texts, stopwords=self.stopwords, stemmer=self.stemmer
#             )

#         def encode_queries(
#             self,
#             queries: list[str],
#             batch_size: int = 32,
#             **kwargs: Any,
#         ):
#             return self.encode(queries, kwargs=kwargs)

#         def encode_corpus(
#             self,
#             corpus: list[dict[str, str]] | dict[str, list[str]],
#             batch_size: int = 32,
#             **kwargs: Any,
#         ):
#             sentences = corpus_to_texts(corpus)
#             return self.encode(sentences, kwargs=kwargs)

#     return BM25Search(**kwargs)


# bm25_s = ModelMeta(
#     loader=partial(bm25_loader, model_name="bm25s"),  # type: ignore
#     name="bm25s",
#     languages=["eng_Latn"],
#     open_source=True,
#     revision="0_1_10",
#     release_date="2024-07-10",  # release of version 0.1.10
# )




from __future__ import annotations

import logging
from functools import partial
from typing import Any
import os
import pickle
import bm25s  # Ensure this is available within the worker processes

from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel
from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)



def _retrieve_chunk(chunk_data, retriever, corpus_with_ids, top_k):
    local_qids, local_qtokens = zip(*chunk_data)
    local_results, local_scores = retriever.retrieve(
        local_qtokens, corpus=corpus_with_ids, k=top_k
    )
    return local_qids, local_results, local_scores

def bm25_loader(**kwargs):
    try:
        import bm25s
        import Stemmer
    except ImportError:
        raise ImportError(
            "bm25s or Stemmer is not installed. Please install it with `pip install bm25s Stemmer`."
        )

    class BM25Search(DRESModel):
        """BM25 search"""

        def __init__(
            self,
            previous_results: str = None,
            stopwords: str = "en",
            stemmer_language: str | None = "english",
            **kwargs,
        ):
            super().__init__(
                model=None,
                batch_size=1,
                corpus_chunk_size=1,
                previous_results=previous_results,
                **kwargs,
            )

            self.stopwords = stopwords
            self.stemmer = (
                Stemmer.Stemmer(stemmer_language) if stemmer_language else None
            )

        @classmethod
        def name(self):
            return "bm25s"
        
        def search(
            self,
            corpus: dict[str, dict[str, str]],
            queries: dict[str, str | list[str]],
            top_k: int,
            return_sorted: bool = False,
            **kwargs,
        ) -> dict[str, dict[str, float]]:
            
            # Prepare corpus data
            corpus_ids = list(corpus.keys())
            corpus_with_ids = [{"doc_id": cid, **corpus[cid]} for cid in corpus_ids]
            
            # Either load or build the BM25 retriever
            if "save_index_path" in kwargs and os.path.exists(kwargs["save_index_path"]):
                logger.info("Loading precomputed embeddings...")
                retriever = bm25s.BM25.load(kwargs["save_index_path"], load_corpus=True)
            else:
                logger.info("Encoding Corpus...")
                corpus_texts = [
                    "\n".join([doc.get("title", ""), doc["text"]])
                    for doc in corpus_with_ids
                ]
                encoded_corpus = self.encode(corpus_texts)

                logger.info(
                    f"Indexing Corpus... {len(encoded_corpus.ids):,} documents, {len(encoded_corpus.vocab):,} vocab"
                )
                retriever = bm25s.BM25()
                retriever.index(encoded_corpus)

                if "save_index_path" in kwargs and kwargs["save_index_path"] is not None:
                    logger.info("Saving precomputed embeddings...")
                    retriever.save(kwargs["save_index_path"])

            # Encode all queries
            logger.info("Encoding Queries...")
            query_ids = list(queries.keys())
            self.results = {qid: {} for qid in query_ids}
            queries_texts = [queries[qid] for qid in queries]
            query_token_strs = self.encode(queries_texts, return_ids=False)

            logger.info(f"Retrieving Results... {len(queries):,} queries")

            # Optional parallelization parameters
            n_jobs = kwargs.get("n_jobs", 1)         # how many processes to spawn
            chunk_size = kwargs.get("chunk_size", 500)  # how many queries per chunk

            print(f"n_jobs: {n_jobs}, chunk_size: {chunk_size}")
            
            # Pair each query ID with its tokenized form
            id_token_str_pairs = list(zip(query_ids, query_token_strs))
            # Split into chunks for parallel processing
            chunks = [
                id_token_str_pairs[i : i + chunk_size]
                for i in range(0, len(id_token_str_pairs), chunk_size)
            ]


            final_results = {}

            if n_jobs == 1:
                # Single-process fallback (for debugging or if parallel is undesired)
                for chunk_data in chunks:
                    qids_chunk, results_chunk, scores_chunk = _retrieve_chunk(chunk_data, retriever, corpus_with_ids, top_k)
                    for i, qid in enumerate(qids_chunk):
                        doc_id_to_score = {}
                        for doc_dict, score in zip(results_chunk[i], scores_chunk[i]):
                            doc_id_to_score[doc_dict["doc_id"]] = float(score)
                        final_results[qid] = doc_id_to_score
            else:
                # Multi-process retrieval
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [
                        executor.submit(_retrieve_chunk, chunk_data, retriever, corpus_with_ids, top_k)
                        for chunk_data in chunks
                    ]
                    # Collect results from all processes
                    for future in futures:
                        qids_chunk, results_chunk, scores_chunk = future.result()
                        for i, qid in enumerate(qids_chunk):
                            doc_id_to_score = {}
                            for doc_dict, score in zip(results_chunk[i], scores_chunk[i]):
                                doc_id_to_score[doc_dict["doc_id"]] = float(score)
                            final_results[qid] = doc_id_to_score

            # Store final results in self.results
            self.results = final_results
            return self.results

        def encode(self, texts: list[str], **kwargs):
            """Encode input text as term vectors"""
            return bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer)

        def encode_queries(
            self,
            queries: list[str],
            batch_size: int = 32,
            **kwargs: Any,
        ):
            return self.encode(queries, kwargs=kwargs)

        def encode_corpus(
            self,
            corpus: list[dict[str, str]] | dict[str, list[str]],
            batch_size: int = 32,
            **kwargs: Any,
        ):
            sentences = corpus_to_texts(corpus)
            return self.encode(sentences, kwargs=kwargs)

    return BM25Search(**kwargs)

bm25_s = ModelMeta(
    loader=partial(bm25_loader, model_name="bm25s"),  # type: ignore
    name="bm25s",
    languages=["eng_Latn"],
    open_source=True,
    revision="0_1_10",
    release_date="2024-07-10",  ## release of version 0.1.10
)


        # def search(
        #     self,
        #     corpus: dict[str, dict[str, str]],
        #     queries: dict[str, str | list[str]],
        #     top_k: int,
        #     return_sorted: bool = False,
        #     **kwargs,
        # ) -> dict[str, dict[str, float]]:
            
        #     corpus_ids = list(corpus.keys())
        #     corpus_with_ids = [{"doc_id": cid, **corpus[cid]} for cid in corpus_ids]
            
        #     if "save_index_path"  in kwargs and os.path.exists(kwargs["save_index_path"]):
        #         print("Loading precomputed embeddings...")
        #         retriever = bm25s.BM25.load(kwargs["save_index_path"], load_corpus=True)
        #     else:
        #         print("Encoding Corpus...")
        #         corpus_texts = [
        #             "\n".join([doc.get("title", ""), doc["text"]])
        #             for doc in corpus_with_ids
        #         ]  # concatenate all document values (title, text, ...)
        #         encoded_corpus = self.encode(corpus_texts)

        #         print(
        #             f"Indexing Corpus... {len(encoded_corpus.ids):,} documents, {len(encoded_corpus.vocab):,} vocab"
        #         )

        #         # Create the BM25 model and index the corpus
        #         retriever = bm25s.BM25()
        #         retriever.index(encoded_corpus)

        #         if 'save_index_path' in kwargs and kwargs['save_index_path'] is not None:
        #             print("Saving precomputed embeddings...")
        #             # it will automatically save the corpus `self.corpus`
        #             retriever.save(kwargs['save_index_path'])
                    
        #     logger.info("Encoding Queries...")
        #     query_ids = list(queries.keys())
        #     self.results = {qid: {} for qid in query_ids}
        #     queries_texts = [queries[qid] for qid in queries]

        #     query_token_strs = self.encode(queries_texts, return_ids=False)

        #     logger.info(f"Retrieving Results... {len(queries):,} queries")

        #     queries_results, queries_scores = retriever.retrieve(
        #         query_token_strs, corpus=corpus_with_ids, k=top_k
        #     )

        #     # Iterate over queries
        #     for qi, qid in enumerate(query_ids):
        #         doc_id_to_score = {}
        #         query_results = queries_results[qi]
        #         scores = queries_scores[qi]
        #         doc_id_to_score = {}

        #         # Iterate over results
        #         for ri in range(len(query_results)):
        #             doc = query_results[ri]
        #             score = scores[ri]
        #             doc_id = doc["doc_id"]

        #             doc_id_to_score[doc_id] = float(score)

        #         self.results[qid] = doc_id_to_score

        #     return self.results