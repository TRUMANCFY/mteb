from __future__ import annotations

import logging
import math
from typing import Any, List, Dict, Union, Literal, Callable

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import torch.multiprocessing as mp
import queue

from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "passage"]

logger = logging.getLogger(__name__)

class RepLLaMAWrapper:
    def __init__(self, *args, **kwargs):
        self.base_model_name_or_path = kwargs["base_model_name_or_path"]
        self.peft_model_name_or_path = kwargs["peft_model_name_or_path"]
        self.torch_dtype = kwargs["torch_dtype"]
        self.device_map = kwargs.get("device_map", None)

        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

        # If only one GPU or CPU, load model directly
        if num_gpus <= 1:
            self.base_model = AutoModel.from_pretrained(
                self.base_model_name_or_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            self.model = PeftModel.from_pretrained(self.base_model, self.peft_model_name_or_path)
            self.model = self.model.merge_and_unload()

            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

            self.model.config.max_length = 512
            self.tokenizer.model_max_length = 512

    def create_batch_dict(self, tokenizer, input_texts):
        max_length = self.model.config.max_length
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]
        ]
        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def encode(
        self,
        sentences: List[str],
        *,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        If multiple GPUs are available, start a multi-process pool, encode, then stop the pool.
        Otherwise, uses single-process encoding.
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # Multiple GPUs: start multi-process pool
            pool = self.start_multi_process_pool()
            try:
                embeddings = self.encode_multi_process(sentences, pool, batch_size=batch_size)
            finally:
                # Ensure the pool is stopped even if an exception occurs
                self.stop_multi_process_pool(pool)
            return embeddings
        else:
            # Single GPU or CPU: single-process encoding
            device = self.model.device
            all_embeddings = []
            for i in tqdm.tqdm(range(0, len(sentences), batch_size)):
                batch_texts = sentences[i : i + batch_size]
                batch_dict = self.create_batch_dict(self.tokenizer, batch_texts)
                batch_dict = {key: value.to(device) for key, value in batch_dict.items()}

                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        outputs = self.model(**batch_dict)
                        last_hidden_state = outputs.last_hidden_state
                        seq_lengths = batch_dict["attention_mask"].sum(dim=1) - 1
                        current_batch_size = last_hidden_state.shape[0]
                        reps = last_hidden_state[
                            torch.arange(current_batch_size, device=last_hidden_state.device),
                            seq_lengths,
                        ]
                        embeddings = F.normalize(reps, p=2, dim=-1)
                        all_embeddings.append(embeddings.cpu().numpy())

            return np.concatenate(all_embeddings, axis=0)

    @staticmethod
    def _encode_multi_process_worker(
        device: str,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        torch_dtype: torch.dtype,
        input_queue: mp.Queue,
        results_queue: mp.Queue,
    ) -> None:
        base_model = AutoModel.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=None,
        ).to(device)

        model = PeftModel.from_pretrained(base_model, peft_model_name_or_path)
        model = model.merge_and_unload()
        model.config.max_length = 512

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        def create_batch_dict(input_texts):
            max_length = model.config.max_length
            batch_dict = tokenizer(
                input_texts,
                max_length=max_length - 1,
                return_token_type_ids=False,
                return_attention_mask=False,
                padding=False,
                truncation=True,
            )
            batch_dict["input_ids"] = [
                input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]
            ]
            return tokenizer.pad(
                batch_dict,
                padding=True,
                pad_to_multiple_of=8,
                return_attention_mask=True,
                return_tensors="pt",
            )

        while True:
            try:
                task = input_queue.get(timeout=1)
                if task is None:
                    print(f"Worker on device {device} received None -> break")
                    break

                chunk_id = task["chunk_id"]
                sentences = task["sentences"]
                batch_size = task["batch_size"]

                all_embeddings = []
                for start_idx in range(0, len(sentences), batch_size):
                    batch_texts = sentences[start_idx : start_idx + batch_size]
                    batch_dict = create_batch_dict(batch_texts)
                    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            outputs = model(**batch_dict)
                            last_hidden_state = outputs.last_hidden_state
                            seq_lengths = batch_dict["attention_mask"].sum(dim=1) - 1
                            current_batch_size = last_hidden_state.shape[0]
                            reps = last_hidden_state[
                                torch.arange(current_batch_size, device=last_hidden_state.device),
                                seq_lengths,
                            ]
                            embeddings = F.normalize(reps, p=2, dim=-1)
                            all_embeddings.append(embeddings.cpu().numpy())

                embeddings = np.concatenate(all_embeddings, axis=0)
                results_queue.put([chunk_id, embeddings])
            except queue.Empty:
                print(f"Worker on device {device} queue empty -> break")
                break

    def start_multi_process_pool(
        self, target_devices: List[str] = None
    ) -> Dict[str, Any]:
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            else:
                print("CUDA is not available. Starting 4 CPU workers")
                target_devices = ["cpu"] * 4

        print("Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices))))

        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in target_devices:
            p = ctx.Process(
                target=self._encode_multi_process_worker,
                args=(
                    device_id,
                    self.base_model_name_or_path,
                    self.peft_model_name_or_path,
                    self.torch_dtype,
                    input_queue,
                    output_queue,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)
            print(f"Starting worker on device {device_id}")

        return {"input": input_queue, "output": output_queue, "processes": processes}

    @staticmethod
    def stop_multi_process_pool(pool: Dict[str, Any]) -> None:
        # Send termination signal
        for _ in pool["processes"]:
            pool["input"].put(None)

        for p in pool["processes"]:
            p.join()
            p.close()

        pool["input"].close()
        pool["output"].close()

    def encode_multi_process(
        self,
        sentences: List[str],
        pool: Dict[str, Any],
        batch_size: int = 32,
        chunk_size: int = None,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        if chunk_size is None:
            chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 10000)

        print(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        # Split sentences into chunks and queue them
        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put({
                    "chunk_id": last_chunk_id,
                    "batch_size": batch_size,
                    "sentences": chunk,
                })
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put({
                "chunk_id": last_chunk_id,
                "batch_size": batch_size,
                "sentences": chunk,
            })
            last_chunk_id += 1

        output_queue = pool["output"]

        # Collect results
        results_list = sorted(
            [output_queue.get() for _ in tqdm.trange(last_chunk_id, desc="Chunks", disable=not show_progress_bar)],
            key=lambda x: x[0],
        )
        embeddings = np.concatenate([result[1] for result in results_list])
        return embeddings

    def encode_corpus(
        self,
        corpus: List[dict[str, str]] | Dict[str, List[str]] | List[str],
        prompt_name: str = None,
        **kwargs: Any,
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus, sep=" ")
        if "request_qid" in kwargs:
            kwargs.pop("request_qid")
        sentences = [f"passage:  {sentence}".strip() for sentence in sentences]
        print(f"Encoding corpus of length {len(sentences)}")
        print(f"First sentence: {sentences[0]}")
        return self.encode(sentences, **kwargs)

    def encode_queries(self, queries: List[str], **kwargs: Any) -> np.ndarray:
        queries = [f"query:  {query.strip()}".strip() for query in queries]
        print(f"Encoding queries of length {len(queries)}")
        print(queries[0])
        return self.encode(queries, **kwargs)



# class RepLLaMAWrapper:
#     def __init__(self, *args, **kwargs):
#         try:
#             from peft import PeftModel
#         except ImportError:
#             raise ImportError(
#                 "To use the RepLLaMA based models `peft` is required. Please install it with `pip install 'mteb[peft]'`."
#             )

#         self.base_model = AutoModel.from_pretrained(
#             kwargs["base_model_name_or_path"],
#             torch_dtype=kwargs["torch_dtype"],
#             device_map=kwargs["device_map"],
#         )
#         self.model = PeftModel.from_pretrained(
#             self.base_model, kwargs["peft_model_name_or_path"]
#         )
#         self.model = self.model.merge_and_unload()

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             kwargs["base_model_name_or_path"]
#         )
#         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.tokenizer.padding_side = "right"
#         # set the max_length for the evals as they did, although the model can handle longer
#         self.model.config.max_length = 512
#         self.tokenizer.model_max_length = 512

#         print(self.model.layers[0].mlp.gate_proj.weight.sum())


#     def create_batch_dict(self, tokenizer, input_texts):
#         max_length = self.model.config.max_length
#         batch_dict = tokenizer(
#             input_texts,
#             max_length=max_length - 1,
#             return_token_type_ids=False,
#             return_attention_mask=False,
#             padding=False,
#             truncation=True,
#         )
#         batch_dict["input_ids"] = [
#             input_ids + [tokenizer.eos_token_id]
#             for input_ids in batch_dict["input_ids"]
#         ]
#         return tokenizer.pad(
#             batch_dict,
#             padding=True,
#             pad_to_multiple_of=8,
#             return_attention_mask=True,
#             return_tensors="pt",
#         )

#     def encode(
#         self,
#         sentences: list[str],
#         *,
#         prompt_name: str = None,
#         **kwargs: Any,  # noqa
#     ) -> np.ndarray:
#         batch_size = 16 if "batch_size" not in kwargs else kwargs.pop("batch_size")
#         all_embeddings = []
#         for i in tqdm.tqdm(range(0, len(sentences), batch_size)):
#             batch_texts = sentences[i : i + batch_size]

#             batch_dict = self.create_batch_dict(self.tokenizer, batch_texts)
#             batch_dict = {
#                 key: value.to(self.model.device) for key, value in batch_dict.items()
#             }

#             with torch.cuda.amp.autocast():
#                 with torch.no_grad():
#                     outputs = self.model(**batch_dict)
#                     last_hidden_state = outputs.last_hidden_state
#                     sequence_lengths = batch_dict["attention_mask"].sum(dim=1) - 1
#                     batch_size = last_hidden_state.shape[0]
#                     reps = last_hidden_state[
#                         torch.arange(batch_size, device=last_hidden_state.device),
#                         sequence_lengths,
#                     ]
#                     embeddings = F.normalize(reps, p=2, dim=-1)
#                     all_embeddings.append(embeddings.cpu().numpy())

#         return np.concatenate(all_embeddings, axis=0)

#     def encode_corpus(
#         self,
#         corpus: list[dict[str, str]] | dict[str, list[str]] | list[str],
#         prompt_name: str = None,
#         **kwargs: Any,
#     ) -> np.ndarray:
#         sentences = corpus_to_texts(corpus, sep=" ")
#         if "request_qid" in kwargs:
#             kwargs.pop("request_qid")
#         # NOTE: two spaces after the colon
#         sentences = [f"passage:  {sentence}".strip() for sentence in sentences]
#         print(f"Encoding corpus of length {len(sentences)}")
#         print(f"First sentence: {sentences[0]}")
#         return self.encode(sentences, **kwargs)

#     def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
#         # NOTE: two spaces after the colon
#         queries = [f"query:  {query.strip()}".strip() for query in queries]
#         print(f"Encoding queries of length {len(queries)}")
#         print(queries[0])
#         return self.encode(queries, **kwargs)


def _loader(wrapper: type[RepLLaMAWrapper], **kwargs) -> Callable[..., Encoder]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> Encoder:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner


repllama_llama2_original = ModelMeta(
    loader=_loader(
        RepLLaMAWrapper,
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        peft_model_name_or_path="castorini/repllama-v1-7b-lora-passage",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="castorini/repllama-v1-7b-lora-passage",
    languages=["eng_Latn"],
    open_source=True,
    revision="01c7f73d771dfac7d292323805ebc428287df4f9-6097554dfe6e7d93e92f55010b678bcca1e233a8",  # base-peft revision
    release_date="2023-10-11",
)


repllama_llama2_reproduced = ModelMeta(
    loader=_loader(
        RepLLaMAWrapper,
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        peft_model_name_or_path="samaya-ai/RepLLaMA-reproduced",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="samaya-ai/RepLLaMA-reproduced",
    languages=["eng_Latn"],
    open_source=True,
    revision="01c7f73d771dfac7d292323805ebc428287df4f9-ad5c1d0938a1e02954bcafb4d811ba2f34052e71",  # base-peft revision
    release_date="2024-09-15",
)


## Debug code
# import mteb
# model = mteb.get_model("samaya-ai/RepLLaMA-reproduced")
# tasks = mteb.get_tasks(tasks=["SciFact"], languages=["eng"])
# evaluation = mteb.MTEB(tasks=tasks)
# evaluation.run(model)
