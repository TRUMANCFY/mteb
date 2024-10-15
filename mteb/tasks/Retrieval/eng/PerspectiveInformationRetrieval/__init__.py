from .agnews import *
from .allsides import *
from .ambigqa import *
from .exfever import *
from .perspectrum import *
from .story import *
from .exfever_binary import *
from .perspectrum_binary import *

"""Abstract class for retrieval experiments.

Child-classes must implement the following properties:

self.corpus: dict[str, dict[str, str]]
    Semantically, it should contain dict[split_name, dict[sample_id, dict[str, str]]]
    E.g. {"test": {"document_one": {"_id": "d1", "title": "title", "text": "text"}}}

self.queries: dict[str, dict[str, Union[str, List[str]]]]
    Semantically, it should contain dict[split_name, dict[sample_id, str]] or dict[split_name, dict[sample_id, List[str]]] for conversations
    E.g. {"test": {"q1": "query"}}
    or {"test": {"q1": ["turn1", "turn2", "turn3"]}}

self.relevant_docs: dict[str, dict[str, dict[str, int]]]
    Semantically, it should contain dict[split_name, dict[sample_id, dict[doc_id, score]]]
    E.g.: {"test": {"q1": {"document_one": 1}}}
"""