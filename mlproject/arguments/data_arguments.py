from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default='../data/sample',
        metadata={
            'help': "The name of the dataset to use."
        }
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            'help': "Overwrite the cached training and evaluation sets"
        }
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            'help': "The number of processes to use for the processing"
        }
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            'help': r'''
                The maximum total input sequence length after tokenizeation.
                Sequences longer than this will be truncated, sequences shorter will be padded.
            '''
        }
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            'help': r'''
                Whether to pad all samples to `max_seq_length`.
                If False, will pad the samples dynamically when batching to the maximum length 
                in in batch (which can be faster on GPU but will be slower on TPU).
            '''
        }
    )
    doc_stride: int = field(
        default=128,
        metadata={
            'help': "When splitting up a long document into chunks, how much stride to take between chunks."
        }
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            'help': r'''
                The maximum length of an answer that can be generated.
                This is needed because the start and end predictions are not conditioned on one another.
            '''
        }
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={
            'help': "Whether to run passage retrieval using sparse embedding."
        }
    )
    num_clusters: int = field(
        default=64,
        metadata={
            'help': "Define how many clusters to use for faiss."
        }
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            'help': "Define how many top-k passages to retrieve based on similarity."
        }
    )
    use_faiss: bool = field(
        default=False,
        metadata={
            'help': "Whether to build with faiss."
        }
    )
    
    