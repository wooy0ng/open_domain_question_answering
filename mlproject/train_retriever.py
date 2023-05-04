import argparse
import time
import numpy as np
import pandas as pd

from utils import timer
from arguments.retrieval_arguments import RetrievalArguments
from transformers import HfArgumentParser, AutoTokenizer
from datasets import load_from_disk, concatenate_datasets
from retriever.sparse_retrieval import SparseRetrieval


if __name__ == '__main__':
    parser = HfArgumentParser(RetrievalArguments)
    retrieval_args, = parser.parse_args_into_dataclasses()
    
    # Test sparse
    org_dataset = load_from_disk(retrieval_args.dataset_name)
    full_ds = concatenate_datasets([
        org_dataset['train'].flatten_indices(), org_dataset['validation'].flatten_indices()
    ])
    
    print("query datasets")
    print(full_ds)
    
    tokenizer = AutoTokenizer.from_pretrained(
        retrieval_args.model_name_or_path,
        use_fast=False,
    )
    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=retrieval_args.data_path,
        context_path=retrieval_args.context_path
    )
    
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    
    retriever.get_sparse_embedding()
    if retrieval_args.use_faiss:
        retriever.get_faiss_indexer()
        
    else:
        
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df['correct'] = df['original_context'] == df['context']
            
            print(
                "correct retrieval result by exhausive search",
                df['correct'].sum() / len(df)
            )

        with timer("single query by exhausive search"):
            scores, context = retriever.retrieve(query)
            
    pass