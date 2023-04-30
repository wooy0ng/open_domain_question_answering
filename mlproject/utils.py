import os
from arguments.data_arguments import DataTrainingArguments
from transformers import TrainingArguments, PreTrainedTokenizerFast
from transformers.trainer_utils import get_last_checkpoint
from datasets import DatasetDict
from typing import Tuple, Any


def check_no_error(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer
) -> Any:
    
    # last checkpoint가 있는지 확인
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir) 
        and training_args.do_train 
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

        
    # Tokenizer check : Fast tokenizer를 사용하는지 확인
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )
    
    if data_args.max_seq_length > tokenizer.model_max_length:
        print(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    data_args.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    if 'validation' not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    return last_checkpoint