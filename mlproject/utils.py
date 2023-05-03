import os
import time
import sys, threading
from arguments.data_arguments import DataTrainingArguments
from transformers import TrainingArguments, PreTrainedTokenizerFast
from transformers.trainer_utils import get_last_checkpoint
from datasets import DatasetDict
from typing import Tuple, Any
from contextlib import contextmanager


class Spinner():
    def __init__(self, delay=None):
        import warnings
        import time
        warnings.filterwarnings(action='ignore', category=UserWarning)
        
        self.spinner_generator = self.spinning_cursor()
        
        self.desc = ""
        self.busy = False
        self.delay = 0.5
        self.start_time = 0
                
        if delay and float(delay): 
            self.delay = delay

    @staticmethod
    def spinning_cursor():
        while 1: 
            for cursor in '|/-\\': 
                yield cursor

    def spinner_task(self):
        print(f"[+] {self.desc}...", end=' ')
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def start(self, desc: str="spinner"):
        self.busy = True
        self.desc = desc
        self.start_time = time.time()
        threading.Thread(target=self.spinner_task).start()

    def stop(self):
        self.busy = False
        time.sleep(self.delay)
        
        print(f"end time : {time.time()-self.start_time}'s")


@contextmanager
def timer(name):
    t = time.time()
    yield
    print(f"{name} is done in {time.time() - t:.3f}s")


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