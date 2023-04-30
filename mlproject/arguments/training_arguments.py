from transformers import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="./results", metadata={"help": "you must set output directory"}
    )
    do_train: bool = field(default=False, metadata={"help": "training"})
    do_eval: bool = field(default=True, metadata={"help": "evaluation"})
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "set num train epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "train batch size"}
    )
    fp16: bool = field(default=True, metadata={"help": "set fp16"})
    # evaluation_strategy: str = field(
    #     default="epoch",
    #     metadata={"help": "set evaluation strategy"}
    # )
    # save_strategy: str = field(
    #     default="epoch",
    #     metadata={"help": "set save strategy"}
    # )
    
