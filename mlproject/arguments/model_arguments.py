from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelTrainingArguments:
    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "pretrained tokenizer name or path if not the same as model_name"
        },
    )
    
