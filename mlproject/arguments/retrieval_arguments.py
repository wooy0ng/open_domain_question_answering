from dataclasses import dataclass, field


@dataclass
class RetrievalArguments:
    dataset_name: str = field(
        default='../data/train_dataset',
        metadata={
            'help': ""
        }
    )
    model_name_or_path: str = field(
        default='bert-base-multilingual-cased',
        metadata={
            'help': ""
        }
    )
    data_path: str = field(
        default='../data',
        metadata={
            'help': ""
        }
    )
    context_path: str = field(
        default='../data/wikipedia_documents.json',
        metadata={
            'help': ""
        }
    )
    use_faiss: bool = field(
        default=True,
        metadata={
            'help': ""
        }
    )