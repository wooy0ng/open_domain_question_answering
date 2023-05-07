from dataclasses import dataclass, field


@dataclass
class RetrievalArguments:
    dataset_name: str = field(
        default='../data/train_dataset',
        metadata={
            'help': "local에 저장된 dataset의 이름을 입력하세요."
        }
    )
    model_name_or_path: str = field(
        default='bert-base-multilingual-cased',
        metadata={
            'help': "huggingface 모델의 이름을 입력하거나 local에 저장되어 있는 huggingface style의 모델 위치를 입력하세요."
        }
    )
    data_path: str = field(
        default='../data',
        metadata={
            'help': "파이프라인 진행 중 발생되는 데이터가 저장될 위치를 지정하세요."
        }
    )
    context_path: str = field(
        default='../data/wikipedia_documents.json',
        metadata={
            'help': "사용할 말뭉치의 위치를 입력하세요."
        }
    )
    use_elastic: bool = field(
        default=True,
        metadata={
            'help': "elastic search를 사용할 경우 `True`로 지정하세요."
        }
    )
    use_bm25: bool = field(
        default=True,
        metadata={
            'help': "bm25를 사용할 경우 `True`로 지정하세요. 지정하지 않을 경우 TFIDF을 수행합니다."
        }
    )
    use_faiss: bool = field(
        default=False,
        metadata={
            'help': "faiss를 사용할 경우 `True`로 지정하세요. "
        }
    )