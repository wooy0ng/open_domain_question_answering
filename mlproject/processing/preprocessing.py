from transformers import PreTrainedTokenizerFast
from arguments.data_arguments import DataTrainingArguments
from datasets import DatasetDict, Dataset
from tqdm import tqdm


def preprocess_train_features(
    datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
    data_args: DataTrainingArguments,
) -> Dataset:
    """train 시 사용할 데이터에 대한 전처리 진행 함수"""
    pad_on_right = tokenizer.padding_side == "right"
    column_names = datasets["train"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    train_dataset = datasets["train"]

    # [CLS]question[SEP]context
    tokenized = tokenizer(
        train_dataset[question_column_name if pad_on_right else context_column_name],
        train_dataset[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=data_args.max_seq_length,
        return_overflowing_tokens=True,
        stride=data_args.doc_stride,
        return_offsets_mapping=True,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    """
    길이가 긴 context가 등장할 경우 truncate를 진행해야한다.
    해당하는 데이터를 찾을 수 있도록 mapping 가능한 값이 필요하다.
    """
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")  # 문장 index
    offset_mapping = tokenized.pop(
        "offset_mapping"
    )  # offset (token이 문장의 어느 위치에 있는지 알려줌)

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []
    for idx, offsets in tqdm(
        enumerate(offset_mapping),
        desc="find token_start_idx and token_end_idx",
        total=len(offset_mapping),
    ):
        input_ids = tokenized["input_ids"][idx]
        cls_idx = input_ids.index(tokenizer.cls_token_id)  # cls token이 존재하는 idx 위치
        sequence_ids = tokenized.sequence_ids(idx)  # segment

        sentence_idx = sample_mapping[idx]  # 현재 문장의 index
        answers = train_dataset[answer_column_name][
            sentence_idx
        ]  # {'answer_start': [235], 'text': ['하원']}

        # answer가 없을 경우 cls_index를 answer로 설정
        if len(answers["answer_start"]) == 0:
            tokenized["start_positions"].append(cls_idx)
            tokenized["end_positions"].append(cls_idx)
        else:
            # text에서 정답의 start/end character index
            start_char_idx = answers["answer_start"][0]
            end_char_idx = start_char_idx + len(answers["text"][0])

            # text에서 current span의 start token index - context의 시작을 찾기 위함
            token_start_idx = 0
            while sequence_ids[token_start_idx] != (1 if pad_on_right else 0):
                token_start_idx += 1

            # text에서 current span의 end token index - context의 끝을 찾기 위함
            token_end_idx = len(input_ids) - 1
            while sequence_ids[token_end_idx] != (1 if pad_on_right else 0):
                token_end_idx -= 1

            # 정답이 context 내에 있는지 확인 (정답이 없는 경우 CLS token으로 label 되어있음)
            if not (
                offsets[token_start_idx][0] <= start_char_idx
                and offsets[token_end_idx][1] >= end_char_idx
            ):
                tokenized["start_positions"].append(cls_idx)
                tokenized["end_positions"].append(cls_idx)
            else:
                # 정답의 시작 token 위치를 찾기 위함
                while (
                    token_start_idx < len(offsets)
                    and offsets[token_start_idx][0] <= start_char_idx
                ):
                    token_start_idx += 1
                tokenized["start_positions"].append(token_start_idx - 1)

                # 정답의 끝 token 위치를 찾기 위함
                while offsets[token_end_idx][1] >= end_char_idx:
                    token_end_idx -= 1
                tokenized["end_positions"].append(token_end_idx + 1)

    tokenized = Dataset.from_dict(tokenized)
    return tokenized


def preprocess_eval_features(
    datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
    data_args: DataTrainingArguments,
) -> Dataset:
    pad_on_right = tokenizer.padding_side == "right"
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    eval_dataset = datasets["validation"]

    # question[SEP]context
    tokenized = tokenizer(
        eval_dataset[question_column_name if pad_on_right else context_column_name],
        eval_dataset[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=data_args.max_seq_length,
        return_overflowing_tokens=True,
        stride=data_args.doc_stride,
        return_offsets_mapping=True,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []

    for idx in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(idx)
        context_idx = 1 if pad_on_right else 0

        # 하나의 문장이 여러 개의 span을 가질 수 있다.
        sentence_idx = sample_mapping[idx]
        tokenized["example_id"].append(eval_dataset["id"][sentence_idx])

        # context의 token offset만 저장
        tokenized["offset_mapping"][idx] = [
                (o if sequence_ids[k] == context_idx else None)
                for k, o in enumerate(tokenized["offset_mapping"][idx])
        ]
    tokenized = Dataset.from_dict(tokenized)
    return tokenized
