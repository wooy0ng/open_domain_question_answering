import os
import utils
from arguments.model_arguments import ModelTrainingArguments
from arguments.data_arguments import DataTrainingArguments
from arguments.training_arguments import MyTrainingArguments
from processing.preprocessing import preprocess_train_features, preprocess_eval_features
from processing.postprocessing import post_processing_function
from trainer.compute import compute_metric
from trainer.trainer_qa import QuestionAnsweringTrainer
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    Trainer
)
from datasets import DatasetDict, load_from_disk 
from typing import NoReturn


def main() -> NoReturn:
    parser = HfArgumentParser(
        (ModelTrainingArguments, DataTrainingArguments, MyTrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        use_fast=True,
    )

    # model config
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path
    )

    # load model
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(
            ".ckpt" in model_args.model_name_or_path
        ),  # if pretrained model is from tensorflow
        config=config,
    )

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    if training_args.do_train or training_args.do_eval:
        run_mrc(model_args, data_args, training_args, datasets, tokenizer, model)


def run_mrc(
    model_args: ModelTrainingArguments,
    data_args: DataTrainingArguments,
    training_args: MyTrainingArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:
    # check error
    last_checkpoint = utils.check_no_error(
        data_args=data_args,
        training_args=training_args,
        datasets=datasets,
        tokenizer=tokenizer,
    )

    # preprocessing
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")

        train_dataset = preprocess_train_features(
            datasets=datasets,
            tokenizer=tokenizer,
            data_args=data_args,
        )

    if training_args.do_eval:
        eval_dataset = preprocess_eval_features(
            datasets=datasets, 
            tokenizer=tokenizer, 
            data_args=data_args
        )
    
    # data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )
    
    '''
    TODO: custom trainer 공부
        TODO: trainer의 동작 구조 및 evaluation 흐름 이해
    '''
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metric,
        
        # custom trainer's parameters
        eval_examples=datasets['validation'] if training_args.do_eval else None,
        post_process_function=post_processing_function,
        data_args=data_args
    )
    
    # training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()    # save the model and tokenizer
        
        # metrics={'train_runtime': 70.5171, 'train_samples_per_second': 1.503, 'train_steps_per_second': 0.383, 'train_loss': 4.528071650752315, 'epoch': 1.0})
        metrics = train_result.metrics  
        metrics['train_samples'] = len(train_dataset)

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()    # state 저장
        
        output_train_file = os.path.join(training_args.output_dir, 'train_results.txt')
        
        with open(output_train_file, 'w') as f:
            for key, value in sorted(train_result.metrics.items()):
                f.write(f"{key} = {value}\n")
        
    # evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics['eval_samples'] = len(eval_dataset)
        
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)
        
    return


if __name__ == "__main__":
    main()
