from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
from typing import NoReturn, Optional, List, Any, Callable
from datasets import Dataset
from arguments.data_arguments import DataTrainingArguments


class QuestionAnsweringTrainer(Trainer):
    def __init__(
        self,
        *args, 
        eval_examples: Dataset=None, 
        post_process_function: Optional[Callable[..., Any]]=None, 
        data_args: DataTrainingArguments=None, 
        **kwargs
    ):
        super(QuestionAnsweringTrainer, self).__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.data_args = data_args
        
    def evaluate(
        self,
        eval_dataset: Optional[Dataset]=None,
        ignore_keys: Optional[List[str]]=None,
        metric_key_prefix: str='eval',
        eval_examples: Any=None
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try: 
            '''
            output의 인자: 
                predictions: Tuple[np.ndarray, np.ndarray] 
                label_ids: Optional[Any] 
                metrics: dict
                num_samples: int
            '''
            output = self.prediction_loop(
                eval_dataloader,
                description='evaluation',
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )
        finally:
            self.compute_metrics = compute_metrics
        
        if isinstance(eval_dataset, Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format['type'],
                columns=list(eval_dataset.features.keys())
            )
        
        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions, self.args, self.data_args
            )
            metrics = self.compute_metrics(eval_preds)
            self.log(metrics)
        else:
            metrics = {}
        
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        
        return metrics
    
    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)
        
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description='Evaluation',
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys
            )
        finally:
            self.compute_metrics = compute_metrics
        
        if self.post_process_function is None or self.compute_metrics is None:
            return output
        
        if isinstance(test_dataset, Dataset):
            test_dataset.set_format(
                type=test_dataset.format['type'],
                columns=list(test_dataset.features.keys())
            )
        
        predictions = self.post_process_function(
            test_examples, test_dataset, output.predictions, self.args, self.data_args
        )
        return predictions