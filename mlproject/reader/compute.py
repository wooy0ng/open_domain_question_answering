import evaluate
from transformers import EvalPrediction

metric = evaluate.load('squad')

def compute_metric(p: EvalPrediction):
    '''
    metric (squad)  
    Computes SQuAD scores (F1 and EM) \n
    Args: 
        predictions: List of question-answers dictionaries with the following key-values:  
        - 'id' : id of the question-answer pair as given in the references 
        - 'prediction_test' : the text of the answer  \n
        references : List of question-answers dictionaries with the following key-values:  
        - 'id' : id of the question-answer pair  \n
        - 'answers' : a Dict in the SQuAD dataset format  \n
            {
                'text' : list of possible texts for the answer, as a list of strings  \n
                'answer_start' : list of scart positions for the answer, as a list of ints  
            }\n
    Returns:
        'exact_match' : Exact match (the normalized answer exactly match the gold answer)  \n
        'f1' : The F-score of predicted tokens versus the gold answer  \n
    Examples:  
    
        >>> predictions = predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
        >>> references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
        >>> squad_metric = evaluate.load("squad")
        >>> results = squad_metric.compute(predictions=predictions, references=references)
        >>> print(results)
        {'exact_match': 100.0, 'f1': 100.0, stored examples: 0}
    '''
    
    return metric.compute(
        predictions=p.predictions,
        references=p.label_ids
    )