import os
import collections
import numpy as np
import json
from tqdm import tqdm
from typing import Tuple, Optional
from datasets import Dataset
from arguments.training_arguments import MyTrainingArguments
from arguments.data_arguments import DataTrainingArguments
from transformers import EvalPrediction


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool=False,
    n_best_size: int=20,
    max_answer_length: int=30,
    null_score_diff_threshold: float=0.0,
    output_dir: Optional[str]=None,
    prefix: Optional[str]=None,
    is_world_process_zero: bool=True
):
    '''
    postprocess_qa_predictions() : QA 모델의 prediction 값을 후처리하는 함수.
    - 모델은 start logit과 end logit을 반환하기 때문에, 이를 기반으로 original text로 변경하는 후처리 작업 수행
    
    Args:
        examples: 전처리 되지 않은 데이터셋
        features: 전처리 된 데이터셋
        predictions: 모델의 예측값 (start logits, end logits를 나타내는 2개의 array)
        version_2_with_negative: 정답이 없는 데이터셋이 포함되어있는지 여부
        n_best_size: 답변을 찾을 때 생성할 n-best prediction 총 개수
        max_answer_length: 생성할 수 있는 답변의 최대 길이
        null_score_diff_threshold: null 답변을 선택하는 데 사용되는 threshold
        output_dir: 아래의 값이 저장되는 경로
        prefix: output dictionary에 `prefix`가 포함되어 저장
        is_world_process_zero: 이 프로세스가 main process인지 여부 (logging / save를 수행해야 하는지 여부 결정)
    '''
    assert (
        len(predictions) == 2
    ), "`predictions` should be a tuple with two elements (start_logits, end_logits)"
    all_start_logits, all_end_logits = predictions
    
    assert (
        len(predictions[0]) == len(features)
    ), f"Got {len(predictions[0])} predictions and {len(features)} features."
    
    # examples와 mapping되는 feature 생성 
    example_id_to_index = {key: idx for idx, key in enumerate(examples['id'])}
    features_per_example = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        features_per_example[example_id_to_index[feature['example_id']]].append(idx)
        
    # prediction, nbest에 해당하는 OrderedDict 생성
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()
        
    # 전체 example에 대한 loop
    for example_idx, example in enumerate(tqdm(examples, desc='prediction loop', total=len(examples))):
        feature_ids = features_per_example[example_idx]
        
        min_null_prediction = None
        prelim_predictions = []
        
        # 특정 example에 대한 모든 feature를 생성 (한 example에는 여러 feature가 존재할 수 있음)
        for feature_idx in feature_ids:
            start_logits = all_start_logits[feature_idx]    
            end_logits = all_end_logits[feature_idx]
            
            # logit과 original context의 logit과의 mapping을 진행
            offset_mapping = features[feature_idx]['offset_mapping']
            
            # `token_is_max_context`이 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거
            token_is_max_context = features[feature_idx].get(
                'token_is_max_context', None
            )
            
            # minimum null predictions을 업데이트
            feature_null_score = start_logits[0] + end_logits[0]    # ??
            if min_null_prediction is None or min_null_prediction['score'] > feature_null_score:
                min_null_prediction = {
                    'offsets': (0, 0),
                    'score': feature_null_score,
                    'start_logit': start_logits[0],
                    'end_logit': end_logits[0]
                }
                
            # `n_best_size`개수만큼의 start logits, end logits를 살핌 (높은 logit 순으로)
            start_ids = np.argsort(start_logits)[-1:-n_best_size-1:-1].tolist()
            end_ids = np.argsort(end_logits)[-1:-n_best_size-1:-1].tolist()
            
            for start_idx in start_ids:
                for end_idx in end_ids:
                    # out-of-scope answer는 고려하지 않는다.
                    if (
                        start_idx >= len(offset_mapping) 
                        or end_idx >= len(offset_mapping) 
                        or offset_mapping[start_idx] is None
                        or offset_mapping[end_idx] is None
                    ):
                        continue
                        
                    # start_idx, end_idx가 0보다 작거나 max_answer_length보다 큰 경우도 고려하지 않는다.
                    if (end_idx < start_idx or end_idx-start_idx+1 > max_answer_length):
                        continue
                        
                    # 최대 context가 없는 answer도 고려하지 않는다.
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_idx, False))
                    ):
                        continue
                    
                    prelim_predictions.append({
                        'offsets': (
                            offset_mapping[start_idx][0],
                            offset_mapping[end_idx][1],
                        ),
                        'score': start_logits[start_idx] + end_logits[end_idx],
                        'start_logit': start_logits[start_idx],
                        'end_logit': end_logits[end_idx]
                    })
            
            if version_2_with_negative:
                # minimum null prediction을 추가한다.
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction['score']
                
            # 가장 좋은 `n_best_size` predictions만 유지 (`start_logit + end_logit`이 큰 것 순으로 정렬)
            predictions = sorted(
                prelim_predictions, key=lambda x: x['score'], reverse=True
            )[:n_best_size]
            
            # 낮은 점수로 인해 제거된 경우 minimum null prediction을 다시 추가
            if version_2_with_negative and not any(p['offsets'] == (0, 0) for p in predictions):
                predictions.append(min_null_prediction)
                
            # offset을 사용하여 original context에서 answer text를 수집
            context = example['context']
            for pred in predictions:
                offsets = pred.pop('offsets')
                pred['text'] = context[offsets[0]:offsets[1]]
            
            ''' rare edge case에는 null이 아닌 예측이 하나도 없으며 
            failure를 피하기 위해 fake prediction을 만듦 '''
            if (
                len(predictions) == 0 
                or (len(predictions) == 1 and predictions[0]['text'] == "")
            ):
                predictions.insert(
                    0, {'text': 'empty', 'start_logit': 0., 'end_logit': 0., 'score': 0.}
                )
            
            # 모든 점수의 softmax를 계산 (logit이기 때문에 softmax으로 계산해줘야함)
            scores = np.array([pred.pop('score') for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            
            # 예측값에 확률을 추가
            for prob, pred in zip(probs, predictions):
                pred['probability'] = prob
            
            # best prediction을 선택
            if not version_2_with_negative:
                all_predictions[example['id']] = predictions[0]['text']
            else:
                # error case: 비어 있지 않은 최상의 예측을 찾아야 함
                idx = 0
                while predictions[idx]['text'] == "":
                    idx += 1
                best_non_null_pred = predictions[idx]
                
                # threshold를 사용, null prediction을 비교한다.
                score_diff = (
                    null_score - best_non_null_pred['start_logit'] - best_non_null_pred['end_logit']
                )
                scores_diff_json[example['id']] = float(score_diff)
                
            # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
            all_nbest_json[example['id']] = [
                {
                    k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v)
                    for k, v in pred.items()
                }
                for pred in predictions
            ]
        
        # output_dir이 있을 경우 모든 dict 저장
        if output_dir is not None:
            assert os.path.isdir(output_dir), f"{output_dir} is not a directory."
            
            prediction_file = os.path.join(
                output_dir, 'predictions.json' if prefix is None else f'predictions_{prefix}'.json
            )
            nbest_file = os.path.join(
                output_dir, 'nbest_predictions.json' if prefix is None else f'nbest_predictions_{prefix}.json'
            )
            
            if version_2_with_negative:
                null_odds_file = os.path.join(
                    output_dir, 'null_odds.json' if prefix is None else f'null_odds_{prefix}'.json
                )
                
            with open(prediction_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + '\n')
                
            with open(nbest_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) +'\n')
            
            if version_2_with_negative:
                with open(null_odds_file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + '\n')
                    
    return all_predictions


def post_processing_function(
    examples: Dataset, 
    features: Dataset, 
    predictions: Tuple[np.ndarray, np.ndarray], 
    training_args: MyTrainingArguments, 
    data_args: DataTrainingArguments
):
    ''' start logits, end ligits를 original context의 정답과 매치시키는 함수 '''
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        max_answer_length=data_args.max_answer_length,
        output_dir=training_args.output_dir
    )
    
    # metric를 구할 수 있게 Format을 맞추는 작업 수행
    formatted_predictions = [
        {'id': k, 'prediction_text': v} for k, v in predictions.items()
    ]
    
    if training_args.do_predict:
        return formatted_predictions

    elif training_args.do_eval:
        column_names = examples["validation"].column_names
        answer_column_name = "answers" if "answers" in column_names else column_names[2]    
        references = [
            {'id': ex['id'], 'answers': ex[answer_column_name]} for ex in examples
        ]
        return EvalPrediction(
            predictions=formatted_predictions, label_ids=references,
        )