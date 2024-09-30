import numpy as np
import os
import json
import matplotlib.pyplot as plt

from datasets.formatting.formatting import LazyBatch
from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report, ConfusionMatrixDisplay


def tokenize_essay(examples: LazyBatch, tokenizer: BertTokenizer, max_len: int) -> BatchEncoding:
    return tokenizer(examples['text'], padding='max_length', max_length=max_len, truncation=True)


def eval_classification_metrics(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    labels = np.argmax(labels, axis=-1)
    kappa = cohen_kappa_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {
        'f1': f1,
        'kappa': kappa
    }


def compute_evaluation_measures(y_true: list, y_pred: list, results_dict: dict):

    kappa = cohen_kappa_score(y_true, y_pred)

    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    results_dict['all_linear_kappa'].append(kappa)
    results_dict['all_qwk'].append(qwk)

    results_dict['all_accuracy'].append(report_dict['accuracy'])

    results_dict['all_macro_avg_p'].append(dict(report_dict['macro avg'])['precision'])
    results_dict['all_macro_avg_r'].append(dict(report_dict['macro avg'])['recall'])
    results_dict['all_macro_avg_f1'].append(dict(report_dict['macro avg'])['f1-score'])

    results_dict['all_weighted_avg_p'].append(dict(report_dict['weighted avg'])['precision'])
    results_dict['all_weighted_avg_r'].append(dict(report_dict['weighted avg'])['recall'])
    results_dict['all_weighted_avg_f1'].append(dict(report_dict['weighted avg'])['f1-score'])


def save_overall_eval_measures(all_trues: list, all_y_pred: list, results_dict: dict,
                               model_name: str, results_dir: str):

    all_trues = [int(g) for g in all_trues]
    all_y_pred = [int(p) for p in all_y_pred]

    new_results_dict = {}

    for measure_name, measure_values in results_dict.items():

        mean_label = measure_name.replace('all_', 'mean_')
        std_label = measure_name.replace('all_', 'std_')

        new_results_dict[mean_label] = np.mean(measure_values)
        new_results_dict[std_label] = np.std(measure_values)

    results_dict.update(new_results_dict)

    results_dict['all_trues'] = all_trues
    results_dict['all_predictions'] = all_y_pred

    regression_results_dir = os.path.join(results_dir, 'models_results')

    os.makedirs(regression_results_dir, exist_ok=True)

    results_file_path = os.path.join(regression_results_dir, f'{model_name}.json')

    with open(results_file_path, 'w') as fp:
        json.dump(results_dict, fp, indent=4)

    ConfusionMatrixDisplay.from_predictions(all_trues, all_y_pred)

    plt.ylabel('Nota Real')
    plt.xlabel('Nota Predita')

    img_dir = os.path.join(results_dir, 'confusion_matrix')

    os.makedirs(img_dir, exist_ok=True)

    confusion_matrix_name = f'{model_name}_confmatrix.pdf'

    confusion_matrix_name = confusion_matrix_name.lower()

    img_path = os.path.join(img_dir, confusion_matrix_name)

    plt.savefig(img_path, dpi=300)

    plt.clf()
