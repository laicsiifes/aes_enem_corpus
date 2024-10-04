import warnings
import torch
import os
import numpy as np
import torch.nn.functional as f
import json
import matplotlib.pyplot as plt
import time

from load_datasets import (read_proposed_corpus, read_dataset_sourceAWithGraders, filter_common_essays)
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, EarlyStoppingCallback
from custom_trainer import CustomTrainerClassification
from bert_helper import tokenize_essay, eval_classification_metrics
from sklearn.metrics import cohen_kappa_score, classification_report, ConfusionMatrixDisplay


warnings.filterwarnings('ignore')

if __name__ == '__main__':

    our_corpus_directory = 'datasets/our_dataset'
    models_directory = 'data/models_interdatasets_v2/'
    results_directory = 'data/results_interdatasets_v2/'

    extended_essay_br_essays = 'datasets/extended_essay_br/extended_essay-br.csv'
    extended_essay_br_prompts = 'datasets/extended_essay_br/prompts.csv'

    is_turn_off_computer = False

    competences = [
        'c1',
        'c2',
        'c3',
        'c4',
        'c5'
    ]

    BATCH_SIZE = 16
    MAX_LEN = 512

    NUM_EPOCHS = 10

    OPTIM = 'adamw_torch'

    use_fp16 = False

    # test_dataset_name = 'our_corpus'
    test_dataset_name = 'new_benchmark'

    new_benchmark_essays = read_dataset_sourceAWithGraders()

    our_dataset_df = read_proposed_corpus(our_corpus_directory)

    if test_dataset_name == 'our_corpus':
        test_dataset_df = our_dataset_df
        train_dataset_df = new_benchmark_essays
    else:
        test_dataset_df = new_benchmark_essays
        train_dataset_df = our_dataset_df

    # Filtro usando similaridade entre os textos
    train_dataset_df = filter_common_essays(test_dataset_df, train_dataset_df)

    # model_name = 'distil_bertimbau'
    # model_name = 'bertimbau_base'
    model_name = 'bertimbau_large'

    if model_name == 'distil_bertimbau':
        MODEL_PATH = 'adalbertojunior/distilbert-portuguese-cased'
    elif model_name == 'bertimbau_base':
        MODEL_PATH = 'neuralmind/bert-base-portuguese-cased'
    elif model_name == 'bertimbau_large':
        MODEL_PATH = 'neuralmind/bert-large-portuguese-cased'
        BATCH_SIZE = 8
    else:
        print('\n\nERRO. MODEL_NAME invalid!')
        exit(-1)

    print(f'\nTest Dataset: {test_dataset_name}')

    print(f'\nModel {model_name} -- {MODEL_PATH}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device} - Num Epochs: {NUM_EPOCHS} - Batch Size: {BATCH_SIZE}')

    print('\nRunning Experiment')

    for competence in competences:

        print(f'\n\tCompetence: {competence.upper()}')

        models_competence_dir = os.path.join(models_directory, competence, model_name, f'{NUM_EPOCHS}')
        results_competence_dir = os.path.join(results_directory, competence, model_name, f'{NUM_EPOCHS}')

        results_file_path = os.path.join(results_competence_dir, f'{test_dataset_name}.json')

        if os.path.exists(results_file_path):
            continue

        train_essays = train_dataset_df['essay'].values
        train_grades = train_dataset_df[competence].values

        test_essays = test_dataset_df['essay'].values
        test_grades = test_dataset_df[competence].values

        train_essays, validation_essays, train_grades, validation_grades = train_test_split(
            train_essays, train_grades, test_size=0.10, stratify=train_grades, shuffle=True, random_state=42)

        print(f'\n\t\tTotal Essays -- Train: {len(train_essays)} -- Validation: {len(validation_essays)} '
              f'-- Test: {len(test_essays)}')

        print(f'\n\t\tTrain Grades Distributions: {Counter(train_grades)}')
        print(f'\t\tValidation Grades Distributions: {Counter(validation_grades)}')
        print(f'\t\tTest Grades Distributions: {Counter(test_grades)}')

        label_encoder = LabelEncoder()

        train_grades = label_encoder.fit_transform(train_grades)

        validation_grades = label_encoder.transform(validation_grades)
        test_grades = label_encoder.transform(test_grades)

        num_classes = len(set(train_grades))

        class_categories = np.unique(train_grades)

        class_weights_balanced = compute_class_weight(class_weight='balanced', classes=class_categories,
                                                      y=train_grades)

        train_grades = torch.tensor(train_grades)
        validation_grades = torch.tensor(validation_grades)
        test_grades = torch.tensor(test_grades)

        train_grades = f.one_hot(train_grades.to(torch.int64), num_classes=num_classes)
        validation_grades = f.one_hot(validation_grades.to(torch.int64), num_classes=num_classes)
        test_grades = f.one_hot(test_grades.to(torch.int64), num_classes=num_classes)

        train_dict = {'text': train_essays, 'label': train_grades}
        validation_dict = {'text': validation_essays, 'label': validation_grades}
        test_dict = {'text': test_essays, 'label': test_grades}

        train_ds = Dataset.from_dict(train_dict)
        validation_ds = Dataset.from_dict(validation_dict)
        test_ds = Dataset.from_dict(test_dict)

        # Carregar Modelo pré-treinado e Tokenizador

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # Tokenizar

        encoded_train_dataset = train_ds.map(
            lambda x: tokenize_essay(x, tokenizer, MAX_LEN),
            batched=True, batch_size=BATCH_SIZE)

        encoded_valid_dataset = validation_ds.map(
            lambda x: tokenize_essay(x, tokenizer, MAX_LEN),
            batched=True, batch_size=BATCH_SIZE)

        encoded_test_dataset = test_ds.map(
            lambda x: tokenize_essay(x, tokenizer, MAX_LEN),
            batched=True, batch_size=BATCH_SIZE)

        output_dir = os.path.join(models_competence_dir, f'{test_dataset_name }', 'training')

        logging_eval_steps = len(encoded_train_dataset) // BATCH_SIZE

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=1e-5,
            evaluation_strategy='epoch',
            logging_steps=logging_eval_steps,
            eval_steps=logging_eval_steps,
            optim=OPTIM,
            weight_decay=0.01,
            save_total_limit=1,
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='kappa',
            greater_is_better=True,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            fp16=use_fp16
        )

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_classes)

        trainer = CustomTrainerClassification(
            class_weight=class_weights_balanced,
            device=device, model=model,
            args=training_args,
            train_dataset=encoded_train_dataset,
            eval_dataset=encoded_valid_dataset,
            compute_metrics=eval_classification_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        best_model_dir = os.path.join(models_competence_dir, f'{test_dataset_name }', 'best_model')

        trainer.save_model(best_model_dir)

        y_pred, _, _ = trainer.predict(encoded_test_dataset)

        y_pred = np.argmax(y_pred, axis=-1)
        test_grades = np.argmax(test_grades, axis=-1)

        test_grades_test = label_encoder.inverse_transform(test_grades)
        y_pred = label_encoder.inverse_transform(y_pred)

        test_grades_test = list(int(g) for g in test_grades_test)
        y_pred = list(int(y) for y in y_pred)

        os.makedirs(results_competence_dir, exist_ok=True)

        kappa = cohen_kappa_score(test_grades_test, y_pred)

        qwk = cohen_kappa_score(test_grades_test, y_pred, weights='quadratic')

        results_dict = classification_report(test_grades_test, y_pred, zero_division=0, output_dict=True)

        results_dict['kappa'] = kappa
        results_dict['qwk'] = qwk

        print(f'\n\t\t\t\tResults: {results_dict}')

        with open(results_file_path, 'w') as fp:
            json.dump(results_dict, fp, indent=4)

        ConfusionMatrixDisplay.from_predictions(test_grades_test, y_pred)

        plt.ylabel('Nota Real')
        plt.xlabel('Nota Predita')

        confusion_matrix_name = f'{test_dataset_name}_confmatrix.pdf'

        confusion_matrix_name = confusion_matrix_name.lower()

        img_path = os.path.join(results_competence_dir, confusion_matrix_name)

        plt.savefig(img_path, dpi=300)

        plt.clf()

        time.sleep(30)
