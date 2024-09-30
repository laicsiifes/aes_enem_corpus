import warnings
import os
import torch
import numpy as np
import torch.nn.functional as f
import time

from corpus_utils import load_dataset_into_dataframe
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from custom_trainer import CustomTrainerClassification
from bert_helper import (tokenize_essay, eval_classification_metrics, compute_evaluation_measures,
                         save_overall_eval_measures)


warnings.filterwarnings('ignore')

if __name__ == '__main__':

    our_corpus_directory = 'datasets/our_dataset'
    models_directory = 'models/'
    results_directory = 'results/'

    is_turn_off_computer = False

    competences = [
        # 'c1',
        # 'c2',
        # 'c3',
        # 'c4',
        'c5'
    ]

    len_dataset = -1

    N_FOLDS = 5

    BATCH_SIZE = 16
    MAX_LEN = 512

    NUM_EPOCHS = 10

    OPTIM = 'adamw_torch'

    use_fp16 = False

    # model_name = 'distil_bertimbau'
    # model_name = 'bertimbau_base'
    # model_name = 'roberta_pt_br'
    # model_name = 'albertina_100m'
    model_name = 'bertimbau_large'
    # model_name = 'albertina_900m'

    if model_name == 'distil_bertimbau':
        MODEL_PATH = 'adalbertojunior/distilbert-portuguese-cased'
    elif model_name == 'bertimbau_base':
        MODEL_PATH = 'neuralmind/bert-base-portuguese-cased'
    elif model_name == 'bertimbau_large':
        MODEL_PATH = 'neuralmind/bert-large-portuguese-cased'
        BATCH_SIZE = 8
    elif model_name == 'roberta_pt_br':
        MODEL_PATH = 'josu/roberta-pt-br'
    elif model_name == 'albertina_100m':
        MODEL_PATH = 'PORTULAN/albertina-100m-portuguese-ptbr-encoder'
        BATCH_SIZE = 8
        use_fp16 = True
    elif model_name == 'albertina_900m':
        MODEL_PATH = 'PORTULAN/albertina-900m-portuguese-ptbr-encoder'
        BATCH_SIZE = 8
    else:
        print('\n\nERRO. MODEL_NAME invalid!')
        exit(-1)

    print(f'\nModel {model_name} -- {MODEL_PATH}')

    dataset_df = load_dataset_into_dataframe(our_corpus_directory)

    original_dataset_df = dataset_df.copy()

    print(f'\nTotal Essays: {len(original_dataset_df)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device} - Num Epochs: {NUM_EPOCHS} - Batch Size: {BATCH_SIZE}')

    print('\nRunning Experiment')

    for competence in competences:

        print(f'\n\tCompetence: {competence}')

        if len_dataset > 0:
            dataset_df = original_dataset_df[['essay', competence]].head(len_dataset)

        essays = dataset_df['essay'].values
        grades = dataset_df[competence].values

        label_encoder = LabelEncoder()

        grades = label_encoder.fit_transform(grades)

        num_classes = len(set(grades))

        stratified_kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        models_competence_dir = os.path.join(models_directory, competence, model_name, f'{NUM_EPOCHS}')
        results_competence_dir = os.path.join(results_directory, competence, model_name, f'{NUM_EPOCHS}')

        if os.path.exists(results_competence_dir):
            continue

        results_dict = {
            'all_linear_kappa': [],
            'all_qwk': [],
            'all_accuracy': [],
            'all_macro_avg_p': [],
            'all_macro_avg_r': [],
            'all_macro_avg_f1': [],
            'all_weighted_avg_p': [],
            'all_weighted_avg_r': [],
            'all_weighted_avg_f1': []
        }

        all_trues = []
        all_y_pred = []

        for num_fold, (train_index, test_index) in enumerate(stratified_kfold.split(essays, grades), start=1):

            print(f'\n\t\tFold {num_fold}')

            X_train = [item for i, item in enumerate(essays) if i in train_index]
            X_test = [item for i, item in enumerate(essays) if i in test_index]

            y_train = grades[train_index]
            y_test = grades[test_index]

            class_categories = np.unique(y_train)

            class_weights_balanced = compute_class_weight(class_weight='balanced', classes=class_categories,
                                                          y=y_train)

            y_train = torch.tensor(y_train)
            y_test = torch.tensor(y_test)

            y_train = f.one_hot(y_train.to(torch.int64), num_classes=num_classes)
            y_test = f.one_hot(y_test.to(torch.int64), num_classes=num_classes)

            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

            print(f'\n\t\t\tTrain {len(X_train)} -- Validation {len(X_valid)} -- Test {len(X_test)}')

            train = {'text': X_train, 'label': y_train}
            valid = {'text': X_valid, 'label': y_valid}
            test = {'text': X_test, 'label': y_test}

            train_dataset = Dataset.from_dict(train)
            valid_dataset = Dataset.from_dict(valid)
            test_dataset = Dataset.from_dict(test)

            # Carregar Modelo pré-treinado e Tokenizador

            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

            # Tokenizar

            encoded_train_dataset = train_dataset.map(
                lambda x: tokenize_essay(x, tokenizer, MAX_LEN),
                batched=True, batch_size=BATCH_SIZE)

            encoded_valid_dataset = valid_dataset.map(
                lambda x: tokenize_essay(x, tokenizer, MAX_LEN),
                batched=True, batch_size=BATCH_SIZE)

            encoded_test_dataset = test_dataset.map(
                lambda x: tokenize_essay(x, tokenizer, MAX_LEN),
                batched=True, batch_size=BATCH_SIZE)

            output_dir = os.path.join(models_competence_dir, f'folder_{num_fold}', 'training')

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

            best_model_dir = os.path.join(models_competence_dir, f'folder_{num_fold}', 'best_model')

            if os.path.exists(best_model_dir):
                model = AutoModelForSequenceClassification.from_pretrained(best_model_dir, num_labels=num_classes)
            else:
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

            if not os.path.exists(best_model_dir):
                if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
                    trainer.train()
                else:
                    trainer.train(resume_from_checkpoint=True)

                trainer.save_model(best_model_dir)

            y_pred, _, _ = trainer.predict(encoded_test_dataset)

            y_pred = np.argmax(y_pred, axis=-1)
            y_test = np.argmax(y_test, axis=-1)

            y_test = label_encoder.inverse_transform(y_test)
            y_pred = label_encoder.inverse_transform(y_pred)

            compute_evaluation_measures(y_test, y_pred, results_dict)

            print(f'\n\t\t\t\tResults: {results_dict}')

            all_trues.extend(y_test)
            all_y_pred.extend(y_pred)

        os.makedirs(results_competence_dir, exist_ok=True)

        save_overall_eval_measures(all_trues, all_y_pred, results_dict, model_name, results_competence_dir)

        time.sleep(30)

    if is_turn_off_computer:

        print('\n\nTurning off computer ....')

        time.sleep(60)

        os.system('shutdown -h now')
