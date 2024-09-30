import torch

from torch import nn
from transformers import Trainer


class CustomTrainerClassification(Trainer):

    def __init__(self, class_weight, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight
        self.device = device

    def compute_loss(self, model, inputs: dict, return_outputs: bool = False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        weights = torch.tensor(self.class_weight).to(self.device)
        loss_function = nn.CrossEntropyLoss(weight=weights)
        loss = loss_function(logits, labels)
        return (loss, outputs) if return_outputs else loss
