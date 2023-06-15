from torch import nn
from transformers import Trainer
from scipy.special import expit
from sklearn.metrics import hamming_loss
import torch
from torch.nn import CrossEntropyLoss

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.encoder.config.num_labels),
                        labels.float().view(-1, self.model.encoder.config.num_labels))
        return (loss, outputs) if return_outputs else loss


class MultiClassTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.encoder.config.num_labels), 
                        labels.view(-1))
        return (loss, outputs) if return_outputs else loss
