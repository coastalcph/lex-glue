from torch import nn
from transformers import Trainer
from scipy.special import expit
from sklearn.metrics import hamming_loss
import torch

class CurriculumLearningTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = [0] * len(self.train_dataset)
        self.global_sample_index = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.encoder.config.num_labels),
                        labels.float().view(-1, self.model.encoder.config.num_labels))
        
        # Compute Hamming Loss for each sample
        y_preds = (torch.sigmoid(logits) > 0.5).int()
        y_true = labels
        sample_losses = [(true - pred).float().abs().mean().item() for true, pred in zip(y_true, y_preds)]
        #total_batch_size = self.args.per_device_train_batch_size * torch.cuda.device_count()
        # Update losses with global index
        for i, hm_loss in enumerate(sample_losses):
            idx=(self.global_sample_index + i )% len(self.train_dataset)            
            self.losses[idx] = hm_loss

        self.global_sample_index += len(sample_losses)
        self.global_sample_index = self.global_sample_index % len(self.train_dataset)
        return (loss, outputs) if return_outputs else loss



class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.encoder.config.num_labels),
                        labels.float().view(-1, self.model.encoder.config.num_labels))
        return (loss, outputs) if return_outputs else loss
