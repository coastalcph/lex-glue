from torch import nn
from transformers import Trainer, EvalPrediction
from scipy.special import expit
from sklearn.metrics import f1_score, hamming_loss
from torch.utils.data import Sampler
from typing import Dict
import torch
from torch.utils.data import DataLoader, RandomSampler

class CurriculumLearningSampler(Sampler):
    def __init__(self, data_source, losses):
        super().__init__(data_source)
        self.data_source = data_source
        self.losses = losses

    def __iter__(self):
        # Sort indices by accuracy
        sorted_indices = sorted(range(len(self.data_source)), key=lambda i: -self.losses[i] if self.losses[i] is not None else float('-inf'))
        return iter(sorted_indices)

    def __len__(self):
        return len(self.data_source)
    
class CurriculumLearningMultilabelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 1  # Initialize epoch counter
        self.losses = [None] * len(self.train_dataset)

    def on_epoch_end(self):
        self.epoch += 1  # Increment epoch counter
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.encoder.config.num_labels),
                        labels.float().view(-1, self.model.encoder.config.num_labels))
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        # Create your sampler and DataLoader here
        if self.epoch < 4:
            # Pre-training phase, use default sampler
            sampler = RandomSampler(self.train_dataset)
        else:
            # Curriculum Learning phase, use custom sampler
            sampler = CurriculumLearningSampler(self.train_dataset, self.losses)
        def collate_fn(batch):
            keys_needed = ['input_ids_1', 'attention_mask_1', 'token_type_ids_1', 'input_ids_2', 'attention_mask_2', 'token_type_ids_2', 'input_ids_3', 'attention_mask_3', 'token_type_ids_3', 'labels']
            result = {}
            for key in keys_needed:
                items = [item[key] for item in batch]
                result[key] = torch.stack([torch.tensor(item) for item in items])
            return result


        dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, collate_fn=collate_fn,sampler=sampler)
        return dataloader


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.encoder.config.num_labels),
                        labels.float().view(-1, self.model.encoder.config.num_labels))
        return (loss, outputs) if return_outputs else loss
