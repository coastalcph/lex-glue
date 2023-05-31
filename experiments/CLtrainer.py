from torch import nn
from transformers import Trainer
from scipy.special import expit
from sklearn.metrics import f1_score, hamming_loss
from torch.utils.data import Sampler

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
    
class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.encoder.config.num_labels),
                        labels.float().view(-1, self.model.encoder.config.num_labels))
        return (loss, outputs) if return_outputs else loss
    def compute_metrics(self, p: EvalPrediction) -> Dict:
        y_true = p.label_ids
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        y_preds = (expit(logits) > 0.5).astype('int32')

        # Compute your original metrics here
        macro_f1 = f1_score(y_true=y_true, y_pred=y_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true=y_true, y_pred=y_preds, average='micro', zero_division=0)

        # Compute Hamming Loss for each sample
        sample_losses = [hamming_loss(true, pred) for true, pred in zip(y_true, y_preds)]
        
        # Update losses
        for i, loss in enumerate(sample_losses):
            self.losses[i] = loss

        return {'macro-f1': macro_f1, 'micro-f1': micro_f1, 'sample_losses': sample_losses}
    def get_train_dataloader(self):
        # Create your sampler and DataLoader here
        if self.losses[0] is None:
            # Pre-training phase, use default sampler
            sampler = RandomSampler(self.train_dataset)
        else:
            # Curriculum Learning phase, use custom sampler
            sampler = CurriculumLearningSampler(self.train_dataset, self.losses)

        return DataLoader(self.train_dataset, sampler=sampler, batch_size=self.args.train_batch_size)


