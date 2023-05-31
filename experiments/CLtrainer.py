from torch import nn
from transformers import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

    
class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.encoder.config.num_labels),
                        labels.float().view(-1, self.model.encoder.config.num_labels))
        return (loss, outputs) if return_outputs else loss




