import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer # base class from Hugging Face

# Focal loss implementation inheriting torch.nn.Module
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha # Balance positive/negative examples
        self.gamma = gamma # Reduce loss contribution from easy examples
        self.reduction = reduction # How final loss is reduced
    
    # Forward pass for Focal Loss, gets called during training
    # Input: model output predictions, targets: ground truth labels
    # Returns computed loss value
    def forward(self, inputs, targets):
        targets = targets.float()

        # Compute binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get models confidence in its predictions
        confidence = torch.exp(-bce_loss).clamp(min=1e-9)

        # Apply alpha and gamma to the loss
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - confidence) ** self.gamma * bce_loss

        if self.reduction == 'mean': # Average over all samples
            return loss.mean()
        elif self.reduction == 'sum': # Total loss
            return loss.sum()
        else: # Per-sample loss
            return loss

# Custom trainer using focal loss, inheriting Hugging Face trainer
class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_loss_alpha=0.25, focal_loss_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = FocalLoss(
            alpha=focal_loss_alpha,
            gamma=focal_loss_gamma,
            reduction='mean' # For compatibility with HFT
        )

    # Override compute_loss method to use focal loss
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        # Extract labels from inputs
        labels = inputs.pop("labels")

        # Forward pass through the model
        outputs = model(**inputs)

        # Extract logits from outputs
        # If model returns a dict, get logits from it else get from tuple
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        # Compute the loss using focal loss function
        loss = self.loss_fct(logits, labels.to(logits.device).float())
        return (loss, outputs) if return_outputs else loss