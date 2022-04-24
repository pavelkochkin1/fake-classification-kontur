from transformers import AutoConfig, AutoModel
import torch.nn as nn

class BertForSequenceClassification(nn.Module):
    """
    [SkolkovoInstitute/russian_toxicity_classifier] model ready for fine-tuning
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int = None,
        dropout: float = 0.5,
    ):
        """
        Args:
            pretrained_model_name: str - HuggingFace model name.
            num_classes: int - the number of class labels
            dropout: float - the number of share of dropout
        """
        super().__init__()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, 
            num_labels = num_classes,
        )

        self.model = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
            self, 
            features, 
            attention_mask = None, 
            head_mask = None,
        ):
        """Computes class scores for the input sequence.
        Args:
            features: torch.Tensor - ids of each token,
                size ([bs, seq_length]
            attention_mask: torch.Tensor) - binary tensor, used to select
                tokens which are used to compute attention scores
                in the self-attention heads, size [bs, seq_length]
            head_mask (torch.Tensor): 1.0 in head_mask indicates that
                we keep the head, size: [num_heads]
                or [num_hidden_layers x num_heads]
        Returns:
            torch.Tensor with predicted class scores
        """

        assert attention_mask is not None, "attention mask is none!"

        bert_output = self.model(
            input_ids=features,
            attention_mask=attention_mask,
            head_mask=head_mask
        )
        
        # we only need the hidden state here (index 0)
        seq_output = bert_output[0] # (bs, seq_len, dim)

        # getting average representation of all tokens
        pooled_output = seq_output.mean(axis=1)  # (bs, dim)
        pooled_output = self.dropout(pooled_output) # (bs, dim)
        pooled_output = self.fc(pooled_output) # (bs, dim)
        pooled_output = self.dropout(pooled_output) # (bs, dim)
        scores = self.classifier(pooled_output) # (bs, num_classes)

        return scores