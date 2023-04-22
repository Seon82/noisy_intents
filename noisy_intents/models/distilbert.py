import torch
from transformers import DistilBertModel


class DistilBERT(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.1, load_pretrained=True):
        super().__init__()
        if load_pretrained:
            self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        else:
            self.l1 = DistilBertModel()
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, x):
        output_1 = self.l1(input_ids=x["ids"], attention_mask=x["mask"])
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
