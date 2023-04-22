import torch
from transformers import BertModel


class BERT(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.1, load_pretrained=True):
        super().__init__()
        if load_pretrained:
            self.l1 = BertModel.from_pretrained("bert-base-uncased")
        else:
            self.l1 = BertModel()
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 300)
        self.classifier = torch.nn.Linear(300, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        _, pooled_output = self.l1(input_ids=x["ids"], attention_mask=x["mask"], return_dict=False)
        x = self.dropout(pooled_output)
        x = self.linear(x)
        x = self.dropout(self.relu(x))
        output = self.classifier(x)
        return output
