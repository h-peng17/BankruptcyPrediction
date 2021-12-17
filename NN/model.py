import pdb 
import math 

import torch
import torch.nn as nn 

from loss import BinaryFocalLossWithLogits


class MLPForBF(nn.Module):
    def __init__(self, args, weight) -> None:
        super(MLPForBF, self).__init__()
        self.args = args 
        self.mlp1 = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        if args.loss_fn == "sigmoid":
            self.fc = nn.Linear(256, 1)
            self.pos_weight = torch.tensor(math.sqrt(weight[1]), dtype=torch.float32, device="cuda")
        elif args.loss_fn == "softmax":
            self.fc = nn.Linear(256, 2)
            self.weight = torch.tensor([1, math.sqrt(weight[1])], dtype=torch.float32, device="cuda")
        else:
            raise ValueError
    
    def forward(self, x, y):
        x = self.mlp1(x)
        x = self.mlp2(x) + x
        x = self.mlp3(x) + x
        if self.args.loss_fn == "sigmoid":
            logits = self.fc(x).squeeze(-1)
            # loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, y)
            if self.training:
                preds = torch.sigmoid(logits) > 0.5
            else:
                preds = torch.sigmoid(logits)
        else:
            logits = self.fc(x)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, y.to(torch.long))
            preds = torch.argmax(logits, dim=1)
        return loss, preds 


class MLPForFilling(nn.Module):
    def __init__(self, args, input_size) -> None:
        super(MLPForFilling, self).__init__()
        self.args = args 
        self.mlp1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc = nn.Linear(256, 1)
    
    def forward(self, x, y=None):
        x = self.mlp1(x)
        x = self.mlp2(x) + x
        x = self.mlp3(x) + x
        logits = self.fc(x).squeeze(-1)
        loss_fn = nn.MSELoss()
        if y is not None:
            loss = loss_fn(logits, y)
            return loss 
        else:
            return logits












