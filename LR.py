import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score, precision_score
import numpy as np
class LogisticRegression(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(self.input_size, 1, bias = True)
        self.sigmoid = nn.Sigmoid()
        self.lossf = nn.CrossEntropyLoss(weight = torch.tensor([1., 3.], dtype = torch.float32))
        
    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out    
    
def train_loop(dataloader, model, optimizer):
    size = len(dataloader)
    model.train()
    train_loss = 0
    loss = torch.tensor(0, dtype = float, requires_grad= True) 
    for batch, (features, target) in enumerate(dataloader):  
        features = features.float()
        optimizer.zero_grad()
        pred = model(features)
        pred_new = torch.zeros([len(pred), 2], dtype = torch.float32)
        for i in range(len(pred)):
            pred_new[i] = torch.cat((torch.tensor([1- pred[i].item()], dtype = torch.float32), pred[i]), dim = 0)      
        loss = nn.functional.cross_entropy(pred_new, target.long())
        train_loss += loss
        loss.backward()
        optimizer.step()
    return train_loss / size    
def test_loop(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    pre = [0, 0]
    recall = [0,0]   
    with torch.no_grad():
        for features, target in dataloader:
            pred = model(features.float())
            pred_new = torch.zeros([len(pred), 2], dtype = torch.float32)
            for i in range(len(pred)):
                pred_new[i] = torch.cat((torch.tensor([1- pred[i].item()], dtype = torch.float32), pred[i]), dim = 0)            
            test_loss += model.lossf(pred_new, target.long())
            pred = pred.squeeze()            
            for i in range(len(pred)):
                if pred[i] <0.5:
                    pred[i] = 0
                else:
                    pred[i] = 1      
            correct = correct + ((pred == target).type(torch.float).sum().item()/ len(features))
            pre[0] += precision_score(target[pred==0].detach().numpy(),pred[pred ==0 ].detach().numpy(), average = "binary", pos_label= 0)
            recall[0] += recall_score(target[target ==0 ].detach().numpy(),pred[target==0].detach().numpy(),average = "binary", pos_label= 0)    
            pre[1] += precision_score(target[pred==1].detach().numpy(),pred[pred ==1 ].detach().numpy(), average = "binary", pos_label= 1)
            recall[1] += recall_score(target[target ==1 ].detach().numpy(),pred[target==1].detach().numpy(), average = "binary", pos_label= 1)                
        test_loss /=num_batches
        correct /= (num_batches)   
        pre = np.array(pre)
        pre /= num_batches
        recall = np.array(recall) 
        recall /= num_batches
    return test_loss, correct, pre, recall                