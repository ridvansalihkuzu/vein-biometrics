"""
This code is generated by Ridvan Salih KUZU @UNIROMA3
LAST EDITED:  20.03.2020
ABOUT SCRIPT:
It is a script of loss function Orthagonality .
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import normalize

class OrthagonalityLoss(torch.nn.Module):
    """
            It is the implementation based on https://github.com/TAMU-VITA/Orthogonality-in-CNNs
            and https://arxiv.org/pdf/1810.09102.pdf
            """
    def __init__(self,batch_size):
        super(OrthagonalityLoss, self).__init__()
        self.batch_size=batch_size

    def forward(self,model):

        loss = None
        for W in model.parameters():
            if W.ndimension() < 2:
                continue
            else:
                cols = W[0].numel()
                rows = W.shape[0]
                w1 = W.view(-1, cols)
                wt = torch.transpose(w1, 0, 1)
                m = torch.matmul(wt, w1)
                ident = Variable(torch.eye(cols, cols))
                ident = ident.cuda()

                w_tmp = (m - ident)
                height = w_tmp.size(0)
                u = normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
                v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
                u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
                sigma = torch.dot(u, torch.matmul(w_tmp, v))

                if loss is None:
                    loss = (sigma**2)
                else:
                    loss = loss + (sigma**2)
        return loss/ (self.batch_size**2)