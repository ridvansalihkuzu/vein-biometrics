import torch

class DecovLoss(torch.nn.Module):

    def forward(self,input,reduce='half_squared_sum'):
        return Decov.apply(input,reduce)

class Decov(torch.autograd.Function):
    """DeCov loss (https://arxiv.org/abs/1511.06068)"""

    @staticmethod
    def forward(ctx,inputs,reduce='half_squared_sum'):

        h = inputs.cuda()
        length=h.size()[0]
        mean_h = torch.mean(h, 0).expand_as(h).cuda()
        std_h=torch.std(h,0).expand_as(h)
        #mean_h = torch.mean(h, 1).unsqueeze_(-1).expand_as(h)
        h_centered = (h - mean_h).div(std_h) # we add with Gabriel
        covariance= h_centered.t().matmul(h_centered).squeeze()
        covariance = covariance.mul((torch.ones(covariance.size()).cuda() - torch.eye(covariance.size()[0]).cuda()))
        covariance/=length
        ctx.save_for_backward(covariance,h_centered)
        if reduce == 'half_squared_sum':
            cost = 0.5 * (torch.pow(torch.norm(covariance,keepdim=False), 2))
            return cost/length
        else:
            return covariance

    @staticmethod
    def backward(ctx, grad_outputs,reduce='half_squared_sum'):
        covariance, h_centered=ctx.saved_tensors
        length=h_centered.size()[0]
        gcost = grad_outputs
        gcost_div_n = gcost / length
        if reduce == 'half_squared_sum':
            gh = 2.0 * h_centered.matmul(covariance)
            gh *= gcost_div_n
        else:
            gcost_div_n = gcost_div_n.mul(
                (torch.ones(gcost_div_n.size()) - torch.eye(gcost_div_n.size()[0])).cuda())
            gh = h_centered.dot(gcost_div_n + gcost_div_n.t())
        return gh,None
