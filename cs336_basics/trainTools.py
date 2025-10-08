import torch
from torch import Tensor
import os
from typing import IO, Any, BinaryIO
import numpy as np
from collections.abc import Iterable
from jaxtyping import Float, Int
import numpy.typing as npt

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | BinaryIO | IO[bytes],
                    ):

   
   dct={'model_state':model.state_dict(),
        'optimizer_state':optimizer.state_dict(),
        'iteration':iteration}
   torch.save(dct,out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    ):
    
   dct=torch.load(src)

   model.load_state_dict(dct['model_state'])
   optimizer.load_state_dict(dct['optimizer_state'])
   return dct['iteration']


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
 
    l2_norm=0
    sumSq=0.
    for p in parameters:
        #print(p.grad)                                                                                                                                       
        if (p.grad != None):
            sq=p.grad*p.grad
            sumSq=sumSq + sq.nansum()
    l2_norm=np.sqrt(sumSq)

    if (l2_norm > max_l2_norm):
        scale=max_l2_norm/l2_norm
        for p in parameters:
            if (p.grad != None):
                p.grad=scale*p.grad

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:

    maxes=torch.max(in_features,dim)[0]
    in_features=torch.transpose(torch.transpose(in_features,0,dim)-maxes,dim,0)
    in_features_exp=torch.exp(in_features)
    sm=torch.sum(in_features_exp,dim)
    return torch.transpose(torch.div(torch.transpose(in_features_exp,0,dim),sm),dim,0)

def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy                                                                               
    loss across examples.                                                                                                                                    
                                                                                                                                                             
    Args:                                                                                                                                                    
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the                                                                                 
            unnormalized logit of jth class for the ith example.                                                                                             
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.                                              
            Each value must be between 0 and `num_classes - 1`.                                                                                              
                                                                                                                                                             
    Returns:                                                                                                                                                 
        Float[Tensor, ""]: The average cross-entropy loss across examples.                                                                                   
    """
    batchSize=inputs.shape[0]

    softm=run_softmax(inputs,1)


    def safeLog(p):

        if (p ==0):
            #I had to find this by bisection...I wish I understood why this is correct, TODO, revisit                                                        
            pIn=1.675e-184
        else:
            pIn=p
        pInAr=np.array([pIn],dtype=np.float64)
        #print('pInAr is ',pInAr)                                                                                                                            
        return np.log(pInAr[0])

    if (len(targets.shape)==1):
       softMaxMat=torch.Tensor([softm[i][targets[i]] for i in range(batchSize)])
       return -torch.Tensor([np.mean([safeLog(softMaxMat[i]) for i in range(softMaxMat.shape[0])])])
    elif (len(targets.shape)==2):
       softMaxMat=torch.zeros(targets.shape)
       for i in range(targets.shape[0]):
          for j in range(targets.shape[1]):
             softMaxMat[i][j]=softm[i][j][targets[i][j]]
       return -torch.Tensor([np.mean([safeLog(softMaxMat[i][j]) for i in range(softMaxMat.shape[0]) for j in range(softMaxMat.shape[1]) ])])
    else:
       raise Exception('cross entropy only implemented for 1 and 2 dimensional target tensor')

    #import pdb; pdb.set_trace()
    #return -torch.Tensor([safeLog(softMaxMat).mean()])
    #return -torch.Tensor([np.mean([safeLog(softMaxMat[i]) for i in range(softMaxMat.shape[0])])])


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
 
    def sigmoid(x):
        return 1./(1. + torch.exp(-x))
    return in_features*sigmoid(in_features)

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int,
              device: str) -> tuple[torch.Tensor, torch.Tensor]:

   numTok=len(dataset)

   maxStart=numTok - context_length -1
   
   samples=torch.zeros(batch_size,context_length,dtype=torch.int64,device=device)
   labels=torch.zeros(batch_size,context_length,dtype=torch.int64,device=device)
   for bi in range(batch_size):
      strt=np.random.randint(maxStart + 1)
      samples[bi,:]=torch.from_numpy(dataset[strt:strt + context_length])#,dtype=torch.int64)                                                              
      labels[bi]=torch.from_numpy(dataset[strt + 1:strt + 1 + context_length])
   return (samples,labels)


def get_lr_cosine_schedule(
      it: int,
      max_learning_rate: float,
      min_learning_rate: float,
      warmup_iters: int,
      cosine_cycle_iters: int,
):
   warmupStep=max_learning_rate/warmup_iters
   total_iters=cosine_cycle_iters
   
   if (it <= warmup_iters):
      return it*warmupStep
   elif (it > total_iters):
      return min_learning_rate

   return min_learning_rate + 0.5*(max_learning_rate-min_learning_rate)*( 1+ np.cos(np.pi*(it - warmup_iters)/(cosine_cycle_iters-warmup_iters)))


class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr,weight_decay,betas,eps):
        super(AdamW,self).__init__(params,dict(lr=lr,weight_decay=weight_decay,betas=betas,eps=eps))

    def step(self):
        for group in self.param_groups:
            lr=group['lr']
            lambd=group['weight_decay']
            betas=group['betas']
            beta1=betas[0]
            beta2=betas[1]
            eps=group['eps']

            for p in group['params']:

                state=self.state[p]

                step_count=state.get('step_count',1)

                grad=p.grad.data

                g=grad #+ lambd*p.data                                                                                                           

                mom=state.get('mom',torch.zeros_like(grad))
                mom=beta1*mom + (1-beta1)*g

                state['mom']=mom
                v=state.get('v',torch.zeros_like(grad))


                #grad2=state.get('grad2',torch.zeros_like(grad))                                                                                 
                #grad2 += torch.square(grad)                                                                                                     

                g2=torch.square(g)
                v=beta2*v + (1-beta2)*g2

                state['v']=v

                momhat=mom/(1-np.power(beta1,step_count))
                vhat=v/(1 - np.power(beta2,step_count))
                state['step_count']=step_count + 1

                eta=1.
                p.data = p.data - eta*(lr*(momhat/(torch.sqrt(vhat) + eps) + lambd*p.data ))



                
   

   
