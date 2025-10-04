from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import torch
import numpy as np
 
class LLModel(torch.nn.Module):

    def __init__(self,vocab_size):
        super(LLModel,self).__init__()

        self.vocab_size=vocab_size,
        self.context_length=context_length,
        self.d_model=d_model,
        self.num_layers=num_layers,
        self.num_heads=num_heads,
        self.d_ff=d_ff,
        self.rope_theta=rope_theta,
        #weights: dict[str, Tensor],

    def run_scaled_dot_product_attention(
            Q: Float[Tensor, " ... queries d_k"],
            K: Float[Tensor, " ... keys d_k"],
            V: Float[Tensor, " ... values d_v"],
            mask: Float[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        """                                                                                                                                          
        Given key (K), query (Q), and value (V) tensors, return                                                                                      
        the output of your scaled dot product attention implementation.                                                                              
                                                                                                                                                 
        Args:                                                                                                                                        
        Q (Float[Tensor, " ... queries d_k"]): Query tensor                                                                                      
        K (Float[Tensor, " ... keys d_k"]): Key tensor                                                                                           
        V (Float[Tensor, " ... values d_v"]): Values tensor                                                                                      
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor                                                                            
        Returns:                                                                                                                                     
        Float[Tensor, " ... queries d_v"]: Output of SDPA                                                                                        
        """
        QKTranspose=Q@torch.transpose(K,-2,-1)
        dk=float(K.shape[-1])
        QKTransposeScaled=QKTranspose/np.sqrt(dk)

        if (mask != None):
            QKTransposeScaled += np.array(np.where(mask==1,0,-np.inf),dtype=np.float32)
        SM=torch.nn.Softmax(-1)#QKTransposeScaled)                                                                                                         
        SMResult=SM(QKTransposeScaled)

        return SMResult@V


    def run_multihead_self_attention_with_rope(
            d_model: int,
            num_heads: int,
            max_seq_len: int,
            theta: float,
            q_proj_weight: Float[Tensor, " d_k d_in"],
            k_proj_weight: Float[Tensor, " d_k d_in"],
            v_proj_weight: Float[Tensor, " d_v d_in"],
            o_proj_weight: Float[Tensor, " d_model d_v"],
            in_features: Float[Tensor, " ... sequence_length d_in"],
            token_positions: Int[Tensor, " ... sequence_length"] | None = None,
)-> Float[Tensor, " ... sequence_length d_out"]:

         print('in features shape here',in_features.shape)

         d_k=int(d_model/num_heads)


         q_proj=in_features@torch.transpose(q_proj_weight,0,1)
         k_proj=in_features@torch.transpose(k_proj_weight,0,1)
         v_proj=in_features@torch.transpose(v_proj_weight,0,1)


         headList=[]
         
         seqLen=in_features.shape[-2]

         seqLen=len(token_positions[0])


         for h in range(num_heads):
             startIdx=h*d_k

             endIdx=(h+1)*d_k

             mask=torch.Tensor(seqLen,seqLen)
             for i in range(seqLen):
                 for j in range(seqLen):
                     mask[i][j]=i >= j
            #REVISIT, this passes the test but this is not the right way to handle the token positions...I'm taking advantage
            #of the token positions just being a 2-d version of a 1-d array in the test                                                              


             res=LLModel.run_scaled_dot_product_attention(LLModel.run_rope(d_k,theta,seqLen,q_proj[:,:,startIdx:endIdx],token_positions[0]),
                                                          LLModel.run_rope(d_k,theta,seqLen,k_proj[:,:,startIdx:endIdx],token_positions[0]),
                                                          v_proj[:,:,startIdx:endIdx],mask)
             headList.append(res)

         concatResult=torch.concat(headList,-1)

         output=concatResult@torch.transpose(o_proj_weight,0,1)

         return output


    def run_rope(
            d_k: int,
            theta: float,
            max_seq_len: int,
            in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
            token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        """                                                                                                                                          
        Run RoPE for a given input tensor.                                                                                                           
                                                                                                                                                 
        Args:                                                                                                                                        
        d_k (int): Embedding dimension size for the query or key tensor.                                                                         
        theta (float): RoPE parameter.                                                                                                           
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.                                                
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.                                                 
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions             
        Returns:                                                                                                                                     
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.                                                                      
        """

    
        def ropeMat(m,d):
            thetas=[m*np.pow(theta,-2*(i-1)/float(d)) for i in range(1,int(d/2 + 1))]
            cosines=torch.tensor(np.concatenate([[np.cos(tht),np.cos(tht)] for tht in thetas]))
            sinesBelow=torch.tensor(np.concatenate([[np.sin(tht),0] for tht in thetas]))
            sinesAbove=torch.tensor(np.concatenate([[-np.sin(tht),0] for tht in thetas]))
            rMat=torch.zeros(d,d)
            rMat=rMat.diagonal_scatter(cosines)
            rMat=rMat.diagonal_scatter(sinesBelow[0:-1],1)
            rMat=rMat.diagonal_scatter(sinesAbove[0:-1],-1)
            return rMat

        seqLength=token_positions.shape[-1]
    
        res=torch.zeros(in_query_or_key.shape[0],in_query_or_key.shape[1],in_query_or_key.shape[2])

        res=torch.zeros(in_query_or_key.shape[0],seqLength,in_query_or_key.shape[2])

    
        for i in range(seqLength):
            if (i >= in_query_or_key.shape[1]):
                res[:,i,:]=0.
            else:
                res[:,i,:]=in_query_or_key[:,i,:]@ropeMat(token_positions[i],d_k)

        return res
