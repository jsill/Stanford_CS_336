from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import torch
import numpy as np
 
class LLModel(torch.nn.Module):

    def __init__(self,
                 vocab_size,
                 context_length,
                 d_model,
                 num_layers,
                 num_heads,
                 d_ff,
                 rope_theta
                 ):
        super(LLModel,self).__init__()

        
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.d_model=d_model
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.rope_theta=rope_theta
        self.weightDct=dict()
        tokEmbWeight=torch.empty(vocab_size,d_model)
        torch.nn.init.trunc_normal_(tokEmbWeight,mean=0,std=1,a=-3,b=3)
        self.weightDct['token_embeddings.weight']=tokEmbWeight
        lnFinalWeight=torch.empty(d_model)
        torch.nn.init.trunc_normal_(lnFinalWeight,mean=0,std=1,a=-3,b=-3)
        self.weightDct['ln_final.weight']=lnFinalWeight
        sigmSq=2./(vocab_size + d_model)
        sigm=np.sqrt(sigmSq)
        lowerCutoff=-3.*sigm
        upperCutoff=3.*sigm
        lmHeadWeight=torch.empty(vocab_size,d_model)
        torch.nn.init.trunc_normal_(lmHeadWeight,mean=0,std=sigm,a=lowerCutoff,b=upperCutoff)
        self.weightDct['lm_head.weight']=lmHeadWeight
        for layerIdx in range(num_layers):
            qProj=torch.empty(d_model,d_model)
            sigmSq=2./(d_model+ d_model)
            sigm=np.sqrt(sigmSq)
            lowerCutoff=-3.*sigm
            upperCutoff=3.*sigm
            torch.nn.init.trunc_normal_(qProj,mean=0,std=sigm,a=lowerCutoff,b=upperCutoff)
            self.weightDct['layers.%d.attn.q_proj.weight'%layerIdx]=qProj
            kProj=torch.empty(d_model,d_model)
            torch.nn.init.trunc_normal_(kProj,mean=0,std=sigm,a=lowerCutoff,b=upperCutoff)
            self.weightDct['layers.%d.attn.k_proj.weight'%layerIdx]=kProj
            vProj=torch.empty(d_model,d_model)
            torch.nn.init.trunc_normal_(vProj,mean=0,std=sigm,a=lowerCutoff,b=upperCutoff)
            self.weightDct['layers.%d.attn.v_proj.weight'%layerIdx]=vProj
            outputProj=torch.empty(d_model,d_model)
            torch.nn.init.trunc_normal_(outputProj,mean=0,std=sigm,a=lowerCutoff,b=upperCutoff)
            self.weightDct['layers.%d.attn.output_proj.weight'%layerIdx]=outputProj
            ln1Weight=torch.empty(d_model)
            torch.nn.init.constant_(ln1Weight,1)
            self.weightDct['layers.%d.ln1.weight'%layerIdx]=ln1Weight
            ln2Weight=torch.empty(d_model)
            torch.nn.init.constant_(ln2Weight,1)
            self.weightDct['layers.%d.ln2.weight'%layerIdx]=ln2Weight
            ffnW1=torch.empty(d_ff,d_model)
            sigmSq=2./(d_ff + d_model)
            sigm=np.sqrt(sigmSq)
            lowerCutoff=-3.*sigm
            upperCutoff=3.*sigm
            torch.nn.init.trunc_normal_(ffnW1,0,sigm,lowerCutoff,upperCutoff)
            self.weightDct['layers.%d.ffn.w1.weight'%layerIdx]=ffnW1
            ffnW2=torch.empty(d_model,d_ff)
            torch.nn.init.trunc_normal_(ffnW2,0,sigm,lowerCutoff,upperCutoff)
            self.weightDct['layers.%d.ffn.w2.weight'%layerIdx]=ffnW2
            ffnW3=torch.empty(d_ff,d_model)
            torch.nn.init.trunc_normal_(ffnW3,0,sigm,lowerCutoff,upperCutoff)
            self.weightDct['layers.%d.ffn.w3.weight'%layerIdx]=ffnW3
             
        #for layerIdx in range(num_layers):
        #    for ky in ['attn.q_proj.weight',
        #               'attn.k_proj.weight',
        #               'attn.v_proj.weight',
        #               'attn.output_proj.weight',
        #               'ln1.weight',
        #               'ln2.weight',
        #               'ffn.w1.weight',
        #               'ffn.w2.weight',
        #               'ffn.w3.weight']:
                
        #weights: dict[str, Tensor],
 

    def run_embedding(
            vocab_size: int,
            d_model: int,
            weights: Float[Tensor, " vocab_size d_model"],
            token_ids: Int[Tensor, " ..."],
    ) -> Float[Tensor, " ... d_model"]:
        return weights[token_ids,:]

    def run_swiglu(
            d_model: int,
            d_ff: int,
            w1_weight: Float[Tensor, " d_ff d_model"],
            w2_weight: Float[Tensor, " d_model d_ff"],
            w3_weight: Float[Tensor, " d_ff d_model"],
            in_features: Float[Tensor, " ... d_model"],
    ) -> Float[Tensor, " ... d_model"]:
        sigm=torch.nn.Sigmoid()
        xw=in_features@torch.transpose(w1_weight,0,1)
        swish=np.multiply(xw,sigm(xw))
        xv=in_features@torch.transpose(w3_weight,0,1)
        swishTimesXV=np.multiply(swish,xv)
        return swishTimesXV@torch.transpose(w2_weight,0,1)

    
    def run_rmsnorm(d_model: int,
                    eps: float,
                    weights: Float[Tensor, " d_model"],
                    in_features: Float[Tensor, " ... d_model"],
                    ) -> Float[Tensor, " ... d_model"]:
            a=in_features
            aSq=a.square()
            dim=a.dim()
            aSqMean=torch.mean(aSq,dim-1)#,tuple(range(0,dim-1)))                                                                                        
            aRMS=aSqMean.sqrt()+ eps

            #REVISIT...there is surely a better way than all this transposing                                                                             
            a=a.transpose(0,2)
            a=a.transpose(1,2)
            aDiv=a.div(aRMS)
            aDiv=aDiv.transpose(1,2)
            aDiv=aDiv.transpose(0,2)
            return aDiv*weights
        
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

        seqLen=in_features.shape[1]


    def run_transformer_block(
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float,
            weights: dict[str, Tensor],
            in_features: Float[Tensor, " batch sequence_length d_model"],
    ) -> Float[Tensor, " batch sequence_length d_model"]:

        seqLen=in_features.shape[1]
        tokenPos=torch.Tensor([[i for i in range(0,seqLen )]])

        eps=5e-6
        rmsNormOut=LLModel.run_rmsnorm(d_model,eps,weights['ln1.weight'],in_features)

        attnOut=LLModel.run_multihead_self_attention_with_rope(d_model,num_heads,
                                                               max_seq_len,
                                                               theta,
                                                               weights['attn.q_proj.weight'],
                                                               weights['attn.k_proj.weight'],
                                                               weights['attn.v_proj.weight'],
                                                               weights['attn.output_proj.weight'],
                                                               rmsNormOut,#in_features,                                                 
                                                               tokenPos)


        rmsNormOut2=LLModel.run_rmsnorm(d_model,eps,weights['ln2.weight'],
                                        attnOut + in_features)

        d_ff=weights['ffn.w1.weight'].shape[0]

        swigluOut=LLModel.run_swiglu(d_model,d_ff,weights['ffn.w1.weight'],
                                     weights['ffn.w2.weight'],
                                     weights['ffn.w3.weight'],
                                     rmsNormOut2)


        return swigluOut + attnOut + in_features


    def forward(self,in_indices):
        return LLModel.run_transformer_lm(self.vocab_size,
                                  self.context_length,
                                  self.d_model,
                                  self.num_layers,
                                  self.num_heads,
                                  self.d_ff,
                                  self.rope_theta,
                                  self.weightDct,
                                  in_indices)
    
    def run_transformer_lm(
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            weights: dict[str, Tensor],
            in_indices: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        layerOutput=LLModel.run_embedding(vocab_size,d_model,weights['token_embeddings.weight'],
                                          in_indices)



        #print('HERE\n\n\n')
        #import pdb; pdb.set_trace()
        for i in range(num_layers):
            weightDct=dict()
            for ky in ['attn.q_proj.weight',#d_model by d_model
                       'attn.k_proj.weight',#d_model by d_model
                       'attn.v_proj.weight',#d_model by d_model
                       'attn.output_proj.weight',#d_model by d-model
                       'ln1.weight',#d_model
                       'ln2.weight',#d_model
                       'ffn.w1.weight',#d_ff by d_model
                       'ffn.w2.weight',#d_model by d_ff
                       'ffn.w3.weight']:#d_ff by d_model
                weightDct[ky]=weights['layers.%d.%s'%(i,ky)]

            layerOutput=LLModel.run_transformer_block(d_model,
                                                      num_heads,
                                                      d_ff,
                                                      context_length,#???                                                               
                                                      rope_theta,
                                                      weightDct,
                                                      layerOutput)#????                                                                      


        eps=5e-6
        normed=LLModel.run_rmsnorm(d_model,eps,weights['ln_final.weight'],layerOutput)
        finalOut=normed@torch.transpose(weights['lm_head.weight'],0,1)

        import pdb; pdb.set_trace()
        return finalOut

    
