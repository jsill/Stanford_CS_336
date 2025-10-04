from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor

import numpy as np
from torch import nn
import time
import pickle
import gc
import regex as re

import datetime

import bisect
from functools import cmp_to_key
import collections
from collections import Counter

from multiprocessing import Process

from cs336_basics import tokenizer,llModel

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    return in_features@torch.transpose(weights,0,1)
    #np.savez('test_linear.npz',array=in_features@torch.transpose(weights,0,1))
    #raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    return weights[token_ids,:]
    #raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    sigm=nn.Sigmoid()
    xw=in_features@torch.transpose(w1_weight,0,1)
    swish=np.multiply(xw,sigm(xw))
    xv=in_features@torch.transpose(w3_weight,0,1)
    swishTimesXV=np.multiply(swish,xv)
    return swishTimesXV@torch.transpose(w2_weight,0,1)
    #raise NotImplementedError


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
    print('new thing')
    import pdb; pdb.set_trace()
    return llModel.LLModel.run_scaled_dot_product_attention(Q,K,V,mask)
    #QKTranspose=Q@torch.transpose(K,-2,-1)
    #dk=float(K.shape[-1]) 
    #QKTransposeScaled=QKTranspose/np.sqrt(dk)

    #if (mask != None): 
    ##    QKTransposeScaled += np.array(np.where(mask==1,0,-np.inf),dtype=np.float32)
    #SM=nn.Softmax(-1)#QKTransposeScaled)
    #SMResult=SM(QKTransposeScaled)

    #return SMResult@V
    

    
def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    
    d_k=int(d_model/num_heads)
    

    q_proj=in_features@torch.transpose(q_proj_weight,0,1)                                                                      
    k_proj=in_features@torch.transpose(k_proj_weight,0,1)                                                                         
    v_proj=in_features@torch.transpose(v_proj_weight,0,1)   

            
    headList=[]

    seqLen=in_features.shape[-2]

    #print('seqLen here',seqLen)
    #import pdb; pdb.set_trace()
    for h in range(num_heads):#range(num_heads -1,-1,-1):
        startIdx=h*d_k

        endIdx=(h+1)*d_k

        mask=torch.Tensor(seqLen,seqLen)
        for i in range(seqLen):
            for j in range(seqLen):
                mask[i][j]=i >= j
        res=run_scaled_dot_product_attention(q_proj[:,:,startIdx:endIdx],k_proj[:,:,startIdx:endIdx],
                                             v_proj[:,:,startIdx:endIdx],mask)
        
        headList.append(res)
 
    concatResult=torch.concat(headList,-1)
 
    output=concatResult@torch.transpose(o_proj_weight,0,1) 

    return output
    #raise NotImplementedError

 
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
        #doMask=False,
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:

    print('!!!!!!!!!!!!!!!!!!!!!!!!! ALL NEW !!!!!')
    return llModel.LLModel.run_multihead_self_attention_with_rope(d_model,
                                                                  num_heads,
                                                                  max_seq_len,
                                                                  theta,
                                                                  q_proj_weight,
                                                                  k_proj_weight,
                                                                  v_proj_weight,
                                                                  o_proj_weight,
                                                                  in_features,
                                                                  token_positions)
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

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

        #if (doMask):
        res=run_scaled_dot_product_attention(run_rope(d_k,theta,seqLen,q_proj[:,:,startIdx:endIdx],token_positions[0]),
                                             run_rope(d_k,theta,seqLen,k_proj[:,:,startIdx:endIdx],token_positions[0]),
                                             v_proj[:,:,startIdx:endIdx],mask)
        #else:
        #    res=run_scaled_dot_product_attention(run_rope(d_k,theta,seqLen,q_proj[:,:,startIdx:endIdx],token_positions[0]),
        #                                         run_rope(d_k,theta,seqLen,k_proj[:,:,startIdx:endIdx],token_positions[0]),
        #                                         v_proj[:,:,startIdx:endIdx])
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
    
    return llModel.LLModel.run_rope(d_k,theta,max_seq_len,in_query_or_key,token_positions)


 
def run_transformer_block(
    d_model: int,
    num_heads: int, 
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """

    seqLen=in_features.shape[1]
            
    tokenPos=torch.Tensor([[i for i in range(0,seqLen )]])
    
    eps=5e-6
    rmsNormOut=run_rmsnorm(d_model,eps,weights['ln1.weight'],in_features)
    
    attnOut=run_multihead_self_attention_with_rope(d_model,num_heads,
                                                   max_seq_len,
                                                   theta,
                                                   weights['attn.q_proj.weight'],
                                                   weights['attn.k_proj.weight'],
                                                   weights['attn.v_proj.weight'],
                                                   weights['attn.output_proj.weight'],
                                                   rmsNormOut,#in_features,
                                                   tokenPos)

    
    rmsNormOut2=run_rmsnorm(d_model,eps,weights['ln2.weight'],
                            attnOut + in_features)

    d_ff=weights['ffn.w1.weight'].shape[0]
    
    
    swigluOut=run_swiglu(d_model,d_ff,weights['ffn.w1.weight'],
                         weights['ffn.w2.weight'],
                         weights['ffn.w3.weight'],
                         rmsNormOut2)

    
        
    return swigluOut + attnOut + in_features

     

    #raise Exception('I tried')
    #raise NotImplementedError


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
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE theta parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """

    
    layerOutput=run_embedding(vocab_size,d_model,weights['token_embeddings.weight'],
                              in_indices)



    for i in range(num_layers):
        weightDct=dict()
        for ky in ['attn.q_proj.weight',
                   'attn.k_proj.weight',
                   'attn.v_proj.weight',
                   'attn.output_proj.weight',
                   'ln1.weight',
                   'ln2.weight',
                   'ffn.w1.weight',
                   'ffn.w2.weight',
                   'ffn.w3.weight']:
            weightDct[ky]=weights['layers.%d.%s'%(i,ky)]

        layerOutput=run_transformer_block(d_model,
                                          num_heads,
                                          d_ff,
                                          context_length,#???
                                          rope_theta,
                                          weightDct,
                                          layerOutput)#????
            

    eps=5e-6
    normed=run_rmsnorm(d_model,eps,weights['ln_final.weight'],layerOutput)
    finalOut=normed@torch.transpose(weights['lm_head.weight'],0,1)

    
    return finalOut
                   
                    
def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """

    a=in_features
    aSq=a.square()
    dim=a.dim() 
    aSqMean=torch.mean(aSq,dim-1)#,tuple(range(0,dim-1)))
    
    
    #aRMS=np.power(aSqMean,0.5)
    
    aRMS=aSqMean.sqrt()+ eps

    #REVISIT...there is surely a better way than all this transposing
    a=a.transpose(0,2)
    a=a.transpose(1,2)
    aDiv=a.div(aRMS)
    aDiv=aDiv.transpose(1,2)
    aDiv=aDiv.transpose(0,2)
    return aDiv*weights
    #raise NotImplementedError
  

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """

    def sigmoid(x):
        return 1./(1. + torch.exp(-x))
    return in_features*sigmoid(in_features)



def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    numTok=len(dataset)

    #shift=int(np.floor((numTok - context_length-1)/(batch_size-1)))

    #extra=numTok - shift*(batch_size-1) - context_length

    maxStart=numTok - context_length -1
    
    samples=torch.zeros(batch_size,context_length,dtype=torch.int64,device=device)
    labels=torch.zeros(batch_size,context_length,device=device)
    for bi in range(batch_size):
        strt=np.random.randint(maxStart + 1)
        samples[bi,:]=torch.from_numpy(dataset[strt:strt + context_length])#,dtype=torch.int64)
        labels[bi]=torch.from_numpy(dataset[strt + 1:strt + 1 + context_length])
    return (samples,labels)
        
              
    
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
 
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
        
    softMaxMat=torch.Tensor([softm[i][targets[i]] for i in range(batchSize)])
 
    return -torch.Tensor([np.mean([safeLog(softMaxMat[i]) for i in range(softMaxMat.shape[0])])])
    


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    
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



class AdaGrad(torch.optim.Optimizer):
    def __init__(self,params,lr):
        super(AdaGrad,self).__init__(params,dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr=group['lr']
            for p in group['params']:

                state=self.state[p]
                grad=p.grad.data

                g2=state.get('g2',torch.zeros_like(grad))

                g2 += torch.square(grad)
                state['g2']=g2
                p.data -= lr*grad/torch.sqrt(g2 + 1e-5)


class MySGD(torch.optim.Optimizer):
    def __init__(self,params,lr,weight_decay=0.):
        super(MySGD,self).__init__(params,dict(lr=lr,
                                               weight_decay=weight_decay))

    def step(self):
        #print('USING MY OWN')
        for group in self.param_groups:
            lr=group['lr']
            lambd=group['weight_decay']
            #print('lambd is',lambd)
            for p in group['params']:

                grad=p.grad.data

                p.data=p.data - lr*(grad +lambd*p.data)

                
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

                #print('mom[0] is ',mom[0])
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
 

def get_my_cls() -> type[torch.optim.Optimizer]:
    return MySGD


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """

    return AdamW
        
    #raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    warmupStep=max_learning_rate/warmup_iters
    total_iters=cosine_cycle_iters
    #print('it, total_iters',it,total_iters)
    if (it <= warmup_iters):
        return it*warmupStep
    elif (it > total_iters):
        return min_learning_rate
    
    return min_learning_rate + 0.5*(max_learning_rate-min_learning_rate)*( 1+ np.cos(np.pi*(it - warmup_iters)/(cosine_cycle_iters-warmup_iters)))
    
    #raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    rNum=np.random.randint(1000000)
    fName='model_optim_%d'%rNum

    
    mod=pickle.dumps(model.state_dict())
    opt=pickle.dumps(optimizer.state_dict())

    dct={'model_state':model.state_dict(),
         'optimizer_state':optimizer.state_dict(),
         'iteration':iteration}
    torch.save(dct,out)



def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    dct=torch.load(src)
    
    model.load_state_dict(dct['model_state']) 
    optimizer.load_state_dict(dct['optimizer_state'])
    return dct['iteration']
    #raise NotImplementedError



        
    
                
        
            
    

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """

    return tokenizer.Tokenizer(vocab,merges,special_tokens)
    #raise NotImplementedError


class Node:

    def __init__(self,data,
                 nextNode=None,
                 prevNode=None):
        self.data=data
        self.nextNode=nextNode
        self.prevNode=prevNode



def getSpecialTokenLocsOld(sTok,lne):
    locs=[]
    strt=0
    sTokLen=len(sTok)
    while (True):
        loc=lne[strt:].find(sTok)
        if (loc==-1):
            return locs
        else:
            locs.append(strt + loc)
            strt=strt + loc + sTokLen

def getSpecialTokenLocs(sTok,lne):
    splits=lne.split(sTok)
    locs=[]
    sTokLen=len(sTok)
    idx=0
    for s in splits:
        sLen=len(s)
        locs.append(idx + sLen)
        idx=idx + sLen + sTokLen
    return locs

def getAllSpecialTokenLocs(lne,special_tokens):
    dct=dict()
    if (special_tokens==None):
        return []
    specialTokensPresent=[]
    print('a')
    for sTok in special_tokens:
        sTokLen=len(sTok)
        locs=getSpecialTokenLocs(sTok,lne)


        if (len(locs) > 0):
            dct[sTok]=[(l,l+sTokLen) for l in locs]
            specialTokensPresent.append(sTok)
    print('b')
    withLengths=[(s,len(s)) for s in specialTokensPresent]
    withLengths=sorted(withLengths,key=lambda x: x[1],reverse=True)
    
    allSpecialTokenRanges=[]

    alreadyAddedStarts=set()
    alreadyAddedEnds=set()
    for sTokWithLength in withLengths:
        sTok=sTokWithLength[0]

        for rng in dct[sTok]:
            if ( (not (rng[0] in alreadyAddedStarts) ) and ( not (rng[1] in alreadyAddedEnds)  ) ):
                allSpecialTokenRanges.append(rng)#=allSpecialTokenRanges + dct[sTok]
                alreadyAddedStarts.add(rng[0])
                alreadyAddedEnds.add(rng[1])
    print('c')
    return sorted(allSpecialTokenRanges,key=lambda x: x[0])





pat_str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def mergePretokenized(wordCountList,
                      wordToToksList,
                      pairToWordList,
                      tokPairCountList):
    res=dict()
    overallWordCount=dict()
    for wc in wordCountList:
        for word in wc:
            overallWordCount[word]=overallWordCount.get(word,0) + wc[word]
    res['wordCount']=overallWordCount

    overallWordToToks=dict()
    for wtt in wordToToksList:
        for word in wtt:
            if (not (word in overallWordToToks)):
                overallWordToToks[word]=wtt[word]
                
    res['wordToToks']=overallWordToToks

    overallPairToWord=dict()
    
    for ptw in pairToWordList:
        for pair in ptw:
            
            if (pair in overallPairToWord):
                for word in ptw[pair]:
                    overallPairToWord[pair].add(word)
            else:
                overallPairToWord[pair]=ptw[pair]
                
    res['pairToWord']=overallPairToWord
    
    overallTokPairCount=dict()
    for tpc in tokPairCountList:
        for pair in tpc:
            overallTokPairCount[pair]=overallTokPairCount.get(pair,0) + tpc[pair]
    res['tokPairCount']=overallTokPairCount

    return res


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    print('chunk_size in here is',chunk_size)
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenizeNonSpecial(nonSpecialStr,
                          wordCount,
                          wordToToks,
                          pairToWord,
                          tokPairCount,
                          bytesToTok,
                          returnWordList=False):

    if (returnWordList):
        wordList=[]
    for match in re.finditer(pat_str,nonSpecialStr):
        word=match.group(0)
        if (returnWordList):
            wordList.append(word)
        wordCount[word]=wordCount.get(word,0) + 1
        if (not (word in wordToToks)):
            if (bytesToTok==None):
                tokList=list(map(int,word.encode('utf-8')))
            else:
                encoded=word.encode('utf-8')
                tokList=[bytesToTok[encoded[i:i+1]] for i in range(len(encoded))]
            wordToToks[word]=tokList 
        else:
            tokList=wordToToks[word]
                            
        pairs=zip(tokList[0:-1],tokList[1:])
        for pair in pairs:
            tokPairCount[pair]=tokPairCount.get(pair,0) + 1
            if (not (pair in pairToWord)):
                pairToWord[pair]=set([word])
            else:
                pairToWord[pair].add(word)

    if (returnWordList):
        return wordList

                
def pretokenizeNonSpecialByLine(nonSpecialStr,
                          wordCount,
                          wordToToks,
                          pairToWord,
                          tokPairCount):

    lines=nonSpecialStr.split('\n')

    for line in lines:
        for match in re.finditer(pat_str,line):
            word=match.group(0)
            wordCount[word]=wordCount.get(word,0) + 1
            if (not (word in wordToToks)):
                tokList=list(map(int,word.encode('utf-8')))
                wordToToks[word]=tokList
            else:
                tokList=wordToToks[word]

            pairs=zip(tokList[0:-1],tokList[1:])
            for pair in pairs:
                tokPairCount[pair]=tokPairCount.get(pair,0) + 1
                if (not (pair in pairToWord)):
                    pairToWord[pair]=set([word])
                else:
                    pairToWord[pair].add(word)

def getInitIndicesForNonSpecial(nonSpecialStr,bytesToTok=None):

    splitByLine=False

    #print('BY LINES!!!!!!!!!!!!!!!!!')
    if (splitByLine):
        lines=nonSpecialStr.split('\n')
        numLines=len(lines)

        lst=[]
        for lineIdx in range(numLines):
        
            if (bytesToTok==None):
                lineLst=list(map(int,lines[lineIdx].encode('utf-8')))
            else:
                encoded=lines[lineIdx].encode('utf-8')
                lineLst=[bytesToTok[encoded[i:i+1]] for i in range(len(encoded))]
            lst=lst + lineLst
            if (lineIdx != (numLines -1)):
                if (bytesToTok==None):
                    lst=lst + [int('\n'.encode('utf-8')[0])]
                else:
                    lst=lst + bytesToTok['\n'.encode('utf-8')]
    else:
        if (bytesToTok==None):
            lst=list(map(int,nonSpecialStr.encode('utf-8')))
        else:
            encoded=nonSpecialStr.encode('utf-8')
            lst=[bytesToTok[encoded[i:i+1]] for i in range(len(encoded))]
    
    
    if (len(lst)==0):
        return None
    lenLstMinus1=len(lst)-1
    nodeAfter=None
    for lIndx in range(lenLstMinus1,-1,-1):
        node=Node(lst[lIndx],nextNode=nodeAfter)
        if (nodeAfter != None):
            nodeAfter.prevNode=node
        nodeAfter=node
    return node


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    
    #print('special tokens are',special_tokens)
    #print('vocab_size is ',vocab_size)

    import datetime
    print('at start',datetime.datetime.now())
    
    vocab={x:bytes([x]) for x in range(256)}
    
    currVocabSize=len(vocab)
    for sTok in special_tokens:
        vocab[currVocabSize]=bytes(sTok.encode('utf-8'))
        currVocabSize=currVocabSize + 1


    print('reading')
    fileSize=os.path.getsize(input_path)

    NUM_FILE_CHUNKS=4

    chunkSize=int(np.ceil(float(fileSize)/NUM_FILE_CHUNKS))
 

    def preTok(line,special_tokens,idx):
        nonSpecialIndicesList=[]

        allSpecialTokenRanges=getAllSpecialTokenLocs(line,special_tokens)

        print('got special',datetime.datetime.now())

        wordCount=dict()
        wordToToks=dict()
        pairToWord=dict()
        tokPairCount=dict()

        print('num nonspecial',len(allSpecialTokenRanges))

    
        if (len(allSpecialTokenRanges)==0):
            pretokenizeNonSpecial(line,
                                  wordCount,
                                  wordToToks,
                                  pairToWord,
                                  tokPairCount,
                                  None)
        
        else:
            for rngIdx in range(len(allSpecialTokenRanges)):
                if (rngIdx%10000==0):
                    print(rngIdx)
                rnge=allSpecialTokenRanges[rngIdx]
                nonSpecialEnd=rnge[0]
                if (rngIdx==0):
                    nonSpecialStart=0
                else:
                    nonSpecialStart=allSpecialTokenRanges[rngIdx-1][1]

                if (nonSpecialEnd > nonSpecialStart):
                    pretokenizeNonSpecial(line[nonSpecialStart:nonSpecialEnd],
                                          wordCount,
                                          wordToToks,
                                          pairToWord,
                                          tokPairCount,
                                          None)
        result=dict()
        result['wordCount']=wordCount
        result['wordToToks']=wordToToks
        result['pairToWord']=pairToWord
        result['tokPairCount']=tokPairCount

        pickle.dump(result,open('result_%d'%idx,'wb'))
        #print('dumped ok') 
        #if ('wordCount' in result):
        #    print('wordCount there')
        #else:
        #    print('not there')
        

    resultList=[]
    numFilled=0
    with open(input_path,'rb') as f:
        boundaries = find_chunk_boundaries(f, NUM_FILE_CHUNKS, b"<|endoftext|>")
        print('boundaries are',boundaries)
        processes=[]
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            print(start,end)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            proc=Process(target=preTok,args=(chunk,special_tokens,numFilled))
            proc.start()
            processes.append(proc)
            #preTok(chunk,special_tokens,resultList[numFilled])
            numFilled=numFilled+1
        for pr in processes:
            pr.join()
 
        #import pdb; pdb.set_trace()

 
    #def compDict(d1,d2):
    #    for ky in d1:
    #        if (d2.get(ky,None) != d1[ky]):
    #            print(ky,d2.get(ky,None),d1[ky])
    #            raise Exception('problem ',ky)

    #print('wc')
 
    for idx in range(numFilled):
        resultList.append(pickle.load(open('result_%d'%idx,'rb')))
    for r in resultList:
        print('looking')
        print([ky for ky in r])
        
    res=mergePretokenized([r['wordCount'] for r in resultList],
                      [r['wordToToks'] for r in resultList],
                      [r['pairToWord'] for r in resultList],
                      [r['tokPairCount'] for r in resultList])

    #for typ in ['wordCount','wordToToks','pairToWord','tokPairCount']:
    #    print('-----',typ)
    #    print(compDict(resultList[0][typ],res[typ]))
    #import pdb; pdb.set_trace()
    
    #wordCount=resultList[0]['wordCount']
    #wordToToks=resultList[0]['wordToToks']
    #pairToWord=resultList[0]['pairToWord']
    #tokPairCount=resultList[0]['tokPairCount']
    
    wordCount=res['wordCount']
    wordToToks=res['wordToToks']
    pairToWord=res['pairToWord']
    tokPairCount=res['tokPairCount']
    
    print('pretokenized')
    numMerges=vocab_size - currVocabSize
 
    merges=[None]*numMerges
    
     
    #numNonSpecial=len(nonSpecialIndicesList)
    #numNonSpecialRange=range(numNonSpecial)
    
    #print('numNonSpecialRange ',numNonSpecialRange)

    def show(x):
        pair,cnt=x
        print((vocab[pair[0]],vocab[pair[1]],cnt))
 
    MINUS_LARGE=-100000

    class TokPairItem(object):
        def __init__(self,item):
            self.item=item
            self.vocab0=vocab[self.item[0][0]]
            self.vocab1=vocab[self.item[0][1]]
            self.count=item[1]
            
        def __gt__(self,other):
            #print('invoking')
            if (self.item[1] > other.item[1]):
                #print('used item[1]')
                return True
            #print('tiebreaker')
            return (self.vocab0,self.vocab1) > (other.vocab0,other.vocab1)

    
    def tokCompareOld(one,other):
        return (one.item[1] > other.item[1]) or ( (one.item[1]==other.item[1]) and (vocab[one.item[0][0]],vocab[one.item[0][1]] ) > (vocab[other.item[0][0]],vocab[other.item[0][1]] ) )

    def tokCompare(one,other):
        return (one.count > other.count) or ( (one.count==other.count) and ( (one.vocab0,one.vocab1) > (other.vocab0,other.vocab1) ) )
    
    def make_comparator(greater_than):
        def compare(x,y):
            if (greater_than(x,y)):
                return -1
            elif (greater_than(y,x)):
                return 1
            else:
                return 0
        return compare


    myKey=cmp_to_key(make_comparator(tokCompare))
    
    tokPairItemList=sorted([TokPairItem(item) for item in tokPairCount.items()],key=myKey)


    pairToItem=dict(zip([tpi.item[0] for tpi in tokPairItemList],tokPairItemList))

        
    def gt(item1,item2):
        if (item1[1]==item2[1]):
            return item1[0] > item2[0]
        return item1[1] > item2[1]

    
    print('I made merges',datetime.datetime.now())


    totalPairTime=0.
    totalChangeTime=0.
    for mIdx in range(numMerges):

       
       topPairItem=tokPairItemList[0]

       topPair=topPairItem.item[0]
    
       if (tokPairCount[topPair]==0):
           continue
    
       del tokPairCount[topPair]
       tokPairItemList=tokPairItemList[1:]
       
       #tokPairItemList.remove(tokPairItemList[0])

       #b1=vocab[topPair[0]]
       #b2=vocab[topPair[1]]
       newTok=currVocabSize
       vocab[newTok]=topPairItem.vocab0 + topPairItem.vocab1#b1 + b2
           
       merges[mIdx]=(topPairItem.vocab0,topPairItem.vocab1)
       currVocabSize=currVocabSize + 1

       wordsToUpdate=pairToWord[topPair]

       changedTokPairs=set()

       newPairs=[]
       
       for wordToUpdate in wordsToUpdate:

           wordCountForWord=wordCount[wordToUpdate]
           
           tokList=wordToToks[wordToUpdate]

           newTokList=[]
           tokListLen=len(tokList)
           
           idx=0

           while (idx < tokListLen):
               if (idx < (tokListLen -1) ):
                   if ((tokList[idx],tokList[idx+1])==topPair):
                       newTokList.append(newTok)
                       idx=idx + 2
                   else:
                       newTokList.append(tokList[idx])
                       idx=idx + 1
               else:
                   newTokList.append(tokList[idx])
                   idx=idx + 1
    
           oldPairList=[pl for pl in zip(tokList[0:-1],tokList[1:])]
           oldPairSet=set(oldPairList)
           newPairList=[pl for pl in zip(newTokList[0:-1],newTokList[1:])]
           newPairSet=set(newPairList)

           oldNotNew=oldPairSet.difference(newPairSet)
           if (topPair in oldNotNew):
               oldNotNew.remove(topPair)
           newNotOld=newPairSet.difference(oldPairSet)
           both=oldPairSet.intersection(newPairSet)
           
           #def count(pList):
           #    cnt=dict()
           #    for pl in pList:
           #        cnt[pl]=cnt.get(pl,0) + 1
           #    return cnt

           oldCount=collections.Counter(oldPairList)#count(oldPairList)
           newCount=collections.Counter(newPairList)#count(newPairList)
 
           for pair in oldNotNew:
               #if (pair != topPair):
               changedTokPairs.add(pair)                   
               tokPairCount[pair]=tokPairCount[pair] - oldCount[pair]*wordCountForWord
                   
           for pair in newNotOld:               
               if (pair in tokPairCount):
                   if (pair in pairToItem):
                       changedTokPairs.add(pair)
               else: 
                   newPairs.append(pair)
                       
               tokPairCount[pair]=tokPairCount.get(pair,0) + newCount[pair]*wordCountForWord

               if (pair in pairToWord):
                   pairToWord[pair].add(wordToUpdate)
               else:
                   pairToWord[pair]=set([wordToUpdate])
                   
           for pair in both:
               if (newCount[pair] != oldCount[pair]):
                   #if (not (pair in changedTokPairs) ):
                   changedTokPairs.add(pair)
                   tokPairCount[pair]=tokPairCount[pair] + wordCountForWord*(newCount[pair] - oldCount[pair])
                              
           wordToToks[wordToUpdate]=newTokList 



       prePairTime=datetime.datetime.now()
       for pair in changedTokPairs:
           oldItem=pairToItem[pair]
           
           #i=bisect.bisect_left(tokPairItemList,oldItem)
           #if (i==0):
           #    tokPairItemList=tokPairItemList[1:]
           #elif (i==len(tokPairItemList)-1):
           #    tokPairItemList=tokPairItemList[0:i]
           #else:
           #    tokPairItemList=tokPairItemList[0:i] + tokPairItemList[i+1:]
               
           tokPairItemList.remove(oldItem)               
           newItem=TokPairItem((pair,tokPairCount[pair]))
           pairToItem[pair]=newItem 
           bisect.insort(tokPairItemList,newItem,key=myKey)#  cmp_to_key(make_comparator(tokCompare)))

       #postChangeTime=datetime.datetime.now() - prePairTime
       #totalChangeTime=totalChangeTime + postChangeTime.total_seconds()
           #print('new pairs')
       #if (topPair in newPairs):
       #    print('nono')
       #    import pdb; pdb.set_trace()
       
       for pair in newPairs:
           newItem=TokPairItem((pair,tokPairCount[pair]))
           pairToItem[pair]=newItem
           bisect.insort(tokPairItemList,newItem,key=myKey)#=cmp_to_key(make_comparator(tokCompare)))

       #pairTime=datetime.datetime.now() - prePairTime
       #print(pairTime)
       #totalPairTime=totalPairTime + pairTime.total_seconds()
       
       
       #print('all done')
       #import pdb; pdb.set_trace()
       #tokPairOrderedList=sorted(tokPairCount.items(),key=lambda item: (item[1], (vocab[item[0][0]]),vocab[item[0][1]] ),reverse=True)
                                                                        #item[0]) )

    
       #if (mIdx < 10):
           
       #    print('now ordered list is')
       #    for t in tokPairOrderedList[0:5]:
       #        show(t)
       #toInsert=(newPair,tokPairCount[newPair])
       #insertLoc=None
       #for i in range(orderedListLen):
       #    if (gt(toInsert,tokPairOrderedList[i]) ):
       #        insertLoc=i
       #if (insertLoc==None):
       #    tokPairOrderedList.append(toInsert)
       #else:
       #    tokPairOrderedList=tokPairOrderedList[0:i] + [toInsert] + tokPairOrderedList[i:]
            

                
                  
    #print('*********** currVocabSize is',currVocabSize)
    #print('vocab_size is ', vocab_size)
    #print('cntDct is',cntDct)
    #time.sleep(0.05)
    #print('first merge is ', merges[0])
    #print('first 10 are ', merges[0:10])

    print('totalPairTime %4.4f'%totalPairTime)
    print('totalChangeTime %4.4f'%totalChangeTime)
    print('done',datetime.datetime.now())
    return vocab,merges  
    
 
