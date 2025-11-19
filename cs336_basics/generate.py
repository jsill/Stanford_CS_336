import trainTools,tokenizer,llModel
import sys
import numpy as np
import os
import pickle
import torch
import gc
import posixpath

from io import BytesIO

#if __name__=='__main__':


def load(dumpIter):
    #going to guess batch size 1000 and 1000 steps based on context_length 256 and recommended total tokens processed 327 million
    #fName=sys.argv[1]

     
 
    batchSize=500
    contextLength=256
    CHUNK_SIZE=contextLength*batchSize#256000
 

    
    SPECIAL_TOK=b'<|endoftext|>'


    
    vocab,merges=pickle.load(open('/workspace/bpeResults','rb'))

    vocabSize=len(vocab)
    
    bpeTokenizer=tokenizer.Tokenizer(vocab,merges,[str(SPECIAL_TOK)])

    print('loaded tokenizer')
    d_model=512
    num_layers=4
    num_heads=16
    d_ff=1344
    rope_theta=10000
    
    model=llModel.LLModel(vocabSize,
                          contextLength,
                          d_model,
                          num_layers,
                          num_heads,
                          d_ff,
                          rope_theta
                          )

    #model=torch.compile(model)
    
    #import pdb; pdb.set_trace() 
    optimizer=trainTools.AdamW(model.parameters(),
                               #lr=1e-3,
                               lrMax=1e-4,
                               lrMin=1e-4,
                               warmup_iters=1,
                               cosine_cycle_iters=1, 
                               weight_decay=1e-3,
                               betas=(0.9,0.999),
                               #eps=0.5)
                               eps=1e-8)

    if (dumpIter != None):
        inFName='cosine_preEncoded_modelDump_0_%d'%dumpIter
        trainTools.load_checkpoint(inFName,model,optimizer)#,inFName)#BytesIO(open(inFName,'rb').read() ))
        #import pdb; pdb.set_trace()
        print('loaded checkpoint')

    return bpeTokenizer,model

def generateStory(inputTextString,bpeTokenizer,model,temperature,
                  show=None,showLoops=None,verbose=False,seed=1234):
    np.random.seed(seed)
    chooseMax=False
    inputEncoded=np.array(bpeTokenizer.encode(inputTextString))
    #print('encoded')
    numInputTokens=len(inputEncoded)
    inputTensor=torch.from_numpy(inputEncoded)
    inputTensor=inputTensor.unsqueeze(0)
    #import pdb; pdb.set_trace()
    #inputLen=inputTensor.size()[0]

    #padding=torch.tensor([32]*(256-inputLen),dtype=torch.int64)
    #inputTensor=torch.concat([inputTensor,padding])
    #inputTensor=inputTensor.unsqueeze(0)
    numExtra=100

    #totalSize=256 + numExtra
    #import pdb; pdb.set_trace()
    output=torch.zeros((1,256),dtype=torch.int64)
    output[0,0:numInputTokens]=inputTensor

    def sample(probs):
        #vocabLen=len(probs)
        cumul=np.cumsum(probs)
        rNum=np.random.rand()
        #print('sampling')
        #import pdb; pdb.set_trace()
        return np.searchsorted(cumul,rNum)

    endFound=False
    i=0
    totalOutput=[]
    while (not endFound):
        
    #for i in range(numExtra):
       #predictions=model.forward(inputTensor)
       #import pdb; pdb.set_trace()
       if (verbose):
           print('numInputTokens is',numInputTokens)
           print('output is',output[0][0:numInputTokens + i])
       predictions=model.forward(output)
       #import pdb; pdb.set_trace()
       if (numInputTokens + i -1 > 255):
           idx=255
       else:
           idx=numInputTokens + i -1
       probs=trainTools.run_softmax(predictions[0,idx].detach().cpu()/temperature,0)

       if (show != None):
           print('Top Probability Tokens:\n')
           argsort=np.argsort(-probs)
           for j in range(show):
               print(argsort[j].item(),bpeTokenizer.vocab[argsort[j].item()],"%3.3f"%probs[argsort[j]])
           if (showLoops != None):
               if (i > showLoops):
                   endFound=True

           print('\n')
            
       if (chooseMax): 
           pred=np.argmax(predictions[0,idx].detach().cpu())
       else: 
           pred=sample(probs.detach().cpu())#np.argmax(predictions[0,-1,:].detach().cpu())
       if (verbose or show):
           print('chosen token: ',bpeTokenizer.vocab[pred.item()])
           print('\n\n')
       #print(i,pred)
       if (numInputTokens+ i > 255):
           output[0,0:255]=output[0,1:].clone()
           output[0,255]=pred
       else:
           output[0,numInputTokens+i]=pred
       i=i+1
       #totalOutput.append(pred.item())
       try:
           if (bpeTokenizer.vocab[pred.item()].decode('utf-8').find('<') != -1):
               endFound=True
               #print('found the end')
           else:
               totalOutput.append(pred.item())
       except:
           totalOutput.append(pred.item())
        
       if (i > 2000):
           raise Exception('could not finish the sotry')
    #import pdb; pdb.set_trace()
    print(bpeTokenizer.decode(totalOutput))
    #print(bpeTokenizer.decode([o.item() for o in output[0,0:numInputTokens + numExtra]]))
    
          
#if __name__=='__main__':

#    do(sys.argv[1],int(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]))

       

     
    
        
    
