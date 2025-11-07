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
def do(inputTextString,dumpIter,temperature,chooseMax):
    #going to guess batch size 1000 and 1000 steps based on context_length 256 and recommended total tokens processed 327 million
    #fName=sys.argv[1]

    
 
    batchSize=500
    contextLength=256
    CHUNK_SIZE=contextLength*batchSize#256000
 

    
    SPECIAL_TOK=b'<|endoftext|>'


    
    vocab,merges=pickle.load(open('/workspace/bpeResults','rb'))

    vocabSize=len(vocab)
    
    bpeTokenizer=tokenizer.Tokenizer(vocab,merges,[str(SPECIAL_TOK)])

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
                               lr=1e-4,
                               weight_decay=1e-3,
                               betas=(0.9,0.999),
                               #eps=0.5)
                               eps=1e-8)

    if (dumpIter != None):
        inFName='modelDump_%d'%dumpIter
        trainTools.load_checkpoint(inFName,model,optimizer)#,inFName)#BytesIO(open(inFName,'rb').read() ))
        #import pdb; pdb.set_trace()


    #inputTextString="There was once a boy who had a cute dog. The boy enjoyed taking the dog for walks. "

    #inputTextString="Sally loved to play catch with her father. One day, Sally and her father were throwing a ball back and forth in the backyard when a neighbor's dog came into the yard. the dog barked several times as Sally played catch with her father. Finally, the dog leapt and grabbed the ball out of the air and then landed and ran out of the yard. Sally really wanted the ball back. She decided to chase after the dog. The dog ran down the street and then leapt over a fence into a neighbor's yard. Sally decided to climb over the fence into the neighbor's yard. Just as she landed in the yard, she saw the dog go into the house. Against her better judgment, she decided to follow the dog into the house. As she entered the basement family room of the house, an angry man came down the stairs and confronted her. The man wanted to know what she was doing in the house. She told him that his dog had stolen her ball. The man told her that she should have knocked on his front door instead of trespassing and barging into his house. She thought about it for a second and then she apologized. However, she still wanted her ball back. She asked the man to take "#dad in their backyard."
    #import pdb; pdb.set_trace()

    #inputTextString="Sally loved to play marbles with her brother. Every afternoon, they played marbles. Sometimes, Sally won. Sometimes, her brother won."

    #inputTextString="George was an 8 year old boy who loved his dog Spot. One day, Spot ran away. George searched the whole neighborhood for Spot but could not find him. "
    
    inputEncoded=np.array(bpeTokenizer.encode(inputTextString))
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
       predictions=model.forward(output)
       #import pdb; pdb.set_trace()
       if (numInputTokens + i -1 > 255):
           idx=255
       else:
           idx=numInputTokens + i -1
       probs=trainTools.run_softmax(predictions[0,idx].detach().cpu()/temperature,0)
       
       #import pdb; pdb.set_trace()
       if (chooseMax):
           pred=np.argmax(predictions[0,idx].detach().cpu())
       else:
           pred=sample(probs.detach().cpu())#np.argmax(predictions[0,-1,:].detach().cpu())
       #print(i,pred)
       if (numInputTokens+ i > 255):
           output[0,0:255]=output[0,1:].clone()
           output[0,255]=pred
       else:
           output[0,numInputTokens+i]=pred
       i=i+1
       #totalOutput.append(pred.item())
       if (bpeTokenizer.vocab[pred.item()].decode('utf-8').find('<') != -1):
           endFound=True
           print('found the end')
       else:
           totalOutput.append(pred.item())
       if (i > 2000):
           raise Exception('no end token')
    #import pdb; pdb.set_trace()
    print(bpeTokenizer.decode(totalOutput))
    #print(bpeTokenizer.decode([o.item() for o in output[0,0:numInputTokens + numExtra]]))
    
          
if __name__=='__main__':

    do(sys.argv[1],int(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]))

       

     
    
        
    
