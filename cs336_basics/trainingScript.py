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
def do(encodedDirName,dumpIter=None):
    #going to guess batch size 1000 and 1000 steps based on context_length 256 and recommended total tokens processed 327 million
    #fName=sys.argv[1]

    
    #batchSize=500
    batchSize=64
    contextLength=256
    CHUNK_SIZE=contextLength*batchSize#256000
 
    vocab,merges=pickle.load(open('/workspace/bpeResults','rb'))

    vocabSize=len(vocab)


    SPECIAL_TOK='<|endoftext|>'
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
                               lrMax=1e-4,
                               lrMin=1e-5,
                               warmup_iters=1000,
                               cosine_cycle_iters=1000,
                               weight_decay=1e-3,
                               #weight_decay=1e-1,
                               betas=(0.9,0.999),
                               #eps=0.5)
                               eps=1e-9)

    #sIdx=0
    #bIdx=15000
    if (dumpIter != None):
        inFName='cosine_preEncoded_modelDump_1_16596'##%d'%dumpIter
        iterNum=trainTools.load_checkpoint(inFName,model,optimizer)#,inFName)#BytesIO(open(inFName,'rb').read() ))
        #import pdb; pdb.set_trace()
    numSteps=2
  
    encodedFileNames=os.listdir(encodedDirName)
    
    totalIter=0
    for sIdx in range(numSteps):
        bIdx=0
        for encodedFileName in encodedFileNames:

            encodedText=pickle.load(open(os.path.join(encodedDirName,encodedFileName),'rb' ))#np.array(bpeTokenizer.encode(textString))
            #import pdb; pdb.set_trace()
            
        
            batchSamples,batchLabels=trainTools.get_batch(encodedText,batchSize,contextLength,trainTools.DEVICE)

            #import pdb; pdb.set_trace()
            predictions=model.forward(batchSamples)
            
            #print('computing loss')
            #import pdb; pdb.set_Trace()
            loss=trainTools.run_cross_entropy(predictions,batchLabels)

            #print('backwarding')
            loss.backward()
            #print('stepping')
            optimizer.step()
            if (totalIter%10==0):
                print('step is %d bIdx is %d loss is %f'%(sIdx,bIdx,loss))
            
            #del encodedText
            #del batchSamples
            #del batchLabels
            #del predictions
            #if (bIdx%100==0):
            #gc.collect()
            #totalIter=totalIter + 1
             
            if (totalIter%1000==0):
                outFName='cosine_preEncoded_modelDump_%d_%d'%(sIdx,bIdx)
                trainTools.save_checkpoint(model,optimizer,totalIter,outFName)
            totalIter=totalIter + 1
            bIdx=bIdx + 1 
            #valLosses.append(validateChunk(bIdx%numValChunks))
            #recentValLosses=torch.tensor(valLosses[-numValChunks:],device=trainTools.DEVICE)
            #recentValLosses.to(trainTools.DEVICE)
            #print('avg val loss %f'%(recentValLosses.mean()))
          
if __name__=='__main__':
    if (len(sys.argv)==2):
        do(sys.argv[1])#,sys.argv[2])
    else:
        do(sys.argv[1],int(sys.argv[2]))
     
     
        
     
