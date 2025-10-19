import trainTools,tokenizer,llModel
import sys
import numpy as np
import os
import pickle
import torch
  
#if __name__=='__main__':
def do(fName):
    #going to guess batch size 1000 and 1000 steps based on context_length 256 and recommended total tokens processed 327 million
    #fName=sys.argv[1]

    
    fileSize=os.path.getsize(fName)
    
    mmappedFile=np.memmap(fName)

    #batch size 1000 and context 256
 
    batchSize=500
    contextLength=256
    CHUNK_SIZE=contextLength*batchSize#256000
 
    numChunks=fileSize//CHUNK_SIZE

    SPECIAL_TOK=b'<|endoftext|>'

    fle=open(fName,'rb')
    boundaries=tokenizer.find_chunk_boundaries(fle,numChunks,SPECIAL_TOK)

    fle.close()
    
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

    model=torch.compile(model)
    
    #import pdb; pdb.set_trace() 
    optimizer=trainTools.AdamW(model.parameters(),
                               #lr=1e-3,
                               lr=1e-4,
                               weight_decay=1e-3,
                               betas=(0.9,0.999),
                               #eps=0.5)
                               eps=1e-8)
 
    numSteps=200  
    for sIdx in range(numSteps):
        for bIdx in [0]:#range(len(boundaries)):
            textString=mmappedFile[boundaries[bIdx]:boundaries[bIdx + 1]].tobytes().decode('utf-8')

        
            print('encoding step %d boundary %d of %d'%(sIdx,bIdx,len(boundaries)))
            encodedText=np.array(bpeTokenizer.encode(textString))
            print('encoded')
        
            batchSamples,batchLabels=trainTools.get_batch(encodedText,batchSize,contextLength,trainTools.DEVICE)

            predictions=model.forward(batchSamples)
            print('computing loss')
            #import pdb; pdb.set_Trace()
            loss=trainTools.run_cross_entropy(predictions,batchLabels)

            print('backwarding')
            loss.backward()
            print('stepping')
            optimizer.step()
            print('step is %d loss is %f'%(sIdx,loss))
            
        
if __name__=='__main__':
    do(sys.argv[1])

    
    
        
    
