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
def do(fName,val_fName,dumpIter=None):
    #going to guess batch size 1000 and 1000 steps based on context_length 256 and recommended total tokens processed 327 million
    #fName=sys.argv[1]

    
    fileSize=os.path.getsize(fName)
    
    mmappedFile=np.memmap(fName)


    valFileSize=os.path.getsize(val_fName)

    print('fileSize',fileSize)
    print('valFileSize',valFileSize)
    
    valMmappedFile=np.memmap(val_fName)
    
    #batch size 1000 and context 256
 
    batchSize=500
    contextLength=256
    CHUNK_SIZE=contextLength*batchSize#256000
 
    numChunks=fileSize//CHUNK_SIZE
    numValChunks=valFileSize//CHUNK_SIZE
    
    SPECIAL_TOK=b'<|endoftext|>'

    def getBoundaries(aFileName,nChunks):
        fle=open(aFileName,'rb')
        boundaries=tokenizer.find_chunk_boundaries(fle,nChunks,SPECIAL_TOK)

        fle.close()
        return boundaries
    
    trainBoundaries=getBoundaries(fName,numChunks)
    valBoundaries=getBoundaries(val_fName,numValChunks)

    numValBoundaries=len(valBoundaries)
    print('num val boundaries',numValBoundaries)
    
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
    numSteps=1

    
    def validateChunk(bIdx):
        #import pdb; pdb.set_trace()
        valTextString=valMmappedFile[valBoundaries[bIdx]:valBoundaries[bIdx + 1]].tobytes().decode('utf-8')
        valEncoded=np.array(bpeTokenizer.encode(valTextString))
        valBatchSamples,valBatchLabels=trainTools.get_batch(valEncoded,batchSize,contextLength,trainTools.DEVICE)
        valPredictions=model.forward(valBatchSamples)
        import pdb; pdb.set_trace()
        del valBatchSamples
        del valTextString
        del valEncoded
        ret=trainTools.run_cross_entropy(valPredictions,valBatchLabels)
        del valPredictions
        del valBatchLabels
        #import pdb; pdb.set_trace()
        gc.collect()
        return ret

    valLosses=[]
    for sIdx in range(numSteps):
        for bIdx in [80]:#range(len(trainBoundaries)):
            #textString=mmappedFile[trainBoundaries[bIdx]:trainBoundaries[bIdx + 1]].tobytes().decode('utf-8')

        
            #print('encoding step %d boundary %d of %d'%(sIdx,bIdx,len(trainBoundaries)))
            #encodedText=np.array(bpeTokenizer.encode(textString))
            #print('encoded')
        
            #batchSamples,batchLabels=trainTools.get_batch(encodedText,batchSize,contextLength,trainTools.DEVICE)

            #import pdb; pdb.set_trace()
            #predictions=model.forward(batchSamples)
            #print('computing loss')
            #import pdb; pdb.set_Trace()
            #loss=trainTools.run_cross_entropy(predictions,batchLabels)

            #print('backwarding')
            #loss.backward()
            #print('stepping')
            #optimizer.step()
            #print('step is %d loss is %f'%(sIdx,loss))
            #del textString
            #del encodedText
            #del batchSamples
            #del batchLabels
            #del predictions
            #if (bIdx%100==0):
            #gc.collect()
            #totalIter=(sIdx+1)*bIdx
            #if (totalIter%100==0):
            #    outFName='modelDump_%d'%totalIter
            #    trainTools.save_checkpoint(model,optimizer,totalIter,outFName)
            valLosses.append(validateChunk(bIdx%numValChunks))
            recentValLosses=torch.tensor(valLosses[-numValChunks:],device=trainTools.DEVICE)
            recentValLosses.to(trainTools.DEVICE)
            print('avg val loss %f'%(recentValLosses.mean()))
         
if __name__=='__main__':
    if (len(sys.argv)==3):
        do(sys.argv[1],sys.argv[2])
    else:
        do(sys.argv[1],sys.argv[2],int(sys.argv[3]))
     
    
        
    
