import trainTools,tokenizer,llModel
import sys
import numpy as np
import os
import pickle

if __name__=='__main__':

    #going to guess batch size 1000 and 1000 steps based on context_length 256 and recommended total tokens processed 327 million
    fName=sys.argv[1]

    
    fileSize=os.path.getsize(fName)
    
    mmappedFile=np.memmap(fName)

    #batch size 1000 and context 256

    batchSize=10
    contextLength=256
    CHUNK_SIZE=contextLength*batchSize#256000
 
    numChunks=fileSize//CHUNK_SIZE

    SPECIAL_TOK=b'<|endoftext|>'

    fle=open(fName,'rb')
    boundaries=tokenizer.find_chunk_boundaries(fle,numChunks,SPECIAL_TOK)

    fle.close()
    
    vocab,merges=pickle.load(open('../bpeResults','rb'))

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

    for bIdx in range(len(boundaries)):
        textString=mmappedFile[boundaries[bIdx]:boundaries[bIdx + 1]].tobytes().decode('utf-8')

        #import pdb; pdb.set_trace()
        print('encoding')
        encodedText=np.array(bpeTokenizer.encode(textString))
        print('encoded')
        
        batchSamples,batchLabels=trainTools.get_batch(encodedText,batchSize,contextLength,'cpu')

        
        predictions=model.forward(batchSamples)

        #import pdb; pdb.set_trace()
        loss=trainTools.run_cross_entropy(predictions,batchLabels)

        import pdb; pdb.set_trace()
        loss.backward()
        
        #print('i did backward')
        import pdb; pdb.set_trace()

        
    
