from __future__ import annotations

import os
from typing import IO, Any, BinaryIO

import regex as re

import numpy as np

import collections
from collections import Counter

from multiprocessing import Process

from functools import cmp_to_key

import bisect

import pickle

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
    #print('file_size in here is',file_size)
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    #print('chunk_size in here is',chunk_size)
    # Initial guesses for chunk boundary locations, uniformly spaced                                                                               
    # Chunks start on previous index, don't include last index                                                                                     
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    #print('init boundaries',chunk_boundaries)
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time                                                                                     

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        #print('initial position',initial_position)
        #print('cb now',chunk_boundaries)
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


class Tokenizer:

    def __init__(self,vocab,merges,special_tokens):

        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens

        self.maxByteLen=max([len(val) for val in self.vocab.values()])


        toks=[ky for ky in self.vocab]
        self.bytesToTok=dict(zip([self.vocab[t] for t in toks],toks))

    def encode_iterable(self,aFile):
        CHUNK_SIZE=10000
        chunk=aFile.read(CHUNK_SIZE)
        encoding=[]
        i=0
        while (chunk != ''):
            #print(chunk[0:100])                                                                                                                   
            print('------ %d -----'%i)
            encoding=encoding + self.encode(chunk)
            chunk=aFile.read(CHUNK_SIZE)
            #gc.collect()                                                                                                                          
            i=i+1
        return encoding

    def encode(self,textString):

        #if (len(textString) < 500):                                                                                                               
        #    print('textString is::::\n',textString)                                                                                               

        specialTokenRanges=getAllSpecialTokenLocs(textString,self.special_tokens)

        def encodeNonSpecial(nonSpecialTextString):
            #print('nonSpecialTextString::::',nonSpecialTextString,len(nonSpecialTextString))                                                      
            wordCount=dict()
            wordToToks=dict()
            pairToWord=dict()
            tokPairCount=dict()


            wordList=pretokenizeNonSpecial(nonSpecialTextString,
                                           wordCount,
                                           wordToToks,
                                           pairToWord,
                                           tokPairCount,
                                           self.bytesToTok,
                                           returnWordList=True)
            numWords=len(wordList)

            for m in self.merges:
                t1,t2=m
                tok1=self.bytesToTok[t1]
                tok2=self.bytesToTok[t2]
                mergedTok=self.bytesToTok[t1 + t2]

                pair=(tok1,tok2)
                for word in pairToWord.get(pair,set()):
                    oldToks=wordToToks[word]
                    numOldToks=len(oldToks)
                    newToks=[]
                    idx=0
                    while (idx < numOldToks):
                        if ( (idx < (numOldToks-1) ) and
                              (oldToks[idx]==tok1) and
                              (oldToks[idx+1]==tok2) ):
                            newToks.append(mergedTok)
                            idx=idx + 2
                        else:
                            newToks.append(oldToks[idx])
                            idx=idx + 1
                    wordToToks[word]=newToks
                    newPairs=zip(newToks[0:-1],
                                 newToks[1:])
                    for pair in newPairs:
                        if (pair in pairToWord):
                             pairToWord[pair].add(word)
                        else:
                             pairToWord[pair]=set([word])

            ret=[]
            for w in wordList:
                ret=ret + wordToToks[w]
            #print('ret is',ret)                                                                                                                   
            return ret

        encoding=[]

        def findWhichSpecialToken(txtString):
            txtBytes=txtString.encode('utf-8')
            #try:                                                                                                                                  
            return self.bytesToTok[txtBytes]
            #except:                                                                                                                               
            #    print('prob')                                                                                                                     
            #    import pdb; pdb.set_trace()                                                                                                       
        if (specialTokenRanges==[]):
            return encodeNonSpecial(textString)

        def doByLine(allText):
            encodingLst=[]
            lineSplits=allText.split('\n')

            for lsIdx in range(len(lineSplits)):

                encodingLst=encodingLst + encodeNonSpecial(lineSplits[lsIdx])
                if (lsIdx < len(lineSplits) -1):
                    encodingLst=encodingLst + [self.bytesToTok[b'\n']]
            return encodingLst

        for rngIdx in range(len(specialTokenRanges)):
            rnge=specialTokenRanges[rngIdx]

            nonSpecialEnd=rnge[0]
            if (rngIdx==0):
                nonSpecialStart=0
            else:
                nonSpecialStart=specialTokenRanges[rngIdx-1][1]

            if (nonSpecialEnd > nonSpecialStart):
                chunkText=textString[nonSpecialStart:nonSpecialEnd]

                encoding=encoding + encodeNonSpecial(chunkText)#doByLine(chunkText)

            if (rnge[0] < len(textString)):
                encoding.append(findWhichSpecialToken(textString[rnge[0]:rnge[1]]))
                #prevSpecialTokenEnd=rnge[1]                                          

        if (specialTokenRanges[-1][1] < len(textString) ):
            #print('here B')                                                                                                                       
            encoding=encoding + encodeNonSpecial(textString[specialTokenRanges[-1][1]:])#encodeNonSpecial(textString[specialTokenRanges[-1][1]:])  

        return encoding


    def decode(self,tokenList):
        numTok=len(tokenList)

        tokIdx=0
        ret=''

        while (tokIdx < numTok):
            tok=tokenList[tokIdx]
            theBytes=self.vocab[tok]
            #lastByte=theBytes[-1]                                                                                                                 

            bytesLen=len(theBytes)

            numBytesToDecode=bytesLen
            for bIdx in range(len(theBytes)):
                theByte=theBytes[bIdx]
                def numToDecode(aByte):
                    numBytesToDecode=1
                    if ( (theByte >= 192) and (theByte <= 223) ):
                        numBytesToDecode=2
                    elif ( ( theByte >=224) and (theByte <=239) ):
                        numBytesToDecode=3
                    elif ( (theByte >= 240) and (theByte <= 247) ):
                        numBytesToDecode=4
                    return numBytesToDecode
                numBytesToDecodeFromHere=numToDecode(theByte)
                totalNumBytesToDecode=numBytesToDecodeFromHere + bIdx
                if (totalNumBytesToDecode > numBytesToDecode):
                    numBytesToDecode=totalNumBytesToDecode

            toDecode=self.vocab[tokenList[tokIdx]]
            numBytesWeHave=len(toDecode)

            tokIdx=tokIdx + 1
            while ( (tokIdx < len(tokenList)) and (numBytesWeHave < numBytesToDecode) ):
                #print('decoding ',tokenList[tokIdx])                                                                                              
                toDecode=toDecode + self.vocab[tokenList[tokIdx]]
                numBytesWeHave=len(toDecode)
                if ( (self.special_tokens != None) and (tokenList[tokIdx] in self.special_tokens) ):
                    numBytesWeHave=numBytesToDecode
                tokIdx=tokIdx + 1

            try:
                #print('added  ', toDecode.decode('utf-8'))                                                                                        
                ret=ret + toDecode.decode('utf-8')
            except UnicodeDecodeError:
                if (len(tokenList)==1):
                    pass

        return ret

    
def train_bpe(input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

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

            
    for idx in range(numFilled):
        resultList.append(pickle.load(open('result_%d'%idx,'rb')))
    for r in resultList:
        print('looking')
        print([ky for ky in r])

    res=mergePretokenized([r['wordCount'] for r in resultList],
                      [r['wordToToks'] for r in resultList],
                      [r['pairToWord'] for r in resultList],
                      [r['tokPairCount'] for r in resultList])

    wordCount=res['wordCount']
    wordToToks=res['wordToToks']
    pairToWord=res['pairToWord']
    tokPairCount=res['tokPairCount']

    print('pretokenized')
    numMerges=vocab_size - currVocabSize

    merges=[None]*numMerges

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

           tokPairItemList.remove(oldItem)
           newItem=TokPairItem((pair,tokPairCount[pair]))
           pairToItem[pair]=newItem
           bisect.insort(tokPairItemList,newItem,key=myKey)#  cmp_to_key(make_comparator(tokCompare)))                                           

       for pair in newPairs:
           newItem=TokPairItem((pair,tokPairCount[pair]))
           pairToItem[pair]=newItem
           bisect.insort(tokPairItemList,newItem,key=myKey)

    print('totalPairTime %4.4f'%totalPairTime)
    print('totalChangeTime %4.4f'%totalChangeTime)
    print('done',datetime.datetime.now())
    return vocab,merges
