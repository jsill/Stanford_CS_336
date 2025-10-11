import trainingScript,cProfile

if __name__=='__main__':
    cProfile.run('trainingScript.do("TinyStoriesV2-GPT4-train.txt")','output.prof')

    
