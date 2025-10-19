import sys
import pstats

if __name__=='__main__':
    fName=sys.argv[1]
    nTop=int(sys.argv[2])

    st=pstats.Stats(fName)
    
    st.sort_stats('cumtime').print_stats(nTop)


    
