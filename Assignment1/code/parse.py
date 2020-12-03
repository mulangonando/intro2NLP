import random
from providedcode import dataset
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor
from transition import Transition
import sys
from providedcode.dependencygraph import DependencyGraph

if __name__ == '__main__':
    
    try :
        #parsing arbitrary sentences (english):
        in_string=""
        for line in sys.stdin:
             in_string + line+"\n"

        sentence = DependencyGraph.from_sentence(in_string)

        tp = TransitionParser.load('english.model')
        parsed = tp.parse([sentence])
        print parsed[0].to_conll(10).encode('utf-8')

    
    except NotImplementedError:
        print "This did not work"