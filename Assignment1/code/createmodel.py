import random
from providedcode import dataset
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor
from transition import Transition
from providedcode.dependencygraph import DependencyGraph

if __name__ == '__main__':
    sweedTrainData = dataset.get_swedish_train_corpus().parsed_sents()
    random.seed(1234)
    sweedTrainSubdata = random.sample(sweedTrainData, 200)

    sweedTestData = dataset.get_swedish_test_corpus().parsed_sents()
    
    englishTrainData = dataset.get_english_train_corpus().parsed_sents()
    random.seed(1234)
    englishTrainSubdata = random.sample(englishTrainData, 200)

    danishTrainData = dataset.get_danish_test_corpus().parsed_sents()
    random.seed(1234)
    danishTrainSubdata = random.sample(danishTrainData, 200)

    
    try:
        tp = TransitionParser(Transition, FeatureExtractor)

        #print "\n\n"
        #print sweedTrainSubdata

        print "\n\n"
        tp.train(sweedTrainSubdata)
        tp.save('swedish.model')

        tp = TransitionParser(Transition, FeatureExtractor)
        tp.train(englishTrainSubdata)
        tp.save('english.model')

        tp = TransitionParser(Transition, FeatureExtractor)
        tp.train(danishTrainSubdata)
        tp.save('danish.model')

        # parsing arbitrary sentences (english):
        sentence = DependencyGraph.from_sentence('Hi, this is a test')

        tp = TransitionParser.load('english.model')
        parsed = tp.parse([sentence])
        print parsed[0].to_conll(10).encode('utf-8')
    except NotImplementedError:
        print "Stupid Corpus Generator"