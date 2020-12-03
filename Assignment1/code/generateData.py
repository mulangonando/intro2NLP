import random
from providedcode import dataset
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor
from transition import Transition

if __name__ == '__main__':
    data = dataset.get_swedish_train_corpus().parsed_sents()
    random.seed(1234)
    subdata = random.sample(data, 1000)

    try:
        # tp = TransitionParser(Transition, FeatureExtractor)
        # tp.train(subdata)
        # tp.save('swedish.model')

        testdata = dataset.get_swedish_test_corpus().parsed_sents()
        tp = TransitionParser.load('badfeatures.model')

        parsed = tp.parse(testdata)

        with open('test.conll', 'w') as f:
            for p in parsed:
                f.write(p.to_conll(10).encode('utf-8'))
                f.write('\n')

        ev = DependencyEvaluator(testdata, parsed)
        print "UAS: {} \nLAS: {}".format(*ev.eval())

        # parsing arbitrary sentences (english):
        # sentence = DependencyGraph.from_sentence('Hi, this is a test')

        # tp = TransitionParser.load('english.model')
        # parsed = tp.parse([sentence])
        # print parsed[0].to_conll(10).encode('utf-8')
    except NotImplementedError:
        print """