import math
import nltk
import time
from collections import defaultdict

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    #the_sentences = nltk.tokenize.sent_tokenize(training_corpus)
    #sents_for_unigrams=str([sent+' '+STOP_SYMBOL for sent in the_sentences])
    #sents_for_bigrams=[START_SYMBOL+' '+sent+' '+STOP_SYMBOL for sent in the_sentences]
    #sents_for_trigrams=[START_SYMBOL+' '+START_SYMBOL+' '+sent+' '+STOP_SYMBOL for sent in the_sentences]
    
    #unigram_tokens = nltk.word_tokenize(sents_for_unigrams)
    #bigram_tokens = nltk.word_tokenize(sents_for_bigrams)
    #trigram_tokens = nltk.word_tokenize(sents_for_trigrams)
    
    #unigram_tuples = list(nltk.unigrams(unigram_tokens))
    #bigram_tuples = list(nltk.bigrams(bigram_tokens))
    #trigram_tuples = list(nltk.trigrams(trigram_tokens))
    
    tokens = []
    unigram_p = defaultdict(float)
    unigram_c = defaultdict(float)
    bigram_p  = defaultdict(float)
    trigram_p = defaultdict(float)
    trigram_c = defaultdict(float)
    bigram_c =  defaultdict(float)
    
    for line in training_corpus:
        
        #The tokenizer used below is like the nltk WhitespaceTokenizer
        
        tokens = line.strip().split(' ') #(nltk.word_tokenize(line))
        
        tokens += [''+STOP_SYMBOL]
        
        for word in tokens:
            unigram_c[(word,)] += 1.0
                      
        tokens = [''+START_SYMBOL] + tokens
        
        bigram_tuples = (tuple(nltk.bigrams(tokens)))
                
        tokens = [''+START_SYMBOL] + tokens
        trigram_tuples = (tuple(nltk.trigrams(tokens)))
        
        for pair in bigram_tuples:
            bigram_c[pair] += 1.0
            
        for triple in trigram_tuples:
            trigram_c[triple] += 1.0
                         
    length = sum(unigram_c.values())
    print str(length)
    
    for  key in unigram_c:
		unigram_p[key] = math.log(unigram_c[key]/length,2).real
    for key in bigram_c:
		if key[0] == START_SYMBOL:
			bigram_p[key] = math.log(bigram_c[key]/unigram_c[(STOP_SYMBOL,)],2).real
		else:	
			bigram_p[key] = math.log(bigram_c[key]/unigram_c[(key[0],)],2).real
    for keys in trigram_c:
		if keys[0] == START_SYMBOL and keys[1] == START_SYMBOL:
			trigram_p[keys] = math.log(trigram_c[keys],2).real - math.log(unigram_c[(STOP_SYMBOL,)],2).real
		else:	
			trigram_p[keys] = math.log(trigram_c[keys],2).real- math.log(bigram_c[(keys[0], keys[1])],2).real
		
    return unigram_p, bigram_p, trigram_p
    

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    
    scores = []
    for sentence in corpus:
                
        tokens = []
        line_score = 0.0
        
        for i in range(3-n,2):
            tokens += [''+START_SYMBOL] 
        tokens = tokens + sentence.strip().split(' ') #(nltk.whitespaceTokenizer(sentence))
        tokens = tokens + [''+STOP_SYMBOL]
		
        #The number of n_grams is equal to the size of the sentence - n 
        #Here we add the 1 because the range does not get to the final value.
        for i in range(0,len(tokens)-n+1):
            key = ()
            #an inner loop to build the curent n_gram then fetch it's probability from the dictionary   
            for j in range(i,i+n):
                key += (tokens[j],)
            line_score += ngram_p.get((key),MINUS_INFINITY_SENTENCE_LOG_PROB) #The get method takes two arguments the key and the default incase key is not found. If key found it returns the value pointed to by that key
        
        scores.append(line_score)
        
    print str(scores)
    return scores
        
# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    
    for sentence in corpus:
        tokens = sentence.strip().split(' ') #nltk.word_tokenize(sentence)
        tokens = [START_SYMBOL,START_SYMBOL] + tokens + [STOP_SYMBOL]
        mytrigrams = tuple(nltk.trigrams(tokens))

        total_prob = 0.0
        fail = False

        for trigram in mytrigrams:
            bigram = (trigram[1], trigram[2])
            unigram = (trigram[2], )

            if trigram in trigrams:
                trigram_prob = 2**trigrams[trigram]
            else:
                trigram_prob = 0.0
            if bigram in bigrams:
                bigram_prob = 2**bigrams[bigram]
            else:
                bigram_prob = 0.0
            if unigram in unigrams:
                unigram_prob = 2**unigrams[unigram]
            else:
                unigram_prob = 0.0

            prob = trigram_prob + bigram_prob + unigram_prob
            if prob == 0.0:
                fail = True
            else:
                total_prob += math.log(prob,2) + math.log(1.0/3.0, 2)

        if fail:
            scores.append(-1000)
        else:
            scores.append(total_prob.real)
    return scores
    

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
