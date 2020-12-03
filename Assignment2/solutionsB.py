import sys
import nltk
import math
import time
from collections import defaultdict
import itertools

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for line in brown_train:
		tags = []
		words = []
		tokens = line.split(' ')
		for token in tokens:
			tag = token.split('/')[-1]
			tags.append(tag)
			words.append(token[:-(len(tag)+1)])
            
		brown_words.append(words)
		brown_tags.append(tags)
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = defaultdict(float)
       
    tokens = []
    bigram_c = defaultdict(float)
    trigram_c = defaultdict(float)

    for line in brown_tags:
		tokens = [START_SYMBOL] + line + [STOP_SYMBOL]
		bigram_tuples = (list(nltk.bigrams(tokens)))
		tokens = [START_SYMBOL] + tokens
		trigram_tuples = (list(nltk.trigrams(tokens)))
  
		for pair in bigram_tuples:
			bigram_c[pair] += 1.0
		for triple in trigram_tuples:
			trigram_c[triple] += 1.0
		for key in trigram_c:
			if key[0] == START_SYMBOL and key[1] == START_SYMBOL:
				q_values[key] = math.log(trigram_c[key],2).real - math.log(len(brown_tags),2).real
			else:
				q_values[key] = math.log(trigram_c[key],2).real - math.log(bigram_c[(key[0],key[1])],2).real 
    
    
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    words = defaultdict(float)
    for line in brown_words:
		for item in line:
			words[item] += 1
   
    for item in words:
		if words[item] > RARE_WORD_MAX_FREQ:
			known_words.add(item)
    
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    
    for line in brown_words:
		sentence = []
		for word in line:
			if word not in known_words:
				sentence.append(RARE_SYMBOL)
			else:
				sentence.append(word)
		sentence.append(STOP_SYMBOL)
		brown_words_rare.append(sentence)
    
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[0:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = defaultdict(float)
    taglist = set([])
    
    
    #evalues = defaultdict(float)
    tags = defaultdict(float)
	#taglist = []
    for linew, linet in zip(brown_words_rare, brown_tags):
		for tag, word in zip(linet, linew):
			e_values[word, tag] +=1.0
			tags[tag] += 1.0
    for item in e_values:
		e_values[item] = math.log(e_values[item]/tags[item[1]],2)
    for item in tags:
		taglist.add(item)
    
    
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    pi = defaultdict(float)
    bp = {}
    bp[(-1,START_SYMBOL,START_SYMBOL)] = START_SYMBOL
    pi[(-1,START_SYMBOL,START_SYMBOL)] = 0.0
    for line in brown_dev_words:
		#tokens_orig = line.strip.split(' ') #nltk.word_tokenize(line)	
        
        tokens = [w if w in known_words else RARE_SYMBOL for w in line]
        tokens = tokens
		# k = 1 case
        for w in taglist:
			pi[(0, START_SYMBOL, w)] = pi[(-1,START_SYMBOL,START_SYMBOL)] + q_values.get((START_SYMBOL, START_SYMBOL, w), LOG_PROB_OF_ZERO) + e_values.get((tokens[0], w), LOG_PROB_OF_ZERO)
			bp[(0, START_SYMBOL, w)] = START_SYMBOL

		# k = 2 case
        for (w, u) in itertools.product(taglist, taglist):
			key = (START_SYMBOL, w, u)
			pi[(1, w, u)] = pi.get((0, START_SYMBOL, w), LOG_PROB_OF_ZERO) + q_values.get(key, LOG_PROB_OF_ZERO) + e_values.get((tokens[1], u), LOG_PROB_OF_ZERO)
			bp[(1, w, u)] = START_SYMBOL 
		#k >= 2 case
        for k in range (2, len(tokens)):
			for (u, v) in itertools.product(taglist, taglist):
				max_prob = -float('Inf')
				max_tag = ""
				for w in taglist:
					score = pi.get((k-1, w, u), LOG_PROB_OF_ZERO) + q_values.get((w,u,v), LOG_PROB_OF_ZERO) + e_values.get((tokens[k], v), LOG_PROB_OF_ZERO)
					if(score > max_prob):
						max_prob = score
						max_tag = w
				bp[(k,u,v)] = max_tag
				pi[(k,u,v)] = max_prob
		
        max_prob = -float('Inf')
		#finding the max probability of last two tags
        for (u,v) in itertools.product(taglist,taglist):
			score = pi.get((len(line)-1, u, v),LOG_PROB_OF_ZERO) + q_values.get((u,v,STOP_SYMBOL),LOG_PROB_OF_ZERO) 
			if score >  max_prob:
				max_prob = score
				u_max = u
				v_max = v
		#append tags in reverse order
        tags = []
        tags.append(v_max)
        tags.append(u_max)
        count = 0
        for k in range(len(line) - 3, -1, -1):
			tags.append(bp[(k + 2, tags[count+1], tags[count])])
			count +=1
        tagged_sentence = ""
		#reverse tags
        tags.reverse()
		#stringify tags paired with word without start and stop symbols
        for k in range(0, len(line)):
			tagged_sentence = tagged_sentence + line[k] + "/" + str(tags[k]) + " "
        tagged_sentence += "\n"
        tagged.append(tagged_sentence)	
    return tagged		
    

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    #training = bn.tagged_sents(tagset = 'universal')
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff = bigram_tagger)
    for sentence in brown_dev_words:
       		tagged_sentence = trigram_tagger.tag(sentence)
		wordlist = []
        	for tag in tagged_sentence:
			wordlist.append(tag[0] + "/" + tag[1])
		tagged.append(wordlist)
    
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(str(sentence))
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)
    
    #This is just to check the tags and the Words that have been split
    
    q5_output(str(brown_words), OUTPUT_PATH + 'B1_words.txt')
    q5_output(str(brown_tags), OUTPUT_PATH + 'B1_tagged.txt')
        
    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()
#    i=0
#    for line in brown_dev :
#        if i<3:
#            print line,'\n'
#            i+=1

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])
        
    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
