import A
from sklearn.feature_extraction import DictVectorizer

import libw.Universal_tagger as UniversalTagger
import nltk
from sklearn import svm
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
import math
from sklearn.feature_selection import SelectKBest, chi2


# You might change the window size
window_size = 15

# controls the word features
WORD_WINDOW = 5
WORD_HEAD = False
USE_WORD_FREQS_INSTEAD_OF_WORDS = False
USE_RELEVANCY_SCORES = False

# controls the POS features
FORCE_TAGGER_USE = False
POS_WINDOW = 0
POS_HEAD = False

REMOVE_PUNCTUATION = False
REMOVE_STOP_WORDS = False
STEM = False

# Part C
SYN_WINDOW = 0
ADD_SYNONYMS = False
ADD_HYPERNYMS = False
ADD_HYPONYMS = False

USE_FEATURE_SELECTION = False


regex = re.compile('[%s]' % re.escape(string.punctuation))

# B.1.a,b,c,d
def extract_features(data, tagger=None, stemmer=None, relevance_key=None):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}

    # implement your code here
    for (instance_id, left_context, head, right_context, sense_id) in data:
        vector = {}
        # left_tags, head_tag, right_tags = cached_pos_tags[instance_id]
    
        # collapse punctuated words
        if REMOVE_PUNCTUATION:
            left_context = collapse_joint_words(left_context)
            right_context = collapse_joint_words(right_context)
    
        # prepare feature information
        left_tokens = nltk.word_tokenize(left_context)
        right_tokens = nltk.word_tokenize(right_context)
    
        if REMOVE_STOP_WORDS:
            left_tokens = remove_stopwords(left_tokens)
            right_tokens = remove_stopwords(right_tokens)
    
        left_to_tag = left_tokens
        right_to_tag = right_tokens
    
        if STEM:
            left_tokens = stem(stemmer, left_tokens)
            right_tokens = stem(stemmer, right_tokens)
    
        # add features
        word_head = head if WORD_HEAD else None
        pos_head = tagger.tag([head]) if tagger and POS_HEAD else None
    
        if relevance_key:
            left_tokens = add_most_frequent_word_features_to_vector(vector, left_tokens, WORD_WINDOW, relevance_key, 'left')
            right_tokens = add_most_frequent_word_features_to_vector(vector, right_tokens, WORD_WINDOW, relevance_key, 'right')
        else:
            if USE_WORD_FREQS_INSTEAD_OF_WORDS:
                add_k_word_features_count_to_vector(vector, left_tokens, right_tokens, WORD_WINDOW, word_head)
            else:
                add_k_word_features_to_vector(vector, left_tokens, right_tokens, WORD_WINDOW, word_head)
    
        if tagger:
            add_synonym_counts(tagger, left_tokens, right_tokens, vector, SYN_WINDOW)
            add_k_word_POS_features_to_vector(vector, left_to_tag, right_to_tag, POS_WINDOW, tagger ,pos_head)
    
        # track results
        features[instance_id] = vector
        labels[instance_id] = sense_id

    return features, labels


#  Adds wb1 for 1st word before head and wa1 for first word after head... to +-n words
def add_most_frequent_word_features_to_vector(vector, tokens, window_size, relevance_key, key_prefix):

    words = []

    # remove duplicates
    context = list(set(tokens))
    for word in context:
        score = relevance_key[word] if word in relevance_key else -10000
        words.append([score, word])

    # define key function
    def getKey(item):
        return item[0]

    # sort scores
    words = sorted(words, key=getKey, reverse=True)

    # hash scores and return the lookup
    new_context = []
    for idx, tup in enumerate(words):
        score = tup[0]
        word = tup[1]
        new_context.append(word)
        key = 'freq_' + key_prefix + str(idx)
        vector[key] = word

        if idx == window_size-1:
            break
    return new_context


def top_relevant_words_from_data(data):

    # counts of c in a sense
    c_s_freqs = {}

    # total count of c across all contexts
    c_freqs = {}
    for (instance_id, left_context, head, right_context, sense_id) in data:

        # remove punctuation
        left_context = collapse_joint_words(left_context)
        right_context = collapse_joint_words(right_context)

        # tokenize
        left_tokens = nltk.word_tokenize(left_context)
        right_tokens = nltk.word_tokenize(right_context)

        # remove stop words
        left_tokens = remove_stopwords(left_tokens)
        right_tokens = remove_stopwords(right_tokens)

        context = left_tokens + right_tokens

        # remove duplicates so we don't double count
        context = list(set(context))

        for word in context:

            # count times this word appears in all contexts
            c_freqs[word] = c_freqs[word] + 1 if word in c_freqs else 1

            # count union of c and s
            c_s = (sense_id, word)
            c_s_freqs[c_s] = c_s_freqs[c_s] + 1 if c_s in c_s_freqs else 1

    scores = []
    for tup in c_s_freqs:
        sense_id, word = tup
        word_freq_in_context = c_s_freqs[tup]
        total_word_freq_across_all_contexts = c_freqs[word]
        num = word_freq_in_context
        denom = float(total_word_freq_across_all_contexts-word_freq_in_context)
        if denom > 0:
            prob = num/denom
            value = math.log(prob, 2)
        else:
            value = -10000

        scores.append([value, word])

    # define key function
    def getKey(item):
        return item[0]

    # sort scores
    scores = sorted(scores, key=getKey, reverse=True)

    # hash scores and return the lookup
    key = {}
    for tup in scores:
        score = tup[0]
        word = tup[1]
        key[word] = score

    return key


def wordnet_tag_from_penn_tag(tag):
    key_map = {
        'N': 'n',
        'J': 'a',
        'R': 'r',
        'V': 'v'
    }
    a = tag[0]
    return key_map[a] if a in key_map else None


def stem(stemmer, words):
    stemmed = [stemmer.stem(word) for word in words]
    return stemmed


def collapse_joint_words(sentence):
    text = regex.sub('', sentence)
    return text


def add_synonym_counts(tagger, left_tokens, right_tokens, vector, window):
    words = A.k_nearest_words_vector_from_tokens(left_tokens, right_tokens, window)

    for w in words:
        tagged = tagger.tag([w])
        word, tag = tagged[0]
        tag = wordnet_tag_from_penn_tag(tag)
        synonyms = wordnet.synsets(w, pos=tag)
        for synset in synonyms:

            if ADD_SYNONYMS:
                name = synset.name()
                vector[name] = vector[name]+1 if name in vector else 1

            if ADD_HYPONYMS:
                for idx, hypo in enumerate(synset.hyponyms()):
                    name = hypo.name()
                    vector[name] = vector[name]+1 if name in vector else 1

            if ADD_HYPERNYMS:
                for idx, hypper in enumerate(synset.hypernyms()):
                    name = hypper.name()
                    vector[name] = vector[name]+1 if name in vector else 1


# remove stop words
def remove_stopwords(word_list):
    filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    return filtered_words


#  Adds wb1 for 1st word before head and wa1 for first word after head... to +-n words
def add_k_word_features_count_to_vector(vector, left_tokens, right_tokens, window_size, head=None):
    words = A.k_nearest_words_vector_from_tokens(left_tokens, right_tokens, window_size)
    for word in words:
        vector[word] = vector[word] + 1 if word in vector else 1

    if head:
        vector[head] = 1


#  Adds wb1 for 1st word before head and wa1 for first word after head... to +-n words
def add_k_word_features_to_vector(vector, left_tokens, right_tokens, window_size, head=None):
    words = A.k_nearest_words_vector_from_tokens(left_tokens, right_tokens, window_size)
    mid = len(words)/2
    left = words[:mid]
    right = words[mid:]
    for idx, word in enumerate(left):
        key = 'w_b' + str(len(left) - idx)
        vector[key] = word

    for idx, word in enumerate(right):
        key = 'w_a' + str(idx+1)
        vector[key] = word

    if head:
        key = 'w_head'
        vector[key] = head



#  Adds wb1 for 1st word before head and wa1 for first word after head... to +-n words
def add_k_word_POS_features_to_vector(vector, left_tokens, right_tokens, window_size, tagger, head_tag=None):

    words = A.k_nearest_words_vector_from_tokens(left_tokens, right_tokens, window_size)
    mid = len(words)/2
    left = words[:mid]
    right = words[mid:]

    left_tagged = tagger.tag(left)
    right_tagged = tagger.tag(right)

    for idx, (word, tag) in enumerate(left_tagged):
        key = 'pos_b' + str(len(left_tagged) - idx)
        vector[key] = tag

    for idx, (word, tag) in enumerate(right_tagged):
        key = 'pos_a' + str(idx+1)
        vector[key] = tag

    # add POS tag for head
    if head_tag:
        key = 'pos_head'
        word, tag = head_tag[0]
        vector[key] = tag


# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''
    if not USE_FEATURE_SELECTION:
        return X_train, X_test

    chi_square = SelectKBest(chi2, k=10)
    matrix_X_train = []
    matrix_Y_train = []
    matrix_X_test = []
    for sent_id in X_train:
        x = X_train[sent_id]
        matrix_X_train.append(x)
        y = y_train[sent_id]
        matrix_Y_train.append(y)

    for sent_id in X_test:
        x_t = X_test[sent_id]
        matrix_X_test.append(x_t)

    x_trained = chi_square.fit_transform(matrix_X_train, matrix_Y_train)
    x_tested = chi_square.transform(matrix_X_test)

    count = 0
    for sent_id in X_train:
        X_train[sent_id] = x_trained[count]
        count += 1

    count = 0
    for sent_id in X_test:
        X_test[sent_id] = x_tested[count]

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []
    # create x, y lists from training datas

    x_train_list = []
    y_train_list = [] #A.x_y_lists_from_training(X_train, y_train)
    
    for obj in X_train:
        x_train_list.append(X_train[obj])
        y_train_list.append(y_train[obj])


    # train svm
    print 'training svm...'
    svm_clf = svm.LinearSVC()
    svm_clf.fit(x_train_list, y_train_list)

    # predict svm results
    print 'predicting svm...'
    results = A.predictions_from_data(svm_clf, X_test)

    # implement your code here

    return results

# saves tagges corpus for faster processing
def tag_and_save(train, test, language):
    train_tagged = {}
    test_tagged = {}
    print 'training tagger'
    tagger = UniversalTagger.EnglishTagger()
    counter = 0
    for lexelt in train:

        # train
        for (instance_id, left_context, head, right_context, sense_id) in train[lexelt]:

            counter += 1
            print 'saving train' + str(counter)

            # prepare feature information
            left_tokens = nltk.word_tokenize(left_context)
            right_tokens = nltk.word_tokenize(right_context)

            left_tags = tagger.tag(left_tokens)
            middle = tagger.tag([head])
            right_tags = tagger.tag(right_tokens)

            # keys = {}
            # for word, tag in left_tags:
            #     keys[word] = tag
            #
            # for word, tag in right_tags:
            #     keys[word] = tag
            #
            # head, head_tag = middle
            # keys[head] = head_tag

            # add features
            train_tagged[instance_id] = (left_tags, middle, right_tags)

        # test
        for (instance_id, left_context, head, right_context, sense_id) in test[lexelt]:

            counter += 1
            print 'saving test' + str(counter)

            # prepare feature information
            left_tokens = nltk.word_tokenize(left_context)
            right_tokens = nltk.word_tokenize(right_context)

            left_tags = tagger.tag(left_tokens)
            middle = tagger.tag([head])
            right_tags = tagger.tag(right_tokens)

            # add features
            test_tagged[instance_id] = (left_tags, middle, right_tags)

    # save with pickle
    train = language + '-train.p'
    test = language + '-test.p'

    print 'saving train'
    print 'saving test'
    pickle.dump(train_tagged, open(train, 'wb'))
    pickle.dump(test_tagged, open(test, 'wb'))
    print 'saved pickle'



# run part B
def run(train, test, language, answer):
    results = {}
    tagger = None
    if POS_WINDOW > 0 or POS_HEAD or FORCE_TAGGER_USE:
        tagger = UniversalTagger.EnglishTagger()
        if language is 'Spanish':
            tagger = UniversalTagger.SpanishTagger()

        if language is 'Catalan':
            tagger = UniversalTagger.CatalanTagger()

    stemmer = None
    if STEM:
        stemmer = PorterStemmer()

    for lexelt in train:
        relevance_key = None
        if USE_RELEVANCY_SCORES:
            relevance_key = top_relevant_words_from_data(train[lexelt])

        
        train_features, y_train = extract_features(train[lexelt], tagger, stemmer, relevance_key)
        test_features, _ = extract_features(test[lexelt], tagger, stemmer, relevance_key)
        
        #train_features, y_train = extract_features(train[lexelt])
        #test_features, _ = extract_features(test[lexelt])

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)