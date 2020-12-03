PERFORMANCE OF THE BAD FEATURES 
--------------------------------
UAS : 0.23023302131
LAS : 0.125273849831

EVALUATION OF MY IMPROVED_FEATURES MODEL
----------------------------------------

SWEDISH TEST DATA
UAS : 0.769368651663
LAS : 0.652857996415

DANISH TEST DATA
UAS : 0.875848303393
LAS : 0.823552894212

In general : the Oracle trains a better model for the Danish Data than the Swedish Data

COMPLEXITY OF THE ARC-EAGER DEPENDENCY PARSER
---------------------------------------------
Can be viewed as a quadratic Algorithm (O(n^2)) where n is the number of words in a given sentence 
Since for every word, it must check in the list of all other words the dependency relationships. This 
value should be less than the square of n though.

