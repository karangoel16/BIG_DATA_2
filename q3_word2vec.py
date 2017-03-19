import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    y = np.linalg.norm(x,axis=1,keepdims=True)
    x /= y
    return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!    
    ### YOUR CODE HERE 
    N,D=outputVectors.shape #taking the shape of the output Vector to be built later
    Y=np.zeros(N);
    Y[target]=1;
    prob = softmax(np.matmul(outputVectors,predicted))
    cost = -np.sum(Y*np.log(prob))
    diff= prob - Y#this is for the change in the pro 
    grad     = diff.reshape((N,1)) * predicted.reshape((1,D)) #gradient with respect to the all other words , it brings back the dimension , (N,D) , so that we dont have problem
    gradPred = (diff.reshape((1,N)).dot(outputVectors)).flatten()#gradient with respect to the predicted words vector     
    return cost, gradPred, grad

"""Checked till here version -1 """

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE
    grad = np.zeros(outputVectors.shape)#this is to provide the zero matrix to be used to save the values later
    gradPred = np.zeros(predicted.shape)
    indices = [target]
    for k in range(K):
        new_index = dataset.sampleTokenIdx()
        while new_index == target:
            new_index = dataset.sampleTokenIdx()
        indices += [new_index]
    labels = np.array([1] + [-1 for k in range(K)])
    temp_vec = outputVectors[indices,:]
    t = sigmoid(temp_vec.dot(predicted) * labels)
    cost = -np.sum(np.log(t))
    diff = labels * (t - 1)
    gradPred = diff.reshape((1,K+1)).dot(temp_vec).flatten()
    temp = diff.reshape((K+1,1)).dot(predicted.reshape((1,predicted.shape[0])))
    for k in range(K+1):
        grad[indices[k]] += temp[k,:]
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE
#in this output is the inputVector and hence prediction would depend on the input vector
    current_Index = tokens[currentWord]
    predicted = inputVectors[current_Index, :]
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for cwd in contextWords:
        idx = tokens[cwd]
        temp_cost, gradpred, grad_out = word2vecCostAndGradient(predicted, idx, outputVectors, dataset)
        cost =cost+ temp_cost
        gradOut =gradOut+ grad_out
        gradIn[current_Index, :] += gradpred
    return cost, gradIn, gradOut
"""This part is swapped in case of CBOW as in that output and input vectors are interchanged and so is the prediction"""
        

    ### END YOUR CODE
    

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient,pr=False):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    cost    = 0.0
    gradIn  = np.zeros_like(inputVectors)
    gradOut = np.zeros_like(outputVectors)
    ### YOUR CODE HERE
    current_index     = tokens[currentWord] #this here gives the index of the current word which we will have sent to the code
    onehot    = np.zeros((2*C, len(tokens)))#providing one hot conversion from the linear available input tokens
    for i, word in enumerate(contextWords):
        onehot[i, tokens[word]] += 1.
    if pr:#this will work only when we want the print statement to be true , else this part of the code won't work
        print onehot #this could be used to print the one hot module and check its testing
    diff = np.dot(onehot, inputVectors)
    predicted = 0.5 / C * np.sum(diff, axis=0)
    cost, gradPred, gradOut = word2vecCostAndGradient(predicted, current_index, outputVectors,dataset)	
    gradIn = np.zeros(inputVectors.shape)
    for word in contextWords:
        gradIn[tokens[word]] += 0.5 / C * gradPred
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
