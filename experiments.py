import sys
import pickle 
import nltk
import csv  
import math
import random
import re
from nb import nbClassifier
from svm import svmClassifier
from lr import lrClassifier
from knn import knnClassifier
from id3 import id3Classifier

def load_nb():
    classifier = nbClassifier
    return classifier


def load_id3():
    classifier = id3Classifier
    return classifier


def load_svm():
    classifier = svmClassifier
    return classifier


def load_lr():
    classifier = lrClassifier
    return classifier


def load_knn():
    classifier = knnClassifier
    return classifier


def load_goodwords(filename): #good words for passive attack
    goodWords = []
    
    with open(filename, "r", encoding='UTF8') as f:
        for line in f:
            goodWords.append(line.strip("\n"))
            
    return goodWords

def load_spam():
    spam = []
    
    with open("datasets/spam.txt", "r", encoding='UTF8') as f:
        for line in f:
            spam.append(line.strip("\n"))
            
    return spam


def get_word_list(): # get words from dictionary ( random words )
    wordlist = []
    
    with open("datasets/randomWords.txt", "r", encoding='UTF8') as f:
        for line in f:
            wordlist.append(line.strip("\n"))
            
    return wordlist


def get_spam_and_legit():
    spamlist, legitlist = [], []
    
    with open("datasets/testdata.csv", "r", encoding='UTF8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "COMMENT_ID":
                continue
            content = re.sub(r'[^\w\s]', '', row[3]).lower().split(" ")
            if int(row[4]) == 1:
                spamlist.append(content)
            else:
                legitlist.append(content)
                
    return (spamlist, legitlist)


def passive(classifier):
    WORD_LIMIT = 200
    goodWords = load_goodwords('datasets/goodwords.txt')
    spam = load_spam()
    avg = 0
    flag = False
    
    for comment in spam:
        for i in range(WORD_LIMIT):
            comment += " {}".format(random.choice(goodWords))
            if classifier.predict(comment) == False:
                avg += i
                break
            if i == WORD_LIMIT - 1:
                avg += WORD_LIMIT
                
    print(avg / len(spam))
    

def find_witness(spam, legit, classifier):
    length = min([len(spam), len(legit)])
    curr = spam
    i = 0
    flag = False
    prev = curr

    while classifier.predict(" ".join(curr)) == True:
        prev = curr
        for j in range(len(curr)):
            if curr[j] not in spam:
                del curr[j]
                flag = True
                break 
        if not flag:
            for j in range(len(spam)):
                if spam[j] not in curr:
                    curr.append(j)
                    break 
        flag = False
        i += 1
        if i == length:
            return curr, prev
            
    return (curr, prev)


def first_n_words(spam, legit, classifier, wordlist):
    L = []
    to_spam, _ = find_witness(spam, legit, classifier)

    for i in range(100):
        word=random.choice(wordlist)
        if len(word)>8:
            continue
        to_spam.append(word)
        if classifier.predict(" ".join(to_spam)) == False:
            L.append(word)
            to_spam = to_spam[:len(to_spam)-1]
        if len(L)>10:
            break
        
    return L


def run_first_n_words(classifier):
    
    spamlist, legitlist = get_spam_and_legit()
    wordlist = get_word_list()
    length = min([len(spamlist), len(legitlist)])
    n_words = set()
    
    for i in range(length):
        goodList=first_n_words(spamlist[i], legitlist[i], classifier, wordlist)
        for good in goodList:
            n_words.add(good)
        if len(n_words)>200:
            break

    return list(n_words)


def active(classifier):
    WORD_LIMIT = 200
    goodWords = run_first_n_words(classifier)
    spam = load_spam()
    avg = 0
    flag = False
    itrs = 0
    num=0
    check=[]
    add_word=" "
    
    for j in range(len(spam)):
        check.append(0)

    for i in range(WORD_LIMIT):
        j=-1
        add_word += " {}".format(random.choice(goodWords))
        for comment in spam:
            j=j+1
            if check[j]==1:
                continue
            comment += add_word
            if classifier.predict(comment) == False:
                check[j]=1
                num = num+1
        if num*2>len(spam):
            print("good words required : ")
            print(i)
            break
        if i==WORD_LIMIT -1:
            print("LIMIT!")
            print(num, len(spam))
            

def test_nb():
    print("classifier : Naive Bayes ")
    nb=load_nb()
    print("active attack")
    active(nb)
    print("passive attack")
    passive(nb)


def test_svm():
    print("classifier : Support Vector Machine ")
    svm = load_svm()
    print("active attack")
    active(svm)
    print("passive attack")
    passive(svm)


def test_lr():
    print("classifier : Logistic Regression ")
    lr = load_lr()
    print("active attack")
    active(lr)
    print("passive attack")
    passive(lr)


def test_knn():
    print("classifier : K Nearest Neighbor ")
    knn = load_knn()
    print("active attack")
    active(knn)
    print("passive attack")
    passive(knn)


def test_id3():
    print("classifier : id3 ")
    id3 = load_id3()
    print("active attack")
    active(id3)
    print("passive attack")
    passive(id3)
    

if __name__ == "__main__":
    test_nb()
    test_svm()
    test_lr()
    test_knn()
    test_id3()
