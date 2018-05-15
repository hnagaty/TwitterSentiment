#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 08:01:57 2018

@author: hnagaty
"""

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from nltk.stem.porter import PorterStemmer
import itertools


# read the data
tweet1 = pd.read_table("../twitter-2013train.txt",sep="\t",header=None,names=("ID","Label","Value"))
tweet2 = pd.read_table("../twitter-2015train.txt",sep="\t",header=None,names=("ID","Label","Value"))
tweet3 = pd.read_table("../twitter-2016train.txt",sep="\t",header=None,names=("ID","Label","Value"))
allTweetNames=[tweet1,tweet2,tweet3]
allTweets=pd.concat(allTweetNames,ignore_index=True)
del (tweet1)
del (tweet2)
del (tweet3)
del (allTweetNames)

#Some very basic cleaning
unicodeDict={"\\\\u002c":",",
             "\\\\u2019":"'"}
allTweets.replace(unicodeDict,inplace=True,regex=True)



# See the data
print("\n",allTweets[:5])
print(allTweets.info())
print("Checking for class balance")
print(allTweets["Label"].value_counts())

#%%

def textVectorize(myText,myVectorizer): 
    from sklearn.model_selection import train_test_split

    ftweetsTrain,ftweetsTest = train_test_split(myText,test_size=0.1,random_state=213)
    myTrainVals=ftweetsTrain["Value"]
    myTrainLabs=ftweetsTrain["Label"]
    myTestVals=ftweetsTest["Value"]
    myTestLabs=ftweetsTest["Label"]    
    vectorizer = myVectorizer.fit(myTrainVals)
    myTrainVec = vectorizer.transform(myTrainVals)
    myTestVec = vectorizer.transform(myTestVals)   
    myFeatures=vectorizer.get_feature_names()
    return(myTrainVec,myTrainLabs,myTestVec,myTestLabs,myFeatures)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # Downloaded from sklearn website
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def TrainAndTest(myModel,ftrainVec,ftrainLab,ftestVec,ftestLab):
    from sklearn import metrics

    print("\n\nFitting & testing a new model ........")
    print("Testing on test split .....")
    print(str(myModel))
    print ("\nNo. of features: %d" %ftrainVec.shape[1])
    print ("Train set: %d" %ftrainVec.shape[0])
    print ("Test set: %d" %ftestVec.shape[0])
    myModel.fit(ftrainVec,ftrainLab)
    Predicted = myModel.predict(ftestVec)
    acc = metrics.accuracy_score(ftestLab,Predicted)
    f1=metrics.f1_score(ftestLab,Predicted,average="weighted")
    print ("Accuracy %0.2f%%" %(acc*100))
    print ("F1 %0.2f%%" %(f1*100))
    print ("\nClassifcation Metrics:")    
    print (metrics.classification_report(ftestLab,Predicted))
    #print("n\Confusion Matrix:")
    #print (metrics.confusion_matrix(ftestLab,Predicted))    

def TrainAndTestCV(myModel,ftrainVec,ftrainLab):
    from sklearn.model_selection import cross_val_score
    print ("\n\n\n\nCV for a new model.....\n")
    print(str(myModel))
    print ("\nNo. of features: %d" %ftrainVec.shape[1])
    print ("Train set: %d" %ftrainVec.shape[0])
    score=cross_val_score(model,ftrainVec,ftrainLab,cv=10)
    print("\nCross Validation Score:")
    print("Accuracy: %0.2f%% (with sd=%0.2f%%)" % (score.mean()*100, score.std()*100))

def TrainAndTestCV2(myModel,ftrainVec,ftrainLab):
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    from sklearn import metrics
    print ("\n\n\n\nCV for a new model.....\n")
    print(str(myModel))
    print ("\nNo. of features: %d" %ftrainVec.shape[1])
    print ("Train set: %d" %ftrainVec.shape[0])
    sc = ['precision_weighted', 'recall_weighted', 'f1_weighted','accuracy']
    allScores = cross_validate(myModel, ftrainVec, ftrainLab, cv=10, scoring= sc)  
    predicted = cross_val_predict(myModel, ftrainVec, ftrainLab, cv=10)
    print("\nCross Validation Score:")
    print("Accuracy: %0.2f%% (with sd=%0.2f%%)" % (allScores["test_accuracy"].mean()*100, allScores["test_accuracy"].std()*100))
    print("F1 Weighted: %0.2f%% (with sd=%0.2f%%)" % (allScores["test_f1_weighted"].mean()*100, allScores["test_f1_weighted"].std()*100))
    print ("\nClassifcation Metrics:")    
    cReport=metrics.classification_report(ftrainLab,predicted)
    print (cReport)
    print("\nConfusion Matrix:")
    class_names=["negative","neutral","positive"]
    cMatrix=confusion_matrix(ftrainLab,predicted,labels=class_names)
    print(cMatrix)
    #cm = nltk.ConfusionMatrix(ftrainLab,predicted)
    #print(cm)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cMatrix, classes=class_names,
                     title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cMatrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


def tokenize1(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

def tokenize2(text):
    ftokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
    tokens = ftokenizer.tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems
    
#%%    
tokenizer2 = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)


# Base run: Naive Bayes with count vectorizer
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=4, ngram_range=(1, 4))
model=naive_bayes.MultinomialNB()
%time trainVec,trainLab,testVec,testLab,trainFeatures=textVectorize(allTweets,vectorizer)
%time TrainAndTest(model,trainVec,trainLab,testVec,testLab)
%time TrainAndTestCV(model,trainVec,trainLab)
%time TrainAndTestCV2(model,trainVec,trainLab)

#Testing various Vectorizers & tokenizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect1 = CountVectorizer(min_df=3, ngram_range=(1, 2))
vect2 = CountVectorizer(tokenizer=tokenize1,stop_words="english", min_df=3, ngram_range=(1, 2))
vect3 = CountVectorizer(tokenizer=tokenizer2.tokenize,min_df=2) # this seems to be the best one
vect4 = TfidfVectorizer(min_df=3, ngram_range=(1, 2))
allVectorizers=(vect1,vect2,vect3,vect4)
for testVect in allVectorizers:
    print("\n\n\nTesting another Vectorizer")
    print(testVect)
    trainVec,trainLab,testVec,testLab,trainFeatures=textVectorize(allTweets,testVect)
    TrainAndTest(model,trainVec,trainLab,testVec,testLab)

#%%
# Testing other settings for vect3 (as it was the one with the best results)
print("\n\nA new run:")
#vect3 = CountVectorizer(tokenizer=tokenizer2.tokenize,min_df=2,stop_words="english")   
vect3 = CountVectorizer(tokenizer=tokenize2,min_df=2,stop_words="english")
model=naive_bayes.MultinomialNB()
trainVec,trainLab,testVec,testLab,trainFeatures=textVectorize(allTweets,vect3)
#%time TrainAndTestCV(model,trainVec,trainLab)
TrainAndTestCV2(model,trainVec,trainLab)
TrainAndTest(model,trainVec,trainLab,testVec,testLab)

#%%
# Take care of emoticons & other characters
# It reduced the accuracy
# Most probably because a tweeter tokenizer is already used
unicodeDict={"\\\\u002c":",","\\\\u2019":"'",":\)":" smile ",";\)":" smile ",":D":" laugh ",":\(":" sad "}
allTweets.replace(unicodeDict,inplace=True,regex=True)
trainVec,trainLab,testVec,testLab=textVectorize(allTweets,vect3)
TrainAndTest(model,trainVec,trainLab,testVec,testLab)

#%%
from sklearn.model_selection import cross_val_score
trainVec,trainLab,testVec,testLab=textVectorize(allTweets)
testmodel = naive_bayes.MultinomialNB()
print("\n\n\n CV for a test model ....\n")
print(str(testmodel))
score=cross_val_score(testmodel,trainVec,trainLab,cv=10)
fscore = cross_val_score(testmodel,trainVec,trainLab,cv=10, scoring='f1_weighted'  )
print("Accuracy: %0.2f%% (with sd=%0.2f%%)" % (score.mean()*100, score.std()*100))
print("F1: %0.2f%% (with sd=%0.2f%%)" % (fscore.mean()*100, fscore.std()*100))

TrainAndTest(testmodel,trainVec,trainLab,testVec,testLab)

#%%
# reducing features using chi2 metric
from sklearn.feature_selection import chi2
from scipy.stats import describe
print("\n\n\nReducing no. of features.....")
vect3 = CountVectorizer(tokenizer=tokenize2,min_df=2,stop_words="english")  
trainVec,trainLab,testVec,testLab,trainFeatures=textVectorize(allTweets,vect3)
chiVals=chi2(trainVec,trainLab)
print(describe(chiVals[0]))
chiFilter=chiVals[1]<0.15  # select features with p<5%
reducedTrainVec=trainVec[:,chiFilter]
reducedTestVec=testVec[:,chiFilter]
reducedFeatures=np.array(trainFeatures)[chiFilter].tolist()
#%time TrainAndTestCV(model,reducedTrainVec,trainLab)
TrainAndTestCV2(model,reducedTrainVec,trainLab)
TrainAndTest(model,reducedTrainVec,trainLab,reducedTestVec,testLab)
#%%

#Testing various models
testModel1 = sklearn.naive_bayes.MultinomialNB()
testModel2 = sklearn.svm.SVC(kernel="linear")
testModel3 = sklearn.svm.SVC(kernel="rbf")
testModel4 = sklearn.linear_model.LogisticRegression(class_weight="balanced")
#testModel5=descision tree
allModels=[testModel1,testModel2,testModel3,testModel4]

for model in allModels:
    %time TrainAndTestCV2(model,reducedTrainVec,trainLab)


# Searching for hyperparameters
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
clf = GridSearchCV(SVC(), tuned_parameters, cv=10,scoring='f1_weighted')
%time clf.fit(reducedTrainVec,trainLab)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = testLab, clf.predict(testVec)
print(classification_report(y_true, y_pred))
print()


#%%
#read the Kaggle test data
KaggleTestTweets = pd.read_table("../new_english_test.csv",sep=",")
tweets=KaggleTestTweets["tweet"]
ids=KaggleTestTweets["id"]
unicodeDict={"\\\\u002c":",",
             "\\\\u2019":"'"}
tweets.replace(unicodeDict,inplace=True,regex=True)
tweetsVec=vect3.transform(tweets)
reducedTweetsVec=tweetsVec[:,chiFilter]
PredictedSentiment=model.predict(reducedTweetsVec)
PredictedPD = pd.DataFrame(list(zip(ids,PredictedSentiment)),columns=["id","sentiment"])
PredictedPD.to_csv("outfile.csv",index=False)
