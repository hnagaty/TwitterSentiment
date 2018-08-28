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
#import sklearn
from nltk.stem.porter import PorterStemmer
import itertools
import re
import copy as cp
from scipy import sparse
# Import models
import xgboost as xgb
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import string



#%%
#Those functions were actually not used
#They can be removed
def removeUrl(text):
    text2=re.sub(r"http.?://[^\s]+[\s]?","",text)
    return list(text2)
def removeMention(text):
    text2=re.sub(r"@[^\s]+[\s]?","",text)
    return text2
def removeNumber(text):
    text2=re.sub(r"\s?[0-9]+\.?[0-9]*","",text)
    return text2

#%%
def textVectorize(myText,myVectorizer): 
    myVals=myText["Value"]
    myLabs=myText["Label"]    
    vectorizer = myVectorizer.fit(myVals)
    myVec = vectorizer.transform(myVals)
    myFeatures=vectorizer.get_feature_names()
    return(myVec,myLabs,myFeatures)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # I got this function from sklearn website
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


# Tokenizer functions
def tokenize1(text): #nltk standard tokenizer with PorterStemmer
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

def tokenize2(text): #nltk tweet tokenizer with PorterStemmer
    ftokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
    tokens = ftokenizer.tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems
#%%    
#Cleansing & feature extraction functions
def CntAndRemove(s,twtsPD): # counts s (regex) in twts and then removes it
    twtsPD["Value"].replace(s,"HHCNTMEXX",inplace=True,regex=True)
    #c=list(map(lambda st: st.count("HHCNTMEXX"),twtsPD["Value"]))
    c=[line.count("HHCNTMEXX") for line in twtsPD["Value"]]
    print("Total Counts:",sum(c))
    twtsPD["Value"].replace("HHCNTMEXX","",inplace=True,regex=True)    
    return c

def CntByRegex(s,twtsPD): # counts s (regex) in twts and then removes it
    c=([len(re.findall(s,line)) for line in twtsPD["Value"]])
    print("Total Counts:",sum(c))
    return c

def CntAndReplaceByLex(lexi,word,twtsPD): # counts the occurences of words in lexi (file) in each element in the twts(list)
    u="NHPATTERNNG"
    f = open(lexi, 'r')
    lexicons = f.read().splitlines()
    lexs=[re.escape(x) for x in lexicons]
    print("Lexicon size:",len(lexs))
    twtsPD["Value"].replace(lexs,u,inplace=True,regex=True)
    c=[line.count(u) for line in twtsPD["Value"]]
    twtsPD["Value"].replace(u,word,inplace=True,regex=True)    
    return c

def getCnts(lexi,twts): # counts the occurences of words in lexi (file) in each element in the twts(list)
    f = open(lexi, 'r')
    lexicons = f.read().splitlines()
    lexs=[re.escape(x) for x in lexicons]
    print("Lexicon size:",len(lexs))
    fTwts=pd.DataFrame([x.lower() for x in twts],columns=["V"])
    #lexicons=lexs[2000:]
    fTwts["V"].replace(lexs," HHLEGNN ",inplace=True,regex=True)
    c=list(map(lambda st: st.count("HHLEGNN"),fTwts["V"]))
    print("Total Counts:",sum(c))
    return c

def getStemmedCnts(lexi,twts): # counts the occurences of words in lexi (file) in each element in the twts(list)
    f = open(lexi, 'r')
    lexicons = f.read().splitlines()
    lext=list(set(map(lambda x: PorterStemmer().stem(x),lexicons)))
    lexs=[re.escape(x) for x in lext]
    print("Lexicon size:",len(lexs))
    fTwts=pd.DataFrame([" ".join(tokenize1(x)) for x in twts],columns=["V"])
    #lexicons=lexs[2000:]
    fTwts["V"].replace(lexs," HHLEGNN ",inplace=True,regex=True)
    c=list(map(lambda st: st.count("HHLEGNN"),fTwts["V"]))
    print("Total Counts:",sum(c))
    return c

#%%
# read the data
tweet1 = pd.read_table("../twitter-2013train.txt",sep="\t",header=None,names=("ID","Label","Value"))
tweet2 = pd.read_table("../twitter-2015train.txt",sep="\t",header=None,names=("ID","Label","Value"))
tweet3 = pd.read_table("../twitter-2016train.txt",sep="\t",header=None,names=("ID","Label","Value"))
allTweetNames=[tweet1,tweet2,tweet3]
allTweets=pd.concat(allTweetNames,ignore_index=True)
del (tweet1, tweet2, tweet3)
del (allTweetNames)

allTweetsOrg=cp.deepcopy(allTweets)

# See the data
print("Checking for class balance")
print(allTweets["Label"].value_counts())

#%%
# Very basic initial cleasning

unicodeDict={"\\\\u002c":",",
             "\\\\u2019":"'"}
allTweets.replace(unicodeDict,inplace=True,regex=True)



#%%
# Train & test split
from sklearn.model_selection import train_test_split
trainTweets,testTweets = train_test_split(allTweets,test_size=0.1,random_state=213)
trainTweets["Value"].to_csv("trainTwtsBefore.txt",index=False)
testTweets["Value"].to_csv("testTwtsBefore",index=False)

#%%
#remove Urls & emoticons
#emoticons counts will be added later as additional features
trainUrlCnt=CntAndRemove("http.?://[^\s]+[\s]?",trainTweets) # remove Urls
trainPEmots=CntAndReplaceByLex("emotpos.txt","",trainTweets)
trainNEmots=CntAndReplaceByLex("emotneg.txt","", trainTweets)
trainMentionCnt=CntByRegex("@[^\s]+[\s]?",trainTweets)
trainHashCnt=CntByRegex("#[^\s]+[\s]?",trainTweets)
trainExCount = list(trainTweets['Value'].str.count('!'))
trainQCount = list(trainTweets['Value'].str.count('\?'))
trainDotCount = list(trainTweets['Value'].str.count('\.'))
                       
testUrlCnt=CntAndRemove("http.?://[^\s]+[\s]?",testTweets) # remove Urls
testPEmots=CntAndReplaceByLex("emotpos.txt","",testTweets)
testNEmots=CntAndReplaceByLex("emotneg.txt","", testTweets)
testMentionCnt=CntByRegex("@[^s]+[s]?",testTweets)
testHashCnt=CntByRegex("#[^s]+[s]?",testTweets)                        
testExCount = list(testTweets['Value'].str.count('!'))
testQCount = list(testTweets['Value'].str.count('\?'))
testDotCount = list(testTweets['Value'].str.count('\.'))
#%%
# Clear punctuations

trainTweets["Value"].replace("[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']"," ",inplace=True,regex=True) 
trainTweets["Value"].replace("\\\\"," ",inplace=True,regex=True) 
trainNbrCnt=CntAndRemove("\s?[0-9]+\.?[0-9]*",trainTweets) 

testTweets["Value"].replace("[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']"," ",inplace=True,regex=True) 
testTweets["Value"].replace("\\\\"," ",inplace=True,regex=True)
testNbrCnt=CntAndRemove("s?[0-9]+.?[0-9]*",testTweets)

trainTweets["Value"].to_csv("trainTwtsAfter.txt",index=False) #Write final result
testTweets["Value"].to_csv("testTwtsAfter.txt",index=False)
                       
#%%
#BOW
from sklearn.feature_extraction.text import CountVectorizer
vect3 = CountVectorizer(tokenizer=tokenize2,min_df=2,stop_words="english",ngram_range=(1,2))  
trainVec,trainLab,trainFeatures = textVectorize(trainTweets,vect3)
testVec=vect3.transform(testTweets["Value"])
testLab=testTweets["Label"]
                   
#%%
# reducing features using chi2 metric
from sklearn.feature_selection import chi2
chiVals=chi2(trainVec,trainLab)
chiFilter=chiVals[1]<0.1  #this value is selected aTrainAndTest(model,richTrainVec,trainLab,richTestVec,testLab)fter some trial & error
reducedTrainVec=trainVec[:,chiFilter]
reducedTestVec=testVec[:,chiFilter]
reducedFeatures=np.array(trainFeatures)[chiFilter].tolist()
newFeatures=pd.DataFrame(reducedFeatures,chiVals[1][chiFilter],copy=True,columns=["Feature"]).sort_index()

#%%
#Add some features

# On train set
trainPosLex=getStemmedCnts("../positive-words.txt",trainTweets["Value"])
trainNegLex=getStemmedCnts("../negative-words.txt",trainTweets["Value"])
trUpper=list(trainTweets['Value'].str.findall(r'[A-Z]').str.len())
trTweetlength=trainTweets['Value'].apply(len)

ftr1a=sparse.coo_matrix(trainPosLex).transpose() #cnt of +ve words
ftr2a=sparse.coo_matrix(trainNegLex).transpose() #cnt of -ve words
ftr3a=sparse.coo_matrix(trainPEmots).transpose() #cnt of +ve emoticons
ftr4a=sparse.coo_matrix(trainNEmots).transpose() #cnt of -ve emoticons
ftr5a=sparse.coo_matrix(trainUrlCnt).transpose() #Cnt ot URLs
ftr6a=sparse.coo_matrix(trainNbrCnt).transpose() #Cnt of numbers
ftr7a=sparse.coo_matrix(trainMentionCnt).transpose() #Cnt of mentions
ftr8a=sparse.coo_matrix(trainHashCnt).transpose() #Cnt of hashtags
ftr9a=sparse.coo_matrix(trUpper).transpose() #Upper case letters
ftr10a=sparse.coo_matrix(trTweetlength).transpose() #
ftr11a=sparse.coo_matrix(trainExCount).transpose()
ftr12a=sparse.coo_matrix(trainQCount).transpose()
ftr13a=sparse.coo_matrix(trainDotCount).transpose()

# On test set
testPosLex=getStemmedCnts("../positive-words.txt",testTweets["Value"])
testNegLex=getStemmedCnts("../negative-words.txt",testTweets["Value"])
tsUpper=list(testTweets['Value'].str.findall(r'[A-Z]').str.len())
tsTweetlength=testTweets['Value'].apply(len)

ftr1b=sparse.coo_matrix(testPosLex).transpose() #cnt of +ve words
ftr2b=sparse.coo_matrix(testNegLex).transpose() #cnt of -ve words
ftr3b=sparse.coo_matrix(testPEmots).transpose() #cnt of +ve emoticons
ftr4b=sparse.coo_matrix(testNEmots).transpose() #cnt of -ve emoticons
ftr5b=sparse.coo_matrix(testUrlCnt).transpose() #Cnt ot URLs
ftr6b=sparse.coo_matrix(testNbrCnt).transpose() #Cnt of numbers
ftr7b=sparse.coo_matrix(testMentionCnt).transpose() #Cnt of mentions
ftr8b=sparse.coo_matrix(testHashCnt).transpose() #Cnt of hashtags
ftr9b=sparse.coo_matrix(tsUpper).transpose()
ftr10b=sparse.coo_matrix(tsTweetlength).transpose()
ftr11b=sparse.coo_matrix(testExCount).transpose()
ftr12b=sparse.coo_matrix(testQCount).transpose()
ftr13b=sparse.coo_matrix(testDotCount).transpose()

#del(richTrainVec,richTestVec)
richTrainVec=sparse.hstack((reducedTrainVec,ftr3a, ftr4a, ftr8a,ftr9a,ftr11a,ftr12a,ftr1a,ftr2a))
richTestVec=sparse.hstack((reducedTestVec,ftr3b, ftr4b, ftr8b,ftr9b,ftr11b,ftr12b,ftr1b,ftr2b))

#%%
#Testing various models
classWts={"positive":0.4,"neutral":0.5,"negative":2.1}
testModel1 = naive_bayes.MultinomialNB()
testModel2 = SVC(kernel="linear",C=1)
testModel3 = SVC(kernel="rbf",C=100,gamma=0.001)
testModel4 = LogisticRegression(class_weight="balanced",C=0.3)
testModel5 = xgb.XGBClassifier(n_estimators=400,max_depth=3,objective="multi:softmax",
                               num_class=3,learning_rate="0.1",eval_metric="merror")
testModel6 = LogisticRegression(class_weight="balanced", C=1,fit_intercept=True,
                                multi_class="ovr", solver="liblinear")
testModel7=LogisticRegression(class_weight=classWts)
testModel8 = LogisticRegression(class_weight="balanced")
testModel9 = SVC(kernel="linear",C=1,class_weight="balanced")
testModel10 = SVC(kernel="rbf",C=100,gamma=0.001,class_weight="balanced") # this one gave best results on Kaggle
testModel11=SVC(kernel="rbf",C=10,gamma=0.016,class_weight="balanced")
allModels=[testModel4,testModel8]

for model in allModels:
#    %time TrainAndTestCV2(model,reducedTrainVec,trainLab)
    TrainAndTest(model,richTrainVec,trainLab,richTestVec,testLab)
#    TrainAndTestCV2(model,richTrainVec,trainLab)

#classWts={"positive":1,"neutral":1,"negative":1}
clf=SVC(kernel="rbf",C=100,gamma=0.001,class_weight="balanced")
clf=xgb.XGBClassifier(gamma=1, learning_rate= 0.2, subsample= 0.8,reg_lambda= 0.1, reg_alpha= 0, max_depth = 16,num_class=3,eval_metric="merror",objective="multi:softmax",n_estimators=300)
TrainAndTestCV2(clf,richTrainVec,trainLab)
TrainAndTest(clf,richTrainVec,trainLab,richTestVec,testLab)
