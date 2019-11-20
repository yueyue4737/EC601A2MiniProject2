%pylab inline
import matplotlib.pyplot as plt
import string
import jieba
import codecs
import os
from numpy import *
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn import naive_bayes as bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
ham_dataPath = r""
spma_dataPath = r""
with open(ham_dataPath,encoding='utf-8') as f:
    ham_txt_list = f.readlines()
with open(spma_dataPath,encoding="utf-8") as f:
    spam_txt_list = f.readlines()
# add stop words
stopwords = codecs.open(r'E:/ml_program/stopwords.txt','r','UTF8').read().split('\n')
# filter the stopwords
ham_processed_texts = []
for txt in ham_txt_list:
    words = []
    seg_list = jieba.cut(txt)
    for seg in seg_list:
        if (seg.isalpha()) and seg!='\n' and seg not in stopwords:
            words.append(seg)
    sentence = " ".join(words)
    ham_processed_texts.append(sentence)
# spam
spam_processed_texts = []
for txt in spam_txt_list:
    words = []
    seg_list = jieba.cut(txt)
    for seg in seg_list:
        if (seg.isalpha()) and seg!='\n' and seg not in stopwords:
            words.append(seg)
    sentence = " ".join(words)
    spam_processed_texts.append(sentence)
 # word cloud
def showWordCloud(text):
    wc = WordCloud(
        background_color = "white",
        max_words = 200,
        font_path = "C:\\Users\\dell\\Downloads\\1252935776\\simhei.ttf",
        min_font_size = 15,
        max_font_size = 50,
        width = 400
    )
    wordcloud = wc.generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
# print(" ".join(ham_processed_texts))
# showWordCloud(" ".join(ham_processed_texts))
showWordCloud(" ".join(spam_processed_texts))
showWordCloud(" ".join(ham_processed_texts))
def transformTextToSparseMatrix(texts):
    vectorizer = CountVectorizer(binary=False)
    vectorizer.fit(texts) # word list
    vocabulary = vectorizer.vocabulary_ 
    vector = vectorizer.transform(texts) # transform into a vector
#     print(vector.toarray())
    result = pd.DataFrame(vector.toarray())

    keys = []
    values = []
    for key,value in vectorizer.vocabulary_.items():
        keys.append(key)
        values.append(value)
    df = pd.DataFrame(data={"key":keys, "value": values})
    colnames = df.sort_values("value")["key"].values
    result.columns = colnames
    return result
#     print(vocabulary)
import numpy as np
data = []
data.extend(ham_processed_texts)
data.extend(spam_processed_texts)
textMatrix = transformTextToSparseMatrix(data)
textMatrix.head()
features = pd.DataFrame(textMatrix.apply(sum,axis=0))
extractedfeatures = [features.index[i] for i in range(features.shape[0]) if features.iloc[i,0]>5]
textMatrix = textMatrix[extractedfeatures]
textMatrix = textMatrix[extractedfeatures]
labels = []
labels.extend(ones(5000))
labels.extend(zeros(5001))
# split into a train and a validation data set
train,test,trainlabel,testlabel = train_test_split(textMatrix,labels,test_size=0.1)

# bayes
clf = bayes.BernoulliNB(alpha=1,binarize=True)
model = clf.fit(train,trainlabel)
# SVM
model2 = LinearSVC()
model2.fit(train,trainlabel)

print(model.score(test,testlabel))
print(model2.score(test,testlabel))

0.922077922077922
0.987012987012987
import matplotlib.pyplot as plt
from pylab import *                                 #to support Chinese
mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['1000', '2000', '4000', '6000', '8000','9000']
x = range(len(names))
y = [0.8609, 0.900, 0.9133, 0.9242, 0.9355,0.9490]
plt.plot(x, y, marker='o', mec='r', mfc='w')
plt.legend()  # add the legend
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"the number of training sets") #y axis
plt.ylabel("accuracy rate") #y axis
plt.title("贝叶斯中文垃圾邮件分类准确率的变化") #titile
plt.savefig("nb_zh.png")
plt.show()
