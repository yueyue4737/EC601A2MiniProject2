## I had a class use naive bayes to finish the spam detection
katy=read.csv("",stringsAsFactors = FALSE)
lmfao=read.csv("",stringsAsFactors = FALSE)
eminem=read.csv("",stringsAsFactors = FALSE)
shakira=read.csv("",stringsAsFactors = FALSE)
train=rbind(katy,lmfao,eminem,shakira)

#cleaning data
library(qdapRegex)
library(stringdist)
for(i in 1:1606){
  train[i,4]=rm_url(train[i,4]) #removing urls
  train[i,4]<- gsub("[^0-9A-Za-z///' ]"," " , train[i,4] ,ignore.case = TRUE)#replace all the unwanted characters by white space
  train[i,4]=gsub("^[[:space:]]*","",train[i,4]) ## Remove leading whitespaces
  train[i,4]=gsub("[[:space:]]*$","",train[i,4]) ## Remove trailing whitespaces
  train[i,4]=gsub(' +',' ',train[i,4]) #remove unecessary white spaces thus removing unwanted characters
  #did not remove characters directly because if done so will maybe join some words for eg hello*world will become helloworld, hence did not do so
  }
  
write.csv(train,file="")

#num decides how much numbers to compare
#change trainnum and testnum to change how much you want to train or test
testnum=30

trainnum=1606
#declaring the confusion matrix elements
#1=predicted ham and actual ham TN
#2=predicted spam and acutal ham FP
#3=predicted ham and actual spam FN
#4=predicted spam and actual spam TP
jaccard1=0
jaccard2=0
jaccard3=0
jaccard4=0
cosine1=0
cosine2=0
cosine3=0
cosine4=0


#testing loop
for(i in 1225:1606){#make this 2 to 1606 later on
  if(train[i,4]==""){next}#removes comaparing empty elements
  test=train[i,4]
  correct=train[i,5]
  spamtempcosine=0
  hamtempcosine=0
  spamtempjaccard=0
  hamtempjaccard=0
  #first cosine then jaccard
  for(j in 1:1284){#training loop
    if(j==i){next}#removes comaparing the same element
    if(train[j,4]==""){next}#removes comaparing empty elements
    if(train[j,5]==1)
    {
      spamtempcosine=spamtempcosine+stringdist(test,train[j,4], method ="cosine", useBytes = FALSE, q = 1, nthread = getOption("sd_num_thread"))
      spamtempjaccard=spamtempjaccard+stringdist(test,train[j,4], method ="jaccard", useBytes = FALSE, q = 1, nthread = getOption("sd_num_thread"))
    }
    else
    {
      hamtempcosine=hamtempcosine+stringdist(test,train[j,4], method ="cosine", useBytes = FALSE, q = 1, nthread = getOption("sd_num_thread"))
      hamtempjaccard=hamtempjaccard+stringdist(test,train[j,4], method ="jaccard", useBytes = FALSE, q = 1, nthread = getOption("sd_num_thread"))
      
    }
    #checking if predictions are correct spam
    if(correct==1){
      if(spamtempcosine<hamtempcosine){cosine4=cosine4+1}
      else{cosine3=cosine3+1}
      if(spamtempjaccard<hamtempjaccard){jaccard4=jaccard4+1}
      else{jaccard3=jaccard3+1}
    }
    else{
      if(spamtempcosine<hamtempcosine){cosine2=cosine2+1}
      else{cosine1=cosine1+1}
      if(spamtempjaccard<hamtempjaccard){jaccard2=jaccard2+1}
      else{jaccard1=jaccard1+1}
      
    }
    
  }}

c=c(cosine4,cosine2,cosine3,cosine1)
dim(c)=c(2,2)
colnames(c)=c("Predicted Spam","Predicted Ham")
rownames(c)=c("Actual Spam","Actual Ham")
j=c(jaccard4,jaccard2,jaccard3,jaccard1)
dim(j)=c(2,2)
colnames(j)=c("Predicted Spam","Predicted Ham")
rownames(j)=c("Actual Spam","Actual Ham")
print("Confusion Matrix of Cosine Similarity")
print(c)
print("Confusion Matrix of Jaccard Similarity")
print(j)

#printing confusion matrix and accuracy
cosineaccuracy=(cosine4+cosine1)/(cosine1+cosine2+cosine3+cosine4)
cosineprecision=cosine4/(cosine2+cosine4)
cosinerecall=cosine4/(cosine4+cosine3)
cosinef1=(2*cosinerecall*cosineprecision)/(cosinerecall+cosineprecision)
jaccardaccuracy=(jaccard4+jaccard1)/(jaccard1+jaccard2+jaccard3+jaccard4)
jaccardprecision=jaccard4/(jaccard2+jaccard4)
jaccardrecall=jaccard4/(jaccard4+jaccard3)
jaccardf1=(2*jaccardrecall*jaccardprecision)/(jaccardrecall+jaccardprecision)
cat("Cosine")
cat("Accuracy: ",cosineaccuracy)
cat("Precision: ",cosineprecision)
cat("Recall: ",cosinerecall)
cat("F1 Score: ",cosinef1)
cat("Jaccard")
cat("Accuracy: ",jaccardaccuracy)
cat("Precision: ",jaccardprecision)
cat("Recall: ",jaccardrecall)
cat("F1 Score: ",jaccardf1)
