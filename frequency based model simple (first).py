import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import re

with open('C:/Users/user/OneDrive/Documents/1.txt', 'r', encoding='utf-8') as tex:
    sen = tex.read()
print(sen)

   
with open('C:/Users/user/Downloads/stopwords.txt', 'r', encoding='utf-8') as stop:
    stopwords = stop.read()

def createfrequencytable(text_string) -> dict:
   stopWords = set(stopwords)
   words = word_tokenize(text_string)
   ps = PorterStemmer()
   
   freqTable = dict()
   for word in words:
      word=str(word)
      word = ps.stem(word)
      if word in stopWords:
         continue
      if word in freqTable:
         freqTable[word] += 1
      else:
         freqTable[word] = 1
   return freqTable
ft=createfrequencytable(sen)
print(ft)

#tokenization of sentences

sentences = sent_tokenize(sen) # NLTK function
sentences = re.split('[.।?!\n]', sen)
total_documents = len(sentences)
print(sentences)
print(total_documents)

def scoresentences(sentences, freqTable) -> dict:
    sentenceValue = dict()
    
    for sentence in sentences:
            word_count_in_sentence = (len(word_tokenize(sentence)))
            for wordValue in ft:
                if wordValue in sentence.lower():
                    if sentence[:10] in sentenceValue:
                        sentenceValue[sentence[:10]] += ft[wordValue]
                    else:
                        sentenceValue[sentence[:10]] = ft[wordValue]
                       
    
    return sentenceValue
sentence_val=scoresentences(sentences, ft)
print(sentence_val)


def findaverage_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
# Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))
    return average
thresh=findaverage_score(sentence_val)

def _generate_summary(sentences, sentenceValue, thresh):
    sentence_count = 0
    summary = ''
    for sentence in sentences:
            if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (thresh):
                summary += " " + sentence
                sentence_count += 1
    return summary
summary = _generate_summary(sentences, sentence_val, 1.1* thresh)
print(summary)
print(len(sen))
print(len(summary))





#---------------------------------------OUTPUT SUMMARY--------------------------------------


  # गुप्त वंश के लोगों के द्वारा ही संस्कृत की एकता फिर एकजुट हुई  गुप्त वंश के सम्राटों में क्रमश : श्रीगुप्त, घटोत्कच, चंद्रगुप्त प्रथम, समुद्रगुप्त, रामगुप्त, चंद्रगुप्त द्वितीय, कुमारगुप्त प्रथम (महेंद्रादित्य) और स्कंदगुप्त हुए सिंधु घाटी की सभ्यता के साथ भारतीय इतिहास का जन्म हुआ था  सिंधु घाटी की सभ्यता दक्षिण एशिया के पश्चिमी हिस्से में लगभग 2500 बीसी में फैली हुई है सिंधु घाटी की सभ्यता 4 हिस्सों में बांटी गई है:सिंधु घाटी, मिश्र मेसोपोटामिया ,भारत ,चीन इस क्षेत्र से यह पता चलता है कि लगभग 5000 साल पहले यहां अत्यंत उच्च विकसित सभ्यता फैली हुई थ  सिंधु घाटी की सभ्यता एक शहरी सभ्यता थी