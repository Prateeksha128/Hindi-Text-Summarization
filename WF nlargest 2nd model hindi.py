import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import string
from spacy.lang.hi.stop_words import STOP_WORDS
from heapq import nlargest
import os
from spacy.language import Language
from spacy.lang.hi import Hindi 
import re
from rouge import Rouge

stopwords = list(STOP_WORDS)   
nlp=Hindi()
# test the vectors and similarity    


text = "गुप्त साम्राज्य के दो महत्वपूर्ण राजा हुए, समुद्रगुप्त और दूसरे चंद्रगुप्त द्वितीय। गुप्त वंश के लोगों के द्वारा ही संस्कृत की एकता फिर एकजुट हुई। चंद्रगुप्त प्रथम ने 320 ईस्वी को गुप्त वंश की स्थापना की थी और यह वंश करीब 510 ई तक शासन में रहा। 463-473 ई में सभी गुप्त वंश के राजा थे, केवल नरसिंहगुप्त बालादित्य को छोड़कर। लादित्य ने बौद्ध धर्म अपना लिया था, शुरुआत के दौर में इनका शासन केवल मगध पर था, पर फिर धीरे-धीरे संपूर्ण उत्तर भारत को अपने अधीन कर लिया था। गुप्त वंश के सम्राटों में क्रमश : श्रीगुप्त, घटोत्कच, चंद्रगुप्त प्रथम, समुद्रगुप्त, रामगुप्त, चंद्रगुप्त द्वितीय, कुमारगुप्त प्रथम (महेंद्रादित्य) और स्कंदगुप्त हुए। देश में कोई भी  ऐसी शक्तिशाली केन्द्रीय शक्ति नहीं थी , जो अलग-अलग छोटे-बड़े राज्यों को विजित कर एकछत्र शासन-व्यवस्था की स्थापना कर पाती । यह जो काल था वह  किसी महान सेनानायक की महत्वाकांक्षाओं की पूर्ति के लिये सर्वाधिक सुधार का अवसर के बारे में बता रहा था। फलस्वरूप मगध के गुप्त राजवंश में ऐसे महान और बड़े सेनानायकों का विनाश हो रहा था ।सिंधु घाटी की सभ्यता के साथ भारतीय इतिहास का जन्म हुआ था। हड़प्पा सभ्यता इस समय की शुरुआत भी यहीं से मानी जाती है। सिंधु घाटी की सभ्यता दक्षिण एशिया के पश्चिमी हिस्से में लगभग 2500 बीसी में फैली हुई है। इंडियन हिस्ट्री इन हिंदी में सिंधु घाटी सभ्यता के बारे में महत्वपूर्ण जानकारी नीचे दी गई है:आज के समय में पाकिस्तान और पश्चिमी भारत के नाम से जाना जाता है।सिंधु घाटी की सभ्यता 4 हिस्सों में बांटी गई है:सिंधु घाटी, मिश्र मेसोपोटामिया ,भारत ,चीन इस क्षेत्र से यह पता चलता है कि लगभग 5000 साल पहले यहां अत्यंत उच्च विकसित सभ्यता फैली हुई थ। सिंधु घाटी की सभ्यता एक शहरी सभ्यता थी। इसके अंदर लोग सुयोजनाबद्ध और सुनिमित कस्बों  के अंदर रहते थे।।"

doc = nlp(text)
punctuations = string.punctuation + '\n' + "।" + "॥"
word_frequencies = {}
for word in doc:
    if word.text not in stopwords:
        if word.text not in punctuations:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

max_frequency = max(word_frequencies.values())
for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/max_frequency)

pattern = r'[\।?!]+[\s\n]+'

sentence_tokens =re.split(pattern,text)
print(len(sentence_tokens))

sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word]
            else:
                sentence_scores[sent] += word_frequencies[word]

select_length = int(len(sentence_tokens)*0.3)
summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)

final_summary = [word for word in summary]
summary = ' '.join(final_summary)

print(text)
print(summary)
print(len(text))
print(len(summary))

def evaluate_summary(reference_summary, generated_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary)
    return scores

# Example usage
reference_summary = "गुप्त वंश के लोगों के द्वारा ही संस्कृत की एकता फिर एकजुट हुई  गुप्त वंश के सम्राटों में क्रमश : श्रीगुप्त, घटोत्कच, चंद्रगुप्त प्रथम, समुद्रगुप्त, रामगुप्त, चंद्रगुप्त द्वितीय, कुमारगुप्त प्रथम (महेंद्रादित्य) और स्कंदगुप्त हुए सिंधु घाटी की सभ्यता के साथ भारतीय इतिहास का जन्म हुआ था  सिंधु घाटी की सभ्यता दक्षिण एशिया के पश्चिमी हिस्से में लगभग 2500 बीसी में फैली हुई है सिंधु घाटी की सभ्यता 4 हिस्सों में बांटी गई है:सिंधु घाटी, मिश्र मेसोपोटामिया ,भारत ,चीन इस क्षेत्र से यह पता चलता है कि लगभग 5000 साल पहले यहां अत्यंत उच्च विकसित सभ्यता फैली हुई थ  सिंधु घाटी की सभ्यता एक शहरी सभ्यता थी"
generated_summary = "इंडियन हिस्ट्री इन हिंदी में सिंधु घाटी सभ्यता के बारे में महत्वपूर्ण जानकारी नीचे दी गई है:आज के समय में पाकिस्तान और पश्चिमी भारत के नाम से जाना जाता है।सिंधु घाटी की सभ्यता 4 हिस्सों में बांटी गई है:सिंधु घाटी, मिश्र मेसोपोटामिया ,भारत ,चीन इस क्षेत्र से यह पता चलता है कि लगभग 5000 साल पहले यहां अत्यंत उच्च विकसित सभ्यता फैली हुई थ लादित्य ने बौद्ध धर्म अपना लिया था, शुरुआत के दौर में इनका शासन केवल मगध पर था, पर फिर धीरे-धीरे संपूर्ण उत्तर भारत को अपने अधीन कर लिया था फलस्वरूप मगध के गुप्त राजवंश में ऐसे महान और बड़े सेनानायकों का विनाश हो रहा था ।सिंधु घाटी की सभ्यता के साथ भारतीय इतिहास का जन्म हुआ था देश में कोई भी  ऐसी शक्तिशाली केन्द्रीय शक्ति नहीं थी , जो अलग-अलग छोटे-बड़े राज्यों को विजित कर एकछत्र शासन-व्यवस्था की स्थापना कर पाती ."

evaluation_scores = evaluate_summary(reference_summary, generated_summary)
print(evaluation_scores)



#---------------------------------------OUTPUT SUMMARY--------------------------------------


# इंडियन हिस्ट्री इन हिंदी में सिंधु घाटी सभ्यता के बारे में महत्वपूर्ण जानकारी नीचे दी गई है:आज के समय में पाकिस्तान और पश्चिमी भारत के नाम से जाना जाता है।सिंधु घाटी की सभ्यता 4 हिस्सों में बांटी गई है:सिंधु घाटी, मिश्र मेसोपोटामिया ,भारत ,चीन इस क्षेत्र से यह पता चलता है कि लगभग 5000 साल पहले यहां अत्यंत उच्च विकसित सभ्यता फैली हुई थ लादित्य ने बौद्ध धर्म अपना लिया था, शुरुआत के दौर में इनका शासन केवल मगध पर था, पर फिर धीरे-धीरे संपूर्ण उत्तर भारत को अपने अधीन कर लिया था फलस्वरूप मगध के गुप्त राजवंश में ऐसे महान और बड़े सेनानायकों का विनाश हो रहा था ।सिंधु घाटी की सभ्यता के साथ भारतीय इतिहास का जन्म हुआ था देश में कोई भी  ऐसी शक्तिशाली केन्द्रीय शक्ति नहीं थी , जो अलग-अलग छोटे-बड़े राज्यों को विजित कर एकछत्र शासन-व्यवस्था की स्थापना कर पाती 