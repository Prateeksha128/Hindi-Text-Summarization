import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx
from spacy.lang.hi.stop_words import STOP_WORDS
from heapq import nlargest
import os
from spacy.language import Language
from spacy.lang.hi import Hindi 
import re
from rouge import Rouge

# Set the appropriate NLTK data path for Hindi language
nltk.data.path.append("/path/to/nltk_data")

text = '''गुप्त साम्राज्य के दो महत्वपूर्ण राजा हुए, समुद्रगुप्त और दूसरे चंद्रगुप्त द्वितीय। गुप्त वंश के लोगों के द्वारा ही संस्कृत की एकता फिर एकजुट हुई। चंद्रगुप्त प्रथम ने 320 ईस्वी को गुप्त वंश की स्थापना की थी और यह वंश करीब 510 ई तक शासन में रहा। 463-473 ई में सभी गुप्त वंश के राजा थे, केवल नरसिंहगुप्त बालादित्य को छोड़कर। लादित्य ने बौद्ध धर्म अपना लिया था, शुरुआत के दौर में इनका शासन केवल मगध पर था, पर फिर धीरे-धीरे संपूर्ण उत्तर भारत को अपने अधीन कर लिया था। गुप्त वंश के सम्राटों में क्रमश : श्रीगुप्त, घटोत्कच, चंद्रगुप्त प्रथम, समुद्रगुप्त, रामगुप्त, चंद्रगुप्त द्वितीय, कुमारगुप्त प्रथम (महेंद्रादित्य) और स्कंदगुप्त हुए। देश में कोई भी  ऐसी शक्तिशाली केन्द्रीय शक्ति नहीं थी , जो अलग-अलग छोटे-बड़े राज्यों को विजित कर एकछत्र शासन-व्यवस्था की स्थापना कर पाती । यह जो काल था वह  किसी महान सेनानायक की महत्वाकांक्षाओं की पूर्ति के लिये सर्वाधिक सुधार का अवसर के बारे में बता रहा था। फलस्वरूप मगध के गुप्त राजवंश में ऐसे महान और बड़े सेनानायकों का विनाश हो रहा था ।सिंधु घाटी की सभ्यता के साथ भारतीय इतिहास का जन्म हुआ था। हड़प्पा सभ्यता इस समय की शुरुआत भी यहीं से मानी जाती है। सिंधु घाटी की सभ्यता दक्षिण एशिया के पश्चिमी हिस्से में लगभग 2500 बीसी में फैली हुई है। इंडियन हिस्ट्री इन हिंदी में सिंधु घाटी सभ्यता के बारे में महत्वपूर्ण जानकारी नीचे दी गई है:आज के समय में पाकिस्तान और पश्चिमी भारत के नाम से जाना जाता है।सिंधु घाटी की सभ्यता 4 हिस्सों में बांटी गई है:सिंधु घाटी, मिश्र मेसोपोटामिया ,भारत ,चीन इस क्षेत्र से यह पता चलता है कि लगभग 5000 साल पहले यहां अत्यंत उच्च विकसित सभ्यता फैली हुई थ। सिंधु घाटी की सभ्यता एक शहरी सभ्यता थी। इसके अंदर लोग सुयोजनाबद्ध और सुनिमित कस्बों  के अंदर रहते थे।।'''




stop_words = list(STOP_WORDS)   
sentences = sent_tokenize(text)
sentences = re.split('[.।?!\n]', text)
sentences_clean = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in sentences]
  # Use Hindi stopwords
sentence_tokens = [[word for word in sentence.split(' ') if word not in stop_words] for sentence in sentences_clean]

w2v = Word2Vec(sentence_tokens, vector_size=1, min_count=1, epochs=1000)
sentence_embeddings = [[w2v.wv[word][0] for word in words] for words in sentence_tokens]
max_len = max([len(tokens) for tokens in sentence_tokens])
sentence_embeddings = [np.pad(embedding, (0, max_len - len(embedding)), 'constant') for embedding in sentence_embeddings]

similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
for i, row_embedding in enumerate(sentence_embeddings):
    for j, column_embedding in enumerate(sentence_embeddings):
        similarity_matrix[i][j] = 1 - spatial.distance.cosine(row_embedding, column_embedding)

nx_graph = nx.from_numpy_array(similarity_matrix)
scores = nx.pagerank(nx_graph)

top_sentence = {sentence: scores[index] for index, sentence in enumerate(sentences)}
top = dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:6])

for sent in sentences:
    if sent in top.keys():
        print(sent)


def evaluate_summary(reference_summary, generated_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary)
    return scores

# Example usage
reference_summary = "गुप्त वंश के लोगों के द्वारा ही संस्कृत की एकता फिर एकजुट हुई  गुप्त वंश के सम्राटों में क्रमश : श्रीगुप्त, घटोत्कच, चंद्रगुप्त प्रथम, समुद्रगुप्त, रामगुप्त, चंद्रगुप्त द्वितीय, कुमारगुप्त प्रथम (महेंद्रादित्य) और स्कंदगुप्त हुए सिंधु घाटी की सभ्यता के साथ भारतीय इतिहास का जन्म हुआ था  सिंधु घाटी की सभ्यता दक्षिण एशिया के पश्चिमी हिस्से में लगभग 2500 बीसी में फैली हुई है सिंधु घाटी की सभ्यता 4 हिस्सों में बांटी गई है:सिंधु घाटी, मिश्र मेसोपोटामिया ,भारत ,चीन इस क्षेत्र से यह पता चलता है कि लगभग 5000 साल पहले यहां अत्यंत उच्च विकसित सभ्यता फैली हुई थ  सिंधु घाटी की सभ्यता एक शहरी सभ्यता थी"
generated_summary = " गुप्त वंश के लोगों के द्वारा ही संस्कृत की एकता फिर एकजुट हुई चंद्रगुप्त प्रथम ने 320 ईस्वी को गुप्त वंश की स्थापना की थी और यह वंश करीब 510 ई तक शासन में रहा 463-473 ई में सभी गुप्त वंश के राजा थे, केवल नरसिंहगुप्त बालादित्य को छोड़कर गुप्त वंश के सम्राटों में क्रमश : श्रीगुप्त, घटोत्कच, चंद्रगुप्त प्रथम, समुद्रगुप्त, रामगुप्त, चंद्रगुप्त द्वितीय, कुमारगुप्त प्रथम (महेंद्रादित्य) और स्कंदगुप्त हुएफलस्वरूप मगध के गुप्त राजवंश में ऐसे महान और बड़े सेनानायकों का विनाश हो रहा था सिंधु घाटी की सभ्यता दक्षिण एशिया के पश्चिमी हिस्से में लगभग 2500 बीसी में फैली हुई है'"

evaluation_scores = evaluate_summary(reference_summary, generated_summary)
print(evaluation_scores)