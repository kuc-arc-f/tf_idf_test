# encoding: utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from janome.tokenizer import Tokenizer

#
def get_token(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    word = ""
    for token in tokens:
        part_of_speech = token.part_of_speech.split(",")[0]
        if part_of_speech == "名詞":
            word +=token.surface + " "
        if part_of_speech == "動詞":
            word +=token.base_form+ " "
        if part_of_speech == "形容詞":
            word +=token.base_form+ " "
        if part_of_speech == "形容動詞":
            word +=token.base_form+ " "
    return word


words1="利用人数は何人ですか？"
words2="契約期間は、ありますか？"
words3="オープンソースですか？"
words4="オンライン決済は、可能ですか?"
words5="製品価格、値段はいくらですか？"

#words= get_token(words1 )
#print(words )
#quit()
words =[]
words.append(words1 )
words.append(words2 )
words.append(words3 )
words.append(words4 )
words.append(words5 )

#print(words )
tokens=[]
for item in words:
    token=get_token(item)
    tokens.append(token)
#
#print(tokens )
docs = np.array(tokens)

vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
print(tokens)
#quit()
vecs = vectorizer.fit_transform(docs )
str="利用人数は？"
#str="契約期間"
#str="価格は？"

instr = get_token(str ).strip()
print("instr=", instr )
x= vectorizer.transform( [  instr ])
#print( "x=",x)
num_sim=cosine_similarity(x , vecs)
print(num_sim )
index = np.argmax( num_sim )

print("word=", words[index])
print()
    