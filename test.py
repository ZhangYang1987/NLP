import pandas as pd
import nltk
import pymysql as msq
import jieba
import re
import gensim

# conn=msq.connect(db='cpcia',password='Zy2225786!',charset='utf8')
# cur=conn.cursor()
# # # # sql_str='select content from cpcia_news'
# sql_str='select content from cpcia_news where content like "%石油%" limit 10000'
# # # # sql_str='select content from cpcia_news where content like "%农药%"'
# cur.execute(sql_str)
# result=cur.fetchall()
# s=str(result)
# with open('Raw.txt','w',encoding='utf-8') as tf:
#     tf.write(s)
# # print(result)
# cur.close()
# conn.close()

# jieba.load_userdict('C:\\Users\\zy\\PycharmProjects\\zyw\\test\\user_dict.txt')
# s = "我去云南旅游，不仅去了玉龙雪山，还去丽江古城，很喜欢丽江古城，I am learning SVM_model"
#精确模式,能解决歧义，把文本精确的分词(这里玉龙雪山完美分词，但是丽江古城并不能)
# cut = jieba.cut(s)
# 全模式,把文本分成尽可能多的词
# cut = jieba.cut(s,cut_all = True)
# 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
# cut = jieba.cut_for_search(s)
# print (','.join(cut))

# fdis=nltk.FreqDist(cut)
# print(fdis)
# print(fdis.N())
# for key in fdis:
#     print(key,fdis[key])
# print(len(fdis.keys()))
# print(len(set(fdis.keys())))
# 获取词性
# import jieba.posseg as psg
# words = psg.cut(s)
# for word in words:
#     print(word.word,word.flag)

# 获取出现频率Top n的词
# from collections import Counter
# words=list(cut)
# topns=Counter(words).most_common(20)
# for top in topns:
#     print(top[0],top[1])

# 关键词提取，TF-IDF,TextRank
# import jieba.analyse
# TF-IDF
# jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
# –sentence 为待提取的文本
# –topK 为返回几个 TF*IDF 权重最大的关键词，默认值为 20
# –withWeight 为是否一并返回关键词权重值，默认值为 False
# –allowPOS 仅包括指定词性的词，默认值为空，即不筛选
# extract_tags方法是通过计算tf*idf返回关键词权重，其中
# tf为sentence中的词频
# jieba.analyse.idf为D:\Python27\Lib\site-packages\jieba\analyse\idf.txt中记录的数据



# seg = jieba.analyse.extract_tags(s, topK = 30, withWeight = True)
# for tag, weight in seg:
#     print("%s %s" %(tag, weight))

# txtrank=jieba.analyse.textrank(s, topK=30, withWeight=True, allowPOS=())
# for tag, weight in txtrank:
#     print("%s %s" %(tag, weight))



Raw='Raw.txt'
Raw_sentences='Raw_sentences.txt'
sentences_seg='sentences_seg.txt'
SegWord ='SegWord.txt'
TextRead=[]
TextSeg=[]
# with open(Raw,encoding='utf-8') as raw:
#     for line in raw:
#         seperate_line=line.split(u'。')
#         for i in seperate_line:
#             seperate_i=i.split(u'，')
#             for j in seperate_i:
#                 with open(Raw_sentences,'a',encoding='utf-8') as rs:
#                     rs.write(j)
#                     rs.write('\n')

#
#使用 suggest_freq(segment, tune=True) 可调节单个词语的词频，使其能（或不能）被分出来
jieba.suggest_freq('中石油',tune=True)
# jieba.suggest_freq('中石化',tune=True)
jieba.suggest_freq('中海油',tune=True)
# jieba.analyse.set_stop_words('Stop_Words_Dict.txt')
stop_words={}
# with open('Stop_Words_Dict.txt','r',encoding='utf-8') as swd:
#     for word in swd:
#         stop_words[word.strip()]=word.strip()
#
# with open(Raw_sentences,encoding='utf-8') as rs:
#     for line in rs:
#         sentence=line.strip()
#         sentence2=re.sub("[0-9\s+\.\!\/_,$%^*()?;；:-【】+]+|[+——！，;:。？、~@#￥%……&*（）]+","",sentence)
#         wordlist=list(jieba.cut(sentence2))
#         outstr=''
#         for word in wordlist:
#             if word not in stop_words:
#                 outstr+=word
#                 outstr+=' '
#         with open(sentences_seg, 'a',encoding='utf-8') as ss:
#             ss.write(outstr+'\n')

            # ss.write(str(' '.join(list(jieba.cut(line)))))
# #
#


# TextSeg.append([' '.join(list(jieba.cut(TextRead[0])))])
# # # print(TextRead[0][0])
# with open(SegWord,'w') as segsave:
#     for i in range(len(TextSeg)):
#         segsave.write(TextSeg[i][0])

# sentence=[]
# with open(SegWord) as readseg:
#     for i in readseg:
#         # print(i)
#         sentence.append(i)

# sentences=[]
# with open(sentences_seg,encoding='utf-8') as ss:
#     for line in ss:
#         sentences.append(line.split())
# print(sentences[33])
# print(type(sentences[44]))
#
# #
# from gensim.models.word2vec import Word2Vec
# # # # # from gensim.test.utils import common_texts,get_tmpfile
# # # # path=get_tmpfile('word2vec.model')
# model=Word2Vec(sentences,min_count=5,size=100,workers=4)
# model.save("word2vec.model")
# model.wv.save_word2vec_format('mymodel.txt',binary = False)
# for i in model.wv.vocab.keys(): #vocab是dict
#     print(type(i))
#     print(i)


model=gensim.models.Word2Vec.load('word2vec.model')
result = model.most_similar(u'中海油')
for each in result:
    print (each[0] , each[1])

