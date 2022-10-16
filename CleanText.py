from NLP_Models import TextMining as tm
import time
from NLP_Models import openewfile as of
from tqdm import tqdm
#import swifter


def cleanningtext(data, both = True, onlyclean = False, sentiment = False):
    print('Cleaning Text')
    fSlang = of.openfile(path = './NLP_Models/slangword')
    bahasa = 'id'
    stops, lemmatizer = tm.LoadStopWords(bahasa, sentiment = sentiment)
    sw=open(fSlang,encoding='utf-8', errors ='ignore', mode='r');SlangS=sw.readlines();sw.close()
    SlangS = {slang.strip().split(':')[0]:slang.strip().split(':')[1] for slang in SlangS}
  
    start_time = time.time()
    tqdm.pandas()
    
    if both:
        data['tweet_full_text'] = data['tweet_full_text'].astype('str')
        data['tweet_full_text'] = data['tweet_full_text'].str.lower()
        data = data[~data.tweet_full_text.str.contains('unavailable')]
        data['cleaned_tweet_full_text'] = data['tweet_full_text'].progress_apply(lambda x : tm.cleanText(x,fix=SlangS, pattern2 = True, lang = bahasa, lemma=lemmatizer, stops = stops, symbols_remove = True, numbers_remove = True, hashtag_remove=False, min_charLen = 2))
        data['cleaned_tweet_full_text'] = data['cleaned_tweet_full_text'].progress_apply(lambda x : tm.handlingnegation(x))
    elif onlyclean: 
        data['cleaned_tweet_full_text'] = data['tweet_full_text'].progress_apply(lambda x : tm.cleanText(x, fix=SlangS, pattern2 = True, lang = bahasa, lemma=lemmatizer, stops = stops, symbols_remove = True, numbers_remove = True, hashtag_remove=False, min_charLen = 3))
    else:
        data['cleaned_tweet_full_text'] = data['tweet_full_text'].progress_apply(lambda x : tm.handlingnegation(x))
    
    data = data[data['tweet_full_text'].notna()]
    print("%s seconds" %(time.time()-start_time))
    
    return data