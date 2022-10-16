import DeepSentimentAnalysis as dp
import OffensiveComment as oc
import FilterTextPredict as filteks
import CleanText as ct
import pandas as pd
from wordcloud import WordCloud
#import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pandas import ExcelWriter

tqdm.pandas()

def makedirectory(name):
    # define the name of the directory to be created
    path = './ProfillingTwitter/'+ name
    
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s" % path)
    
    return path
        
def save_xlsx(list_dfs, name):
    xls_path = './ProfillingTwitter/'+ name +'/'+ name +'.xlsx'
    listsheet = ['dataall', 'TopActivity', 'Sen_pos', 'Sen_neg', 'Sen_net', 'Offensive', 'Porn', 'Desc_Sentiment', 'Desc_Offensive', 'Desc_Porn', 'Clean_data','WordFreq']
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer, listsheet[n], index = False)
        writer.save()
        
def savefilter_xlsx(list_dfs, name, filterword):
    xls_path = './ProfillingTwitter/'+ name +'/'+ name + '_' + filterword +'.xlsx'
    listsheetfilter = ['datafilter','TopActivity', 'Sen_pos', 'Sen_neg', 'Sen_net', 'Offensive', 'Porn', 'Desc_Sentiment', 'Desc_Offensive', 'Desc_Porn', 'Clean_data','WordFreq']
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer,listsheetfilter[n], index = False)
        writer.save()

def textclassification(data):
    print('\n','Sentiment:')
    data['Sentiment'] = data['tweet'].progress_apply(lambda x : dp.predictsentiment(x)['sentiment'])
    print('\n','Sentiment Done!','\n')
    print('Categorization:')
    data['Categorization'] = data['tweet'].progress_apply(lambda x : filteks.predfiltertext(x)[1])
    print('\n','Categorization Done!','\n')
    print('Offensive:')
    data['Offensive'] = data['tweet'].progress_apply(lambda x : oc.predoffensive(x)[1])
    print('\n','Offensive Done!','\n')
    return data

def statdescriptive(data):
    totaldata = len(data)
    c_neg, c_pos, c_net = len(data[data['Sentiment'].str.contains('Negative')]), len(data[data['Sentiment'].str.contains('Positive')]), len(data[data['Sentiment'].str.contains('Netral')])
    p_neg, p_pos, p_net = c_neg/totaldata, c_pos/totaldata, c_net/totaldata
    c_offen, c_noffen = len(data[data['Offensive'].str.contains('Offensive')]), len(data[data['Offensive'].str.contains('Non Offensive')])
    p_offen, p_noffen = c_offen/totaldata, c_noffen/totaldata
    c_porn = len(data[data['Categorization'].str.contains('Porn')])
    c_noporn = len(data[data['Categorization'].str.contains('Advertaisment')]) + len(data[data['Categorization'].str.contains('Others')])
    p_porn, p_noporn = c_porn/totaldata, c_noporn/totaldata
    
    statsenti = {'Sentiment':['Positive', 'Negative', 'Netral'], 'Count' : [c_pos, c_neg, c_net], 'Persentage': [p_pos, p_neg, p_net]}
    statsenti = pd.DataFrame(statsenti)
    
    statoffen = {'Offensive': ['Offensive', 'Non Offensive'], 'Count' : [c_offen, c_noffen], 'Persentage': [p_offen, p_noffen]}
    statoffen = pd.DataFrame(statoffen)
    
    statporn = {'Porn Detection': ['Porn', 'Non Porn'], 'Count' : [c_porn, c_noporn], 'Persentage': [p_porn, p_noporn]}
    statporn = pd.DataFrame(statporn) 
    
    return {'Descirptive_Sentiment': statsenti, 'Descirptive_Offensive': statoffen, 'Descirptive_Porn': statporn}

def toptweet(data, n):
    print('Top ',n,'Tweet (Like, Retweet, Replies, Sentiment, Porn, Offensive)','\n')
    top_n_likes = data.nlargest(n, ['likes_count'])
    top_n_likes_date = top_n_likes.date.tolist()
    top_n_likes_time = top_n_likes.time.tolist()
    top_n_likes_tweet = top_n_likes.tweet.tolist()
    top_n_likes_score = top_n_likes.likes_count.tolist()
    
    top_n_rt = data.nlargest(n, ['retweets_count'])
    top_n_rt_date = top_n_rt.date.tolist()
    top_n_rt_time = top_n_rt.time.tolist()
    top_n_rt_tweet = top_n_rt.tweet.tolist()
    top_n_rt_score = top_n_rt.retweets_count.tolist()
    
    top_n_replies = data.nlargest(n, ['replies_count'])
    top_n_replies_date = top_n_replies.date.tolist()
    top_n_replies_time = top_n_replies.time.tolist()
    top_n_replies_tweet = top_n_replies.tweet.tolist()
    top_n_replies_score = top_n_replies.retweets_count.tolist()
    
    negative = data[data['Sentiment'].str.contains('Negative')]
    negative = negative[['date', 'time', 'username', 'tweet' ,'replies_count', 'retweets_count', 'likes_count']]
    negative = negative.reset_index(drop=True)
    negative = negative.iloc[0:n,:]
    
    positive = data[data['Sentiment'].str.contains('Positive')]
    positive = positive[['date', 'time', 'username', 'tweet' ,'replies_count', 'retweets_count', 'likes_count']]
    positive = positive.reset_index(drop=True)
    positive = positive.iloc[0:n,:]
    
    netral = data[data['Sentiment'].str.contains('Positive')]
    netral = netral[['date', 'time', 'username', 'tweet' ,'replies_count', 'retweets_count', 'likes_count']]
    netral = netral.reset_index(drop=True)
    netral = netral.iloc[0:n,:]
    
    offensive = data[data['Offensive'].str.contains('Offensive')]
    offensive = offensive[['date', 'time', 'username', 'tweet' ,'replies_count', 'retweets_count', 'likes_count']]
    offensive = offensive.reset_index(drop=True)
    offensive = offensive.iloc[0:n,:]
    
    porn = data[data['Categorization'].str.contains('Porn')]
    porn = porn[['date', 'time', 'username', 'tweet' ,'replies_count', 'retweets_count', 'likes_count']]
    porn = porn.reset_index(drop=True)
    porn = porn.iloc[0:n,:]

    
    hasil = {'Date_likes': top_n_likes_date,'Time_likes':top_n_likes_time  ,'Favorite Tweet': top_n_likes_tweet, 'Score Likes': top_n_likes_score, 
             'Date_RT': top_n_rt_date,'Time_RT':top_n_rt_time, 'Retweeted': top_n_rt_tweet, 'Score Retweet':top_n_rt_score,
             'Date_Replies': top_n_replies_date,'Time_Replies':top_n_replies_time, 'Replies' : top_n_replies_tweet, 'Score Replies': top_n_replies_score}
    hasil = pd.DataFrame(hasil)
    
    print('Top ',n,'Activity (Like, Retweet, Replies, Sentiment, Porn, Offensive) Done!','\n')
    return {'TopActivity': hasil, 'Sen_pos': positive, 'Sen_neg': negative, 'Sen_net': netral,\
            'Offensive': offensive, 'Porn': porn}

def wordcloud(data, path, name):
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    min_font_size = 10).generate(str(data)) 
    
    wordcloud.to_file(path+'/'+ name +'.png')
      
    # plot the WordCloud image                        
#    plt.figure(figsize = (8, 8), facecolor = None) 
#    plt.imshow(wordcloud) 
#    plt.axis("off") 
#    plt.tight_layout(pad = 0) 
#      
#    plt.show()

def frequencyword(data):
    data = data.cleaned_tweet.str.split(expand=True).stack().value_counts()
    data = pd.DataFrame(data)
    data = data.reset_index()
    data = data.rename(columns ={'index':'Words', 0:'Count'})
    return data


def resultprofiling(data, name, filterword ,both = False, onlyclean = True, sentiment = False, n = 5):
    path = makedirectory(name)
    
    #All
    cleandata = ct.cleanningtext(data, both = both, onlyclean = onlyclean)
    cleandata = pd.DataFrame(cleandata)
    wordfreq = frequencyword(cleandata)
    wordcloud(wordfreq, path, name)
    data = textclassification(data)
    topac = toptweet(data, n)
    _1, _2, _3, _4, _5, _6 = topac['TopActivity'], topac['Sen_pos'], topac['Sen_neg'],topac['Sen_net'],\
    topac['Offensive'],topac['Porn']
    deskriptive = statdescriptive(data)
    _a, _b, _c = deskriptive['Descirptive_Sentiment'], deskriptive['Descirptive_Offensive'], deskriptive['Descirptive_Porn']
    
    listdataframe = [data, _1, _2, _3, _4, _5, _6, _a, _b, _c ,cleandata, wordfreq]
    save_xlsx(listdataframe, name)    
    #Filter
    data['tweet_lower'] = data['tweet'].str.lower()
    datafilter = data[data['tweet_lower'].str.contains(filterword)]
    datafilter = datafilter.reset_index(drop=True)
    datafilter_ = pd.DataFrame(datafilter['tweet_lower'].str.replace(filterword,''))
    datafilter = datafilter.drop(columns = 'tweet_lower')
    datafilter__ = pd.concat([datafilter, datafilter_], axis = 1)
    cleandatafilter = pd.DataFrame(datafilter__['cleaned_tweet'])
    wordfreqfilter = frequencyword(cleandatafilter)
    wordcloud(wordfreqfilter, path, name = name+'filter')
    topacfilter = toptweet(datafilter__, n)
    _1f, _2f, _3f, _4f, _5f, _6f = topacfilter['TopActivity'], topacfilter['Sen_pos'], topacfilter['Sen_neg'],topacfilter['Sen_net'],\
    topacfilter['Offensive'],topacfilter['Porn']
    deskriptivefilter = statdescriptive(datafilter__)
    _af, _bf, _cf = deskriptivefilter['Descirptive_Sentiment'], deskriptivefilter['Descirptive_Offensive'], deskriptivefilter['Descirptive_Porn']
    listdataframe = [datafilter__, _1f, _2f, _3f, _4f, _5f, _6f, _af, _bf, _cf ,cleandatafilter, wordfreqfilter]
    savefilter_xlsx(listdataframe, name, filterword)

    return {'Data': data, 'TopTweet': topac, 'CleanText': cleandata, 'WordFreq': wordfreq,\
            'DataFilter': datafilter__, 'TopTweetFilter': topacfilter, 'CleanTextFilter': cleandatafilter,\
            'WordFreqFilter': wordfreqfilter}