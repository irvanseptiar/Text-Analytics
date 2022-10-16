from NLP_Models import CleanText as ct
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.util import ngrams
import matplotlib.pyplot as plt   # for wordclouds & charts
from matplotlib.pyplot import figure

def graphicGroupbyDate(data):
    timeSeries = data.groupby(['date']).size().reset_index(name='counts')
    if len(timeSeries) == 1:
        data['datetime'] = pd.to_datetime(data['time']) 
        data["hourminutes"] = data['datetime'].dt.strftime('%H')
        timeSeries = data.groupby(['hourminutes']).size().reset_index(name='counts')
        timeSeries.columns = ['time', '# Total Tweet']
    else:
        timeSeries = data.groupby(['date']).size().reset_index(name='counts')
        timeSeries.columns = ['time', '# Total Tweet']
    ax = timeSeries.plot.bar(x='time', y='# Total Tweet', rot=0)
    return timeSeries
        

def statsDes(data, n):
    print('TOP {} Tweet by Counts of Retweet')
    top_n_tweet_retweet_count = data.nlargest(n, ['tweet_retweet_count'])[['date', 'time', 'username', 'user_followers_count', 'user_statuses_count' ,'tweet_full_text', 'tweet_retweet_count', 'tweet_favorite_count']]
    #top_n_tweet_retweet_count = top_n_tweet_retweet_count.drop(columns = ['Unnamed: 0'])
    print(top_n_tweet_retweet_count)
    print()
    print('---------------------------------------------------------------')
    
    print('TOP {} Tweet by Counts of Favorite')
    top_n_tweet_favorite_count = data.nlargest(n, ['tweet_favorite_count'])[['date', 'time', 'username', 'user_followers_count', 'user_statuses_count' ,'tweet_full_text', 'tweet_retweet_count', 'tweet_favorite_count']]
    #top_n_tweet_favorite_count = top_n_tweet_favorite_count.drop(columns = ['Unnamed: 0'])
    print(top_n_tweet_favorite_count)
    print()
    print('---------------------------------------------------------------')
    
    print('TOP {} User by Followers')
    top_n_user_followers_count = data.nlargest(n, ['user_followers_count'])[['date', 'time', 'username', 'user_followers_count', 'user_statuses_count' ,'tweet_full_text', 'tweet_retweet_count', 'tweet_favorite_count']]
    #top_n_user_followers_count = top_n_user_followers_count.drop(columns = ['Unnamed: 0'])
    print(top_n_user_followers_count)
    print()
    print('---------------------------------------------------------------')
    
    print('TOP {} User by Status')
    top_n_user_followers_count = data.nlargest(n, ['user_statuses_count'])[['date', 'time', 'username', 'user_followers_count', 'user_statuses_count' ,'tweet_full_text', 'tweet_retweet_count', 'tweet_favorite_count']]
    #top_n_user_followers_count = top_n_user_followers_count.drop(columns = ['Unnamed: 0'])
    print(top_n_user_followers_count)
    print()
    return top_n_tweet_retweet_count, top_n_tweet_favorite_count, top_n_user_followers_count, top_n_user_followers_count


def get_ngrams(text, n=2):
    text = str(text)
    n_grams = ngrams(text.split(), n)
    returnVal = []
    
    try:
        for grams in n_grams:
            returnVal.append('_'.join(grams))
    except(RuntimeError):
        pass
        
    return ' '.join(returnVal).strip()

def preprocessing(data):
    dataclean = ct.cleanningtext(data, both=False, onlyclean=True)
    dataclean['unigram'] = dataclean["cleaned_tweet_full_text"].apply(get_ngrams, n=1)
    dataclean['bigram'] = dataclean["cleaned_tweet_full_text"].apply(get_ngrams, n=2)
    dataclean['trigram'] = dataclean["cleaned_tweet_full_text"].apply(get_ngrams, n=3)
    return dataclean

def generatedBiGram(data):
    tweet_string_list = data['unigram'].tolist()
    tweet_string_list = ' '.join(tweet_string_list)
    wordcloud = WordCloud(width = 2000, height = 1334, random_state=1, background_color='black', colormap='Pastel1', max_words = 75, collocations=False, normalize_plurals=False).generate(tweet_string_list)
    return wordcloud 

def generatedBiGram(data):
    tweet_string_list = data['bigram'].tolist()
    tweet_string_list = ' '.join(tweet_string_list)
    wordcloud = WordCloud(width = 2000, height = 1334, random_state=1, background_color='black', colormap='Pastel1', max_words = 75, collocations=False, normalize_plurals=False).generate(tweet_string_list)
    return wordcloud 

def generatedTriGram(data):
    tweet_string_list = data['bigram'].tolist()
    tweet_string_list = ' '.join(tweet_string_list)
    wordcloud = WordCloud(width = 2000, height = 1334, random_state=1, background_color='black', colormap='Pastel1', max_words = 75, collocations=False, normalize_plurals=False).generate(tweet_string_list)
    return wordcloud 
 

def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(25, 17), dpi=80)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.box(False)
    plt.show()
    plt.close() 
   

def frequencyword(data):
    temp = preprocessing(data)
    temp = temp.cleaned_tweet_full_text.str.split(expand=True).stack().value_counts()
    temp = pd.DataFrame(temp)
    temp = data.reset_index()
    temp = temp.rename(columns ={'index':'Words', 0:'Count'})
    ax = temp.plot.barh(x='Count', y='word', rot=0)
    return temp

