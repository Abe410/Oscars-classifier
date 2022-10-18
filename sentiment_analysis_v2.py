# Imports
#%env ENV=DEV
import datetime
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

from sentiment_preprocessing_v2 import (preprocess_reddit, preprocess_telegram,
                                        preprocess_twitter)

# Connecting to Snowflake

sys.path.append(os.path.join(os.path.expanduser("~"), "tm-data-tools/data/"))
import flake

token_path = os.path.join(os.path.expanduser(
    "~"), "tokenmetrics-ml/sentiment_analysis/tools/token_list.csv")

def data_query (source, date):

    '''
    Description:
    Query data from snowflake

    -------------

    Arguments:
    Source = The source of data, i.e. Reddit, Telegram, or Twitter

    -------------
    
    Returns:
    scrape_df = DataFrame of scraped data
    '''
    
    if source == 'twitter':
        query = f'''select * from RAW_TWEETS where DATE(DATE) = '{date}' order by DATE ASC;'''
    elif source == 'telegram':
        query = f'''select * from RAW_TELEGRAM where DATE(DATE) = '{date}' order by DATE ASC;'''
    elif source == 'reddit':
        query = f'''select * from RAW_REDDIT where DATE(DATE) = '{date}' order by DATE ASC;'''
    else:
        print('Source not valid')
        return None

    print ("Fetching data from database...")
    
    scrape_df = flake.read_snowflake(query, database = 'TM_REFERENCE_DB', schema = 'SENTIMENT')

    return scrape_df

def data_cleaning(df, source):

    '''
    Description:
    Clean data 
    -------------

    Arguments:
    df = The scraped dataframe
    Source = The source of data, i.e. Reddit, Telegram, or Twitter

    -------------
    
    Returns:
    cleaned_df = DataFrame of cleaned data
    '''
    
    
    if source == 'twitter':
        df =  preprocess_twitter(df)
        return df
    elif source == 'telegram':
        return preprocess_telegram(df)
    elif source == 'reddit':
         return preprocess_reddit(df)
    else:
        print('Not a good data source')
        return None

def get_coin_dataframe (df, coin_id):

    
    '''
    Description:
    Get data only of specific coin

    -------------

    Arguments:
    df = The cleaned data frame
    coin_id = ID of coin of interest

    -------------
    
    Returns:
    DF containing only single coin data
    '''
    
    return df.loc[df['token_id']== coin_id].reset_index(drop=True)

def initialize_bert_model (NUM_LABELS):

    '''
    Description:
    Initialize BERT model from checkpoint

    -------------

    Arguments:
    NUM_LABELS = The desired output class labels, 2 for 2 class labels, 3 for 3 class labels including neutral

    -------------
    
    Returns:
    Classifier object
    '''

    # Initializing BERT model

    if NUM_LABELS == 2:
        model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    elif NUM_LABELS == 3:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest" 

    # Download and cache the tokenizer and classification model

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Pre trained pipeline

    classifier = pipeline("sentiment-analysis", model = model, tokenizer = tokenizer)
    return classifier

def get_sentiment (inputs, bert_model, NUM_LABELS):

    '''
    Description:
    Get the sentiment labels for data

    -------------

    Arguments:
    inputs = List of raw text inputs
    bert_model = Instance of BERT model
    NUM_LABELS = The desired output class labels, 2 for 2 class labels, 3 for 3 class labels including neutral

    -------------
    
    Returns:
    predicted_sentiments = Labels of text
    '''
    
    # Getting labels
    
    print('Now passing data through model...')

    sentiment_dict = bert_model(inputs, truncation = True)
    print('Successfully passed data through model!')

    predicted_sentiments = []

    if NUM_LABELS == 2:
        for i in range(len(sentiment_dict)):
            predicted_sentiments.append(sentiment_dict[i]['label'])
        print('Finished getting 2 class labels.')

    elif NUM_LABELS == 3:
        for i in range(len(sentiment_dict)):
            predicted_sentiments.append(1) if sentiment_dict[i]['label'] == 'Positive' else \
            (predicted_sentiments.append(-1) if sentiment_dict[i]['label'] == 'Negative' else predicted_sentiments.append(0)) 
        print('Finished getting 3 class labels.')
    
    return predicted_sentiments

def sentiment_labels(data, NUM_LABELS, bert_classifier):

    '''
    Description:
    Function to append labels to df of coin of interest. This function gets the actual sentiments of the text from the model

    -------------

    Arguments:
    data = The data frame to be passed, this contains all the scraped data from Twitter
    NUM_LABELS = The desired output class labels, 2 for 2 class labels, 3 for 3 class labels including neutral
    bert_classifier = Classifier object for BERT model
    -------------
    
    Returns:
    sentiments_df = Df appended with sentiment labels
    '''

    input_text = list(data['text'])

    sentiments = get_sentiment(input_text, bert_classifier, NUM_LABELS)

    data['BERT_labels'] = sentiments
    
    return data

def labelled_df (df, coin_id, NUM_LABELS, source, bert_classifier):

    '''
    Description:
    Get labeled DF of a specific coin 

    -------------

    Arguments:
    df (pd.DataFrame) = Cleaned and scraped data frame according to source
    coin_id (int) = Token ID for coin of interest
    NUM_LABELS (int) = The desired output class labels, 2 for 2 class labels, 3 for 3 class labels including neutral
    source (str) = Source for which sentiment is needed
    bert_classifier = BERT classifier object

    -------------
    
    Returns:
    Pandas DF with sentiment labels and text for only the coin of interest
    '''
        
    coin_df = get_coin_dataframe (df = df, coin_id = coin_id)

    if isinstance(coin_df, pd.DataFrame) and len(coin_df) != 0:

        coin_df_labelled = sentiment_labels(data = coin_df, NUM_LABELS = NUM_LABELS, bert_classifier = bert_classifier)

        return coin_df_labelled            
    else:
        print (f"There is no clean data in the last 24 hours from {source} for coin ID {coin_id}.")
        return None

def get_weighted_twitter_sentiment(df, NUM_LABELS):

    '''
    Description:
    Function to get weighted sentiment scores of twitter data 

    -------------

    Arguments:
    df = DF with the BERT labels and twitter data
    NUM_LABELS (int) = No. of required labels from the classification

    -------------
    
    Returns:
    Dict of positive and negative percentage
    '''
    if df is not None:
        length = len(df)
        print (f'Total number of twitter samples = {length}' )
       
        scaler = MinMaxScaler()
        df.loc[:, ['followers', 'likes', 'comments', 'retweets']] = scaler.fit_transform(df[['followers', 'likes', 'comments', 'retweets']])
       
        df['weight'] = df.followers * .5 + df.retweets * .3 + df.likes * .1 + df.comments * .1 + 1

        if NUM_LABELS == 2:
            df['BERT_labels'] = df['BERT_labels'].apply(lambda label: 1 if label == 'POSITIVE' else -1)
        
        df['weighted_label'] = list(map(lambda weight, label: weight * label, df['weight'], df['BERT_labels']))

        pos = 0
        neg = 0
        score_weights = [0,100]  #For weighted avg scoring scheme

        for row in df.itertuples(index = False):
            if row.BERT_labels == 1:
                pos += row.weighted_label
            elif row.BERT_labels == -1:
                neg += row.weighted_label

        total = np.sum(df['weight'])

        # try:
        #     # positive_ratio = pos / total
        #     # negative_ratio = abs(neg) / total
            
        # except ZeroDivisionError:
        #     print ('There is no data for this coin')
        #     #sentiment_ratio = {'Positive Sentiment': np.nan, 'Negative Sentiment': np.nan}
            
        # else:
        #     #sentiment_ratio = {'Positive Sentiment': positive_ratio, 'Negative Sentiment': negative_ratio}
        #     print('Finished\n')
        
        # finally:
        #     return sentiment_ratio


        #New start
        label_counts = [abs(neg), pos]
        try:
            sentiment_ratio = sum(list(map(lambda s,l: s*l, label_counts, score_weights)))/sum(label_counts)
        except ZeroDivisionError:
            sentiment_ratio = np.nan
        print('Finished\n')
        return sentiment_ratio
        #New end

    else:
        #sentiment_ratio = {'Positive Sentiment': np.nan, 'Negative Sentiment': np.nan}
        sentiment_ratio = np.nan
        return sentiment_ratio

def get_weighted_telegram_sentiment(df, NUM_LABELS):

    '''
    Description:
    Function to get sentiment scores of telegram data 

    -------------

    Arguments:
    df = DF with the BERT labels and telegram data
    NUM_LABELS (int) = No. of required labels from the classification

    -------------
    
    Returns:
    Dict of positive and negative percentage
    '''
    
    if df is not None:
        score_weights = [0,100]  #For weighted avg scoring scheme
        total = len(df['BERT_labels'])
        print (f'Total number of telegram samples = {total}' )
        
        if NUM_LABELS == 2:
            pos_count = Counter(df['BERT_labels'])['POSITIVE']
            neg_count = Counter(df['BERT_labels'])['NEGATIVE']
            
            
        elif NUM_LABELS == 3:
            pos_count = Counter(df['BERT_labels'])[1]
            neg_count = Counter(df['BERT_labels'])[-1]
        
        label_counts = [neg_count, pos_count]
        #sentiment_ratio = {'Positive Sentiment': pos_count/total, 'Negative Sentiment': neg_count/total}
        #New start
        try:
            sentiment_ratio = sum(list(map(lambda s,l: s*l, label_counts, score_weights)))/sum(label_counts)
        except ZeroDivisionError:
            sentiment_ratio = np.nan
        #New end
        print('Finished\n')
        return sentiment_ratio

    else:
        #sentiment_ratio = {'Positive Sentiment': np.nan, 'Negative Sentiment': np.nan}
        sentiment_ratio = np.nan
        return sentiment_ratio

def get_weighted_reddit_sentiment(df, NUM_LABELS):

    '''
    Description:
    Function to get sentiment scores of reddit data 

    -------------

    Arguments:
    df = Pandas DF with the BERT labels and reddit data
    NUM_LABELS (int) = No. of required labels from the classification

    -------------
    
    Returns:
    Dict of positive and negative percentage
    '''

    if df is not None:
        length = len(df)
        print (f'Total number of reddit samples = {length}')
        
        scaler = MinMaxScaler()
        df.loc[:, ['num_comments', 'score']] = scaler.fit_transform(df[['num_comments', 'score']])
        
        df['weight'] = df.score * .6 + df.num_comments * .4 + 1
        
        if NUM_LABELS == 2:
            df['BERT_labels'] = df['BERT_labels'].apply(lambda label: 1 if label == 'POSITIVE' else -1)
        
        df['weighted_label'] = list(map(lambda weight, label: weight * label, df['weight'], df['BERT_labels']))
        
        pos = 0
        neg = 0
        score_weights = [0, 100]  #For weighted avg scoring scheme

        for row in df.itertuples(index = False):
            if row.BERT_labels == 1:
                pos += row.weighted_label
            elif row.BERT_labels == -1:
                neg += row.weighted_label

        total = np.sum(df['weight'])

        # try:
        #     #positive_ratio = pos / total
        #     #negative_ratio = abs(neg) / total
        
        # except ZeroDivisionError:
        #     print ('There is no data for this coin')
        #     #sentiment_ratio = {'Positive Sentiment': np.nan, 'Negative Sentiment': np.nan}
    
        # else:
        #     #sentiment_ratio = {'Positive Sentiment': positive_ratio, 'Negative Sentiment': negative_ratio}
        #     print('Finished\n')
        
        # finally:
        #     return sentiment_ratio

        #New start
        label_counts = [abs(neg), pos]
        try:
            sentiment_ratio = sum(list(map(lambda s,l: s*l, label_counts, score_weights)))/sum(label_counts)
        except ZeroDivisionError:
            sentiment_ratio = np.nan
        print('Finished\n')
        return sentiment_ratio
        #New end

    else:
        #sentiment_ratio = {'Positive Sentiment': np.nan, 'Negative Sentiment': np.nan}
        sentiment_ratio = np.nan
        return sentiment_ratio

def query_and_clean(date, sources = ['twitter', 'reddit', 'telegram']):
    '''
    Description:
    Get cleaned scraped data from Snowflake 

    -------------

    Arguments:
    sources (str) = Default is for three data sources

    -------------
    
    Returns:
    3 Pandas DFs of cleaned data from all 3 sources
    '''

    for source in sources:
        if source == 'twitter':
            print ('TWITTER\n-----------')
            twitter_scrape_df = data_query(source, date)
            if len(twitter_scrape_df != 0):
                twitter_scrape_df = data_cleaning(twitter_scrape_df, source)
            else:
                print('No data scraped from Twitter')
        elif source == 'reddit':
            print ('REDDIT\n-----------')
            reddit_scrape_df = data_query(source, date)
            if len(reddit_scrape_df != 0):
                reddit_scrape_df = data_cleaning(reddit_scrape_df, source)
            else:
                print('No date scraped from Reddit')
        elif source == 'telegram':
            print ('TELEGRAM\n-----------')
            telegram_scrape_df = data_query(source, date)
            if len(telegram_scrape_df != 0):
                telegram_scrape_df = data_cleaning(telegram_scrape_df, source)
            else:
                print('No date scraped from Telegram')
        else:
            print('Not a valid source')
    
    return twitter_scrape_df, reddit_scrape_df, telegram_scrape_df

def get_market_cap(df_sent, dict_cap):

    '''
    Description:
    Function to get market cap of each coin from database 

    -------------

    Arguments:
    df_sent (pd.DataFrame) = The data frame for the latest sentiments
    dict_cap = Dictionary of total market cap of each coin in the form of a key:value pair
    -------------
    
    Returns:
    List of market cap for each coin
    '''
    market_caps = []
    for idx, token in enumerate(df_sent['TOKEN_ID']):
        total = dict_cap[token]#*df_sent['OVERALL'].values[idx]
        market_caps.append(total)
    return market_caps

def nanaverage(values, weights):
    """
    Allow weighted average with Nan values
    Parameters
    ----------
    A : values to average
    weights : respective weights

    """
    values = np.array(values)
    weights = np.array(weights)

    if np.all(np.isnan(values)):
        return np.nan
    else:
        return np.nansum(values * weights) / ((~np.isnan(values)) * weights).sum()

if __name__ == "__main__":

    NUM_LABELS = 3
    sources =  ['twitter', 'reddit', 'telegram']

    # Getting names of tokens

    token_list = pd.read_csv(token_path)
    token_names = token_list['name'].values
    token_symbols = token_list['symbol'].values
    token_ids = token_list['token_id'].values

    tokens_dict = dict(zip(token_ids, token_names))


    # Checking env date variable and getting date

    ''' Note: The date to be passed to function should be yesterday's i.e. one day before'''

    if os.getenv('PROCESS_DATE') is None:
        date = (datetime.date.today() - datetime.timedelta(days= 1))#.strftime("%Y-%m-%d")
    else:
        date = datetime.datetime.strptime(os.getenv('PROCESS_DATE'), "%Y-%m-%d").date()

    # Initializing aggregated DF

    aggregated_sentiment = pd.DataFrame(columns = ['Date', 'token_id', 'Name', 'Twitter', 'Reddit', 'Telegram', 'Overall'])

    # Initialize the BERT model

    classifier = initialize_bert_model (NUM_LABELS)

    # Loop to iterate through different data sources and coins, and append final sentiments to aggregated sentiment DF

    twitter_df, reddit_df, telegram_df = query_and_clean(date)

    token_id = list(set(np.concatenate((twitter_df['token_id'].unique(),\
                                        reddit_df['token_id'].unique(),\
                                        telegram_df['token_id'].unique()))))

    for token in tqdm(token_id):

        for source in sources:
            print('-----------------------')
            print(f'Getting data and sentiment for token ID {token} from {source}') 

            if source == 'twitter':
                final_df_twitter = labelled_df(df = twitter_df,
                                                coin_id = token,
                                                NUM_LABELS = NUM_LABELS,
                                                source = source,
                                                bert_classifier = classifier
                                                )
                twitter_sentiment_scores = get_weighted_twitter_sentiment(final_df_twitter, NUM_LABELS = NUM_LABELS)
                #twitter_sentiment_scores = np.round(twitter_sentiment_scores['Positive Sentiment']*100,2)
                twitter_sentiment_scores = np.round(twitter_sentiment_scores,2)


            elif source == 'reddit':
                final_df_reddit = labelled_df(df = reddit_df, 
                                            coin_id = token, 
                                            NUM_LABELS = NUM_LABELS, 
                                            source = source, 
                                            bert_classifier =  classifier
                                            )
                reddit_sentiment_scores =  get_weighted_reddit_sentiment(final_df_reddit, NUM_LABELS = NUM_LABELS)
                #reddit_sentiment_scores = np.round(reddit_sentiment_scores['Positive Sentiment']*100,2)
                reddit_sentiment_scores = np.round(reddit_sentiment_scores,2)
            
            elif source == 'telegram':
                final_df_telegram = labelled_df(df = telegram_df, 
                                                coin_id = token, 
                                                NUM_LABELS = NUM_LABELS, 
                                                source = source, 
                                                bert_classifier = classifier
                                                )
                telegram_sentiment_scores =  get_weighted_telegram_sentiment(final_df_telegram, NUM_LABELS = NUM_LABELS)
                #telegram_sentiment_scores = np.round(telegram_sentiment_scores['Positive Sentiment']*100,2)
                telegram_sentiment_scores = np.round(telegram_sentiment_scores,2)
            
        #Getting combined score from all sources

        # 60% Twitter, 25% Reddit, 15% Telegram
        weights = [0.6, 0.25, 0.15]
        sentiment_scores = [twitter_sentiment_scores, reddit_sentiment_scores, telegram_sentiment_scores]
        #Old start
        #sentiment_scores = np.nan_to_num(sentiment_scores, 0)
        #overall_sentiment = np.sum(list(map(lambda sent_score, weight: sent_score*weight, sentiment_scores, weights)))
        #overall_sentiment = np.nansum(sentiment_scores*weights) #Better way
        #Old end

        overall_sentiment = nanaverage(sentiment_scores, weights)

        token_name = tokens_dict.get(token)

        new_data = [[date, token, token_name, twitter_sentiment_scores, reddit_sentiment_scores, telegram_sentiment_scores, np.round(overall_sentiment,2)]]
        
        #Appending to sentiment DF
        aggregated_sentiment = aggregated_sentiment.append(pd.DataFrame(new_data, 
                                                        columns = ['Date', 'token_id', 'Name', 'Twitter', 'Reddit', 'Telegram', 'Overall']),ignore_index=True)

    # deleting any potential data already there
    flake.execute_snowflake(f"DELETE FROM SENTIMENT_ANALYSIS_V2 WHERE DATE = '{date}';", database='TOKENMETRICS_DEV', schema='ANALYTICS')

    # Saving aggregated coinwise sentiment to snowflake
    flake.write_snowflake(aggregated_sentiment, table_name='SENTIMENT_ANALYSIS_V2', if_exists='append', database='TOKENMETRICS_DEV', schema='ANALYTICS')

    # Getting the aggregated crypto market sentiment
    if len(aggregated_sentiment) != 0:
        token_id = tuple(token_id)
        query_crypto_mkt_cap = f'''select TOKEN_ID, MARKET_CAP from COINGECKO_TOKENS_LIVE_PRICE_SUMMARY WHERE TOKEN_ID IN\
                                {token_id} AND DATE(DATE) = '{date}' ''' #to_date(DATE) = to_date(select current_date()-1);'''

        market_cap = flake.read_snowflake(query_crypto_mkt_cap, database = 'CRYPTO_DB', schema = 'COINGECKO')
        market_cap = dict(market_cap.values)

        latest_sentiment = aggregated_sentiment.copy()

        latest_sentiment['Actual Market Cap'] = get_market_cap(latest_sentiment, market_cap)

        total_market_cap = np.sum(latest_sentiment['Actual Market Cap'])

        latest_sentiment['Scaled Market Cap'] = latest_sentiment['Actual Market Cap'].apply(lambda x: x/total_market_cap)

        latest_sentiment['Weighted Sentiment'] = latest_sentiment['Scaled Market Cap'] * latest_sentiment['OVERALL']

        current_cumulative_market_sentiment = np.round(np.sum(latest_sentiment['Weighted Sentiment']),2)
        
        cumulative_market_sentiment = pd.DataFrame({'DATE': [date], 'Market_Sentiment': [current_cumulative_market_sentiment]})
        print (f'Cumulative market sentiment = {current_cumulative_market_sentiment} % positive')

    else:
        cumulative_market_sentiment = pd.DataFrame({'DATE': [date], 'Market_Sentiment': [0]})
        
    # deleting any potential data already there
    flake.execute_snowflake(f"DELETE FROM CUMULATIVE_MARKET_SENTIMENT WHERE DATE = '{date}';", database='TOKENMETRICS_DEV', schema='ANALYTICS')

    # Saving aggregated coinwise sentiment to snowflake
    flake.write_snowflake(cumulative_market_sentiment, table_name='CUMULATIVE_MARKET_SENTIMENT', if_exists='append', database='TOKENMETRICS_DEV', schema='ANALYTICS')
