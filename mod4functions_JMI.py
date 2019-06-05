def quick_table(tuples, col_names=None, caption =None):
    """Accepts a bigram output tuple of tuples and makes captioned table."""
    import pandas as pd
    from IPython.display import display
    if col_names == None:
    
        df = pd.DataFrame.from_records(tuples)
        
    else:
        
        df = pd.DataFrame.from_records(tuples,columns=col_names)
        dfs = df.style.set_caption(caption)
        display(dfs)
            
    return df

def compare_word_cloud(text1,label1,text2,label2):
    """Compares the wordclouds from 2 sets of texts"""
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    wordcloud1 = WordCloud(max_font_size=80, max_words=200, background_color='white').generate(' '.join(text1))
    wordcloud2 = WordCloud(max_font_size=80, max_words=200, background_color='white').generate(' '.join(text2))


    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,15))
    ax[0].imshow(wordcloud1, interpolation='bilinear')
    ax[0].set_aspect(1.5)
    ax[0].axis("off")
    ax[0].set_title(label1, fontsize=20)

    ax[1].imshow(wordcloud2, interpolation='bilinear')
    ax[1].set_aspect(1.5)
    ax[1].axis("off")
    ax[1].set_title(label2, fontsize=20)

    fig.tight_layout()
    return fig,ax


# Define get_tags_ats to accept a list of text entries and return all found tags and ats as 2 series/lists
def get_tags_ats(text_to_search,exp_tag = r'(#\w*)',exp_at = r'(@\w*)', output='series',show_counts=False):
    """Accepts a list of text entries to search, and a regex for tags, and a regex for @'s.
    Joins all entries in the list of text and then re.findsall() for both expressions.
    Returns a series of found_tags and a series of found_ats.'"""
    import re
    import pandas as pd
    # Create a single long joined-list of strings
    text_to_search_combined = ' '.join(text_to_search)
        
    # print(len(text_to_search_combined), len(text_to_search_list))
    found_tags = re.findall(exp_tag, text_to_search_combined)
    found_ats = re.findall(exp_at, text_to_search_combined)
    
    if output.lower() == 'series':
        found_tags = pd.Series(found_tags, name='tags')
        found_ats = pd.Series(found_ats, name='ats')
        
        if show_counts==True:
            print(f'\t{found_tags.name}:\n{found_tags.value_counts()} \n\n\t{found_ats.name}:\n{found_ats.value_counts()}')
                
    if (output.lower() != 'series') & (show_counts==True):
        raise Exception('output must be set to "series" in order to show_counts')
                       
    return found_tags, found_ats


def clean_text(series,is_tokens=False,as_tokens=False, urls=True, hashtags=True, mentions=True, stopwords=True, verbose=False):
    """Accepts a series/df['column'] and tokenizes, removes urls, hasthtags, and @s using regex before tokenizing and removing stopwrods"""
    import pandas as pd
    import re, nltk
    from nltk.corpus import stopwords
    
    series_cleaned=series.copy()
    
    # Remove URLS
    if urls==True:
        urls = re.compile(r"(http[s]?://\w*\.\w*/+\w+)")
        series_cleaned = series_cleaned.apply(lambda x: urls.sub(' ', x))
            
        if verbose==True:
            print('URLs removed...')
            
    # Remove hashtags
    if hashtags==True:
        hashtags = re.compile(r'(\#\w*)')
        series_cleaned = series_cleaned.apply(lambda x: hashtags.sub(' ', x))
        
        if verbose==True:
            print('Hashtags removed...')
    
    # Remove mentions
    if mentions==True:
        mentions = re.compile(r'(\@\w*)')
        series_cleaned = series_cleaned.apply(lambda x: mentions.sub(' ',x))

        if verbose==True:
            print('Mentions removed...')
    
    
    # Regexp_tokenize stopped words (to keep contractions)
    if is_tokens==False:
        pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
        series_cleaned = series_cleaned.apply(lambda x: nltk.regexp_tokenize(x,pattern))
        if verbose==True:
            print('Text regexp_tokenized...\n')
    
    
    # Filter Out Stopwords
    stopwords_list = []
    from nltk.corpus import stopwords
    import string
    
    # Generate Stopwords List
    stopwords_list = stopwords.words('english')
    stopwords_list += list(string.punctuation)
    stopwords_list += ['http','https','...','``','co','“','’','‘','”',
                       'rt',"n't","''","RT",'u','s',"'s",'?']#,'@','#']
    stopwords_list += [0,1,2,3,4,5,6,7,8,9]
    stopwords_list +=['RT','rt',';']
     
    if stopwords==True:
   
        for s in range(len(series_cleaned)):
            text =[]
            text_stopped = []
            text = series_cleaned[s]
            text_stopped = [x.lower() for x in text if x not in stopwords_list]
            series_cleaned[s]= text_stopped
        
        if verbose==True:
            print('Stopwords removed...')
       
    if as_tokens==False:
        series_cleaned = series_cleaned.apply(lambda x: ' '.join(x))
    
    print('\n')
    return series_cleaned
