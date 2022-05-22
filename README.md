# NLP
Unit 12—Tales from the Crypto

In this project we explore the public sentiment for two crpytocurrencies, Bitcoin and Ethereum. We will apply natural language processing to understand the sentiment in the latest news articles featuring these coins. We will also apply fundamental NLP techniques to better understand the other factors involved with the coin prices such as common words and phrases and organizations and entities mentioned in the articles.


Using the news API keys, we pull articles regarding the cryptocurrencies and calculate the positive, negative, and neutral sentiments regarding each coin.


<img width="541" alt="Screen Shot 2022-05-22 at 12 42 45 AM" src="https://user-images.githubusercontent.com/99091066/169679048-cf756c64-31b4-4a0d-8b83-839f57f0223d.png">

  > Scores for Bitcoin


<img width="524" alt="Screen Shot 2022-05-22 at 12 43 14 AM" src="https://user-images.githubusercontent.com/99091066/169679064-d3b06425-f278-45bd-9a19-3ca7c4b9f964.png">


  > Scores for Ethereum


We then apply the `describe` feature to the dataframes to analyze the mean positive/negative, compound, and max scores.

<img width="263" alt="Screen Shot 2022-05-22 at 12 44 42 AM" src="https://user-images.githubusercontent.com/99091066/169679103-66d3b44e-4124-4313-bd8f-a3ab3cb4e747.png">

Based on the results above, we were able to answer the following questions: 

##### Q: Which coin had the highest mean positive score?

A: *Bitcoin has the highest mean positive score at 0.07210 compared to Ethereum's 0.053050.*

##### Q: Which coin had the highest compound score?

A: *Bitcoin had a higher mean compound score at 0.039835. Since this falls within the +0.05 to -0.05 range, the sentiment is neutral. Whereas for Ethereum, the compound score is less than -0.05, indicating a negative sentiment.*

##### Q. Which coin had the highest positive score?

A: *Bitcoin had an overall higher positive score.*

### Natural Language Processing

Using the articles for each coin, we tokenized each word in the article for further analysis. 

```` python
# Complete the tokenizer function
def tokenizer(text):
    """Tokenizes text."""
    #sw = set(stopwords.words('english')+stopw)
    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', text)
    words = word_tokenize(re_clean)
    words=[word.lower() for word in words if word.lower() not in stopw]
    lem = [lemmatizer.lemmatize(word) for word in words]
    output = [word.lower() for word in lem if word.lower() not in sw]
    
    return lem 
 ````
 
 We applied the tokenizer that was defined above, and added another column to the existing dataframe to create the table below.
 
 
 <img width="640" alt="Screen Shot 2022-05-22 at 12 48 53 AM" src="https://user-images.githubusercontent.com/99091066/169679211-d6afda05-2b2c-42f8-89e0-120a0cb18efe.png">


### NGrams and Frequency Analysis
 
from `collections` we import `Counter` 
and from `nltk` we import `ngrams`

For Bitcoin and Ethereum, we use the counter feature where N = 2, and count each time two words appear in the `tokenizer` column created in the previous section.



<img width="643" alt="Screen Shot 2022-05-22 at 12 51 25 AM" src="https://user-images.githubusercontent.com/99091066/169679280-5d1bf698-f9c6-450c-a3ca-7430bb36c3ac.png">

We then define `corpus` and `corpeth` as the `tokenizer` column in each dataframe.

Since the tokenizer column for each dataset was an object dtype, we converted the object to a string and concatenated each string using the `pandas.Series.str.cat` function. 

Using the Counter and the most_common feature, we analyze the top 10 most common words in each article.

For Bitcoin, the top 10 words are as follows:

 Bitcoin (13), Cryptocurrency (12), World (7), Week (5), Reuters (5), Investor (4), Dropped (4), Previous (4), Closebitcoin (4), and Biggest (4). 
 

For Ethereum, the top 10 words are as follows: 

Cryptocurrency (10), Bitcoin (8), World (7), Biggest(5), Reuters (5), Last (5), Ethereum (4), NFT (4), Week (4), and Previous (4). 
 
### Word Clouds

The tokenized words were then displayed on a word map , with the larger sizes being the words that were mentioned more tha others.

Bitcoin:



<img width="640" alt="Screen Shot 2022-05-22 at 1 01 24 AM" src="https://user-images.githubusercontent.com/99091066/169679523-91f13d04-99a6-49f5-aaf3-997b6157bd7a.png">


Ethereum:


<img width="635" alt="Screen Shot 2022-05-22 at 1 01 44 AM" src="https://user-images.githubusercontent.com/99091066/169679568-64c8da36-8333-4246-84a8-f41c4c39f13e.png">


### NER (Named Entity Recognition) 

We first import `spacy` and from `spacy`, import `displacy`

We then take all of the Bitocoin text and concatenate as we did previously and run the NER processor on the text. 

````python

# Run the NER processor on all of the text
btcner=nlp(bner)

# Render the visualization
displacy.render(btcner, style='ent')


````

Below is an example of the output for Bitcoin: 


<img width="709" alt="Screen Shot 2022-05-22 at 1 06 13 AM" src="https://user-images.githubusercontent.com/99091066/169679704-eda076b3-3ed5-4f38-8b81-5cdf18b0cb92.png">

  > the function is able to identidy each word as either a person, organization, date, to name a few. For example, it recognied Kristy Kilburn as a person, Wednesday as a date, and North London as a location. However it does have its limitations as it recognized "Gucci Handbag" as a person.


Lastly, each entity was counted. For Bitcoin, the entity counts are as follows: 


<img width="157" alt="Screen Shot 2022-05-22 at 1 09 22 AM" src="https://user-images.githubusercontent.com/99091066/169679770-d57ac861-db9d-4815-b8f7-6cbc09054173.png">

  > Money was recognized 12 times, there were 8 people mentioned, and 19 dates.
