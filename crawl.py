from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
import pandas as pd
import nltk
#config will allow us to access the specified url for which we are #not authorized. Sometimes we may get 403 client error while parsing #the link to download the article.
nltk.download('punkt')

googlenews=GoogleNews(start='01/01/2020',end='09/17/2020')
googlenews.search('IFTRT')
result=googlenews.result()
df=pd.DataFrame(result)
for i in range(2,20):
    googlenews.getpage(i)
    result=googlenews.result()
    df=pd.DataFrame(result)
list=[]
for ind in df.index:
    dict={}
    article = Article(df['link'][ind])
    article.download()
    article.parse()
    article.nlp()
    dict['Date']=df['date'][ind]
    dict['Title']=article.title
    dict['Summary']=article.summary
    list.append(dict)
news_df=pd.DataFrame(list)
news_df.to_csv("headlines.csv")