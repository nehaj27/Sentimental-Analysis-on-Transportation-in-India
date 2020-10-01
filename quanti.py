import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import time 
import datetime 


raw_data=pd.read_csv('try.csv',parse_dates=[0],infer_datetime_format=True)


for i in range(0,len(raw_data)):
	string=raw_data['Date'][i]
	raw_data['Date'][i]=time.mktime(datetime.datetime.strptime(string, 
                                             "%m/%d/%Y").timetuple())

reindexed_data = raw_data['Title']
reindexed_data.index = raw_data['Date']
reindexed_data.index = pd.to_datetime(reindexed_data.index, unit='s')


print(reindexed_data.head())
blobs = [TextBlob(reindexed_data[i]) for i in range(reindexed_data.shape[0])]


polarity = [blob.polarity for blob in blobs]
subjectivity = [blob.subjectivity for blob in blobs]

sentiment_analysed = pd.DataFrame({'headline_text':reindexed_data, 
                                   'polarity':polarity, 
                                   'subjectivity':subjectivity},
                                  index=reindexed_data.index)



monthly_averages = sentiment_analysed.resample('M').mean()

fig, ax = plt.subplots(2, figsize=(18,10))
ax[0].plot(monthly_averages['subjectivity'], label='Monthly mean subjectivity');
ax[0].set_title('Mean subjectivity scores');
ax[0].legend(loc='upper left');
ax[1].plot(monthly_averages['polarity'], label='Monthly mean polarity');
ax[1].set_title('Mean polarity scores');
ax[1].legend(loc='upper left');
plt.show()

