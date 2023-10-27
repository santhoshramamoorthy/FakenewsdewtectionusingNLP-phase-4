# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:09.090352Z","iopub.execute_input":"2023-10-27T04:25:09.090818Z","iopub.status.idle":"2023-10-27T04:25:09.111791Z","shell.execute_reply.started":"2023-10-27T04:25:09.090776Z","shell.execute_reply":"2023-10-27T04:25:09.110750Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:09.114058Z","iopub.execute_input":"2023-10-27T04:25:09.114420Z","iopub.status.idle":"2023-10-27T04:25:10.242375Z","shell.execute_reply.started":"2023-10-27T04:25:09.114383Z","shell.execute_reply":"2023-10-27T04:25:10.241484Z"}}
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:10.243502Z","iopub.execute_input":"2023-10-27T04:25:10.243810Z","iopub.status.idle":"2023-10-27T04:25:12.786206Z","shell.execute_reply.started":"2023-10-27T04:25:10.243785Z","shell.execute_reply":"2023-10-27T04:25:12.785004Z"}}
fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:12.787408Z","iopub.execute_input":"2023-10-27T04:25:12.787727Z","iopub.status.idle":"2023-10-27T04:25:12.812243Z","shell.execute_reply.started":"2023-10-27T04:25:12.787699Z","shell.execute_reply":"2023-10-27T04:25:12.811273Z"}}
fake['Category'] = 'fake'
fake

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:12.815088Z","iopub.execute_input":"2023-10-27T04:25:12.815417Z","iopub.status.idle":"2023-10-27T04:25:12.831090Z","shell.execute_reply.started":"2023-10-27T04:25:12.815389Z","shell.execute_reply":"2023-10-27T04:25:12.830043Z"}}
true['Category'] = 'true'
true

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:12.832993Z","iopub.execute_input":"2023-10-27T04:25:12.833311Z","iopub.status.idle":"2023-10-27T04:25:12.858234Z","shell.execute_reply.started":"2023-10-27T04:25:12.833283Z","shell.execute_reply":"2023-10-27T04:25:12.857174Z"}}
data = pd.concat([fake, true], ignore_index=True)
data

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:12.859531Z","iopub.execute_input":"2023-10-27T04:25:12.859850Z","iopub.status.idle":"2023-10-27T04:25:12.878307Z","shell.execute_reply.started":"2023-10-27T04:25:12.859821Z","shell.execute_reply":"2023-10-27T04:25:12.877320Z"}}
data['Category'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:12.879444Z","iopub.execute_input":"2023-10-27T04:25:12.879814Z","iopub.status.idle":"2023-10-27T04:25:12.938855Z","shell.execute_reply.started":"2023-10-27T04:25:12.879777Z","shell.execute_reply":"2023-10-27T04:25:12.938061Z"}}
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Category'] = le.fit_transform(data['Category'])
data['date'] = le.fit_transform(data['date'])
data['subject'] = le.fit_transform(data['subject'])

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:12.940237Z","iopub.execute_input":"2023-10-27T04:25:12.940596Z","iopub.status.idle":"2023-10-27T04:25:12.947737Z","shell.execute_reply.started":"2023-10-27T04:25:12.940568Z","shell.execute_reply":"2023-10-27T04:25:12.946740Z"}}
data['Category']

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:12.949272Z","iopub.execute_input":"2023-10-27T04:25:12.949658Z","iopub.status.idle":"2023-10-27T04:25:12.960529Z","shell.execute_reply.started":"2023-10-27T04:25:12.949594Z","shell.execute_reply":"2023-10-27T04:25:12.959542Z"}}
data['date'] 

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:12.961812Z","iopub.execute_input":"2023-10-27T04:25:12.962143Z","iopub.status.idle":"2023-10-27T04:25:12.972834Z","shell.execute_reply.started":"2023-10-27T04:25:12.962116Z","shell.execute_reply":"2023-10-27T04:25:12.971896Z"}}
data['subject'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:12.973990Z","iopub.execute_input":"2023-10-27T04:25:12.974328Z","iopub.status.idle":"2023-10-27T04:25:12.982905Z","shell.execute_reply.started":"2023-10-27T04:25:12.974269Z","shell.execute_reply":"2023-10-27T04:25:12.981901Z"}}
data['title'].shape

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:12.984114Z","iopub.execute_input":"2023-10-27T04:25:12.984444Z","iopub.status.idle":"2023-10-27T04:25:32.114741Z","shell.execute_reply.started":"2023-10-27T04:25:12.984407Z","shell.execute_reply":"2023-10-27T04:25:32.113696Z"}}
vectorizer = TfidfVectorizer()
title = vectorizer.fit_transform(data['title'])
text = vectorizer.transform(data['text'])


# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:32.117764Z","iopub.execute_input":"2023-10-27T04:25:32.118058Z","iopub.status.idle":"2023-10-27T04:25:32.123884Z","shell.execute_reply.started":"2023-10-27T04:25:32.118034Z","shell.execute_reply":"2023-10-27T04:25:32.122921Z"}}
 title

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:32.124993Z","iopub.execute_input":"2023-10-27T04:25:32.125278Z","iopub.status.idle":"2023-10-27T04:25:32.146229Z","shell.execute_reply.started":"2023-10-27T04:25:32.125253Z","shell.execute_reply":"2023-10-27T04:25:32.145176Z"}}
from sklearn.model_selection import train_test_split
X = title
y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:25:32.147343Z","iopub.execute_input":"2023-10-27T04:25:32.147604Z","iopub.status.idle":"2023-10-27T04:29:37.363293Z","shell.execute_reply.started":"2023-10-27T04:25:32.147581Z","shell.execute_reply":"2023-10-27T04:29:37.362265Z"}}
model = SVC()
model.fit(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-27T04:29:37.364346Z","iopub.execute_input":"2023-10-27T04:29:37.364622Z","iopub.status.idle":"2023-10-27T04:29:55.990111Z","shell.execute_reply.started":"2023-10-27T04:29:37.364598Z","shell.execute_reply":"2023-10-27T04:29:55.989153Z"}}
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# %% [code]