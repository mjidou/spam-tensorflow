import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('spam2.csv')

df['Message'] = df['text_hi'].astype(str) + " " + df['text_de'].astype(str) + " " + df['text_fr'].astype(str)

df = df[['Category', 'Message']]

df['Category'] = df['Category'].apply(lambda x: 1 if x == 'ham' else 0)

X = df['Message']
Y = df['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

model = LogisticRegression()

model.fit(X_train_features, Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print('Accuracy on training data:', accuracy_on_training_data)
print('Accuracy on test data:', accuracy_on_test_data)

email = ['spam']
email_data_features = feature_extraction.transform(email)
predictions = model.predict(email_data_features)
print(predictions)

for i, prediction in enumerate(predictions):
    if prediction == 1:
        print(f'Message {i + 1}: Ham')
    else:
        print(f'Message {i + 1}: Spam')
