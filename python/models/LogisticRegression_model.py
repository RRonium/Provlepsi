import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv("")

#These are the variables who's values are to be aquired by the front end
response_variable = "" 
response_variable = ""
split_fraction = None 
classifier_type = ""
C_param = 1.0

label_encoder = LabelEncoder()
df[response_variable] = label_encoder.fit_transform(df[response_variable])

class_key_map = {int(index): str(label) for index, label in enumerate(label_encoder.classes_)}
K_classes = len(label_encoder.classes_)

X = df.drop(columns=[response_variable])
y = df[response_variable]

#Manual split as Time-Series data, can't use train_test_split due to its randomness
split_index = int(len(df) * (1 - split_fraction))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

if classifier_type == "logistic":
    lm = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=C_param, max_iter=1000)
elif classifier_type == "lda":
    lm = LinearDiscriminantAnalysis()
elif classifier_type == "qda":
    lm = QuadraticDiscriminantAnalysis()

lm.fit(X_train, y_train)

sample_predictions = lm.predict(X_test)
sample_probabilities = lm.predict_proba(X_test)

#Error Evaluation, a little different from the other regression models that we've used in the past
accuracy = accuracy_score(y_test, sample_predictions)
precision = precision_score(y_test, sample_predictions, average='weighted')
recall = recall_score(y_test, sample_predictions, average='weighted')
conf_matrix = confusion_matrix(y_test, sample_predictions).tolist()

#User's predicitons
user_entry = {} 
user_df = pd.DataFrame([user_entry])[X.columns]

predicted_class_code = int(lm.predict(user_df)[0])
predicted_class_label = class_key_map[predicted_class_code]

predicted_probabilities = {
    class_key_map[i]: float(prob) 
    for i, prob in enumerate(lm.predict_proba(user_df)[0])
}