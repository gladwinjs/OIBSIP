import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#load dataset
data = pd.read_csv("E:\\Iris.csv")
df = pd.DataFrame(data)
print("First few rows of the dataset:")
print(df.head(10))

#pair plots
sns.pairplot(df, hue = 'Species', palette = 'viridis', markers = ["o", "s", "D"])
plt.suptitle("Pair Plots of Features", y = 1.02)
plt.show()


#feature extraction
X = df.drop(columns=['Id', 'Species'])
y = df['Species']


#Distribution of each feature of each Species
plt.figure(figsize=(15,10))
for idx, feature in enumerate(X.columns):
    plt.subplot(2, 2, idx+1)
    sns.histplot(data = df, x=feature, hue='Species', multiple='stack', palette='viridis')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


#Strip plot
plt.figure(figsize=(15,10))
for idx, feature in enumerate(X.columns):
    plt.subplot(2, 2, idx+1)
    sns.stripplot(data=df, hue='Species', y=feature, palette='magma', jitter=True)
    plt.title(f'Stip plot of {feature}', fontsize=14)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
plt.show()



#train the model
X = df.drop(columns=['Id', 'Species'])
y = df['Species']
target_names = y.unique() 
target_map ={name: idx for idx, name in enumerate(target_names)}
y = y.map(target_map)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring = 'accuracy')
    k_scores.append(scores.mean())
best_k = k_range[np.argmax(k_scores)]
print(f'Best k value: {best_k}')


#KNN graph 
plt.figure(figsize=(10,6))
plt.plot(k_range, k_scores, marker='o', linestyle='--', color='b', label='Accuracy')
plt.xticks(k_range)
plt.xlabel('Value of k for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Hyperparameter Tuning')
plt.grid(True, linestyle = '--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()



#accuracy and confusion matrix calculation

classifier = KNeighborsClassifier(n_neighbors=best_k)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy:{accuracy:.2f}')
conf_matrix = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=target_names))

#Confusion matrix plot
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="BuPu", xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#predict on new data
new_data = pd.DataFrame({
    "SepalLengthCm": [1.5],
    "SepalWidthCm": [2.9],
    "PetalLengthCm": [5.5],
    "PetalWidthCm": [1.9]
})

# Standardize the new data using the previously fitted scaler
new_data = scaler.transform(new_data)

# Predict the species of the new data
new_prediction = classifier.predict(new_data)

# Map the numeric prediction back to species name
predicted_species = target_names[new_prediction[0]]
print(f'Predicted species for the new data: {predicted_species}')

