import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# loading dataset

path = r"G:\CODESOFT TASK\Genre Classification Dataset\train_data.txt"

df = pd.read_csv(
    path,
    sep=" ::: ",
    engine="python",
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
)

print("Dataset Shape:", df.shape)
print("\nFirst 1000 Rows:\n")
print(df.head(1000))

# handle missing values

df["DESCRIPTION"] = df["DESCRIPTION"].fillna("")

# taking description and genre columns

X = df["DESCRIPTION"]
y = df["GENRE"]

# converting text into numbers

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9
)
X = tfidf.fit_transform(X)

print("TF-IDF Shape:", X.shape)

# splitting data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# training model

model = LinearSVC(
    C=1.5,
    random_state=42
)

model.fit(X_train, y_train)

# prediction on test data

y_pred = model.predict(X_test)

# checking accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========\n")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\n========== CLASSIFICATION REPORT ==========\n")

print(classification_report(y_test, y_pred, zero_division=0))

# saving model

with open("movie_genre_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(tfidf, file)

print("\nModel and TF-IDF Vectorizer saved successfully!")

# taking input from user

while True:
    description = input(
        "\nEnter movie description (type 'exit' to quit): "
    )

    if description.lower() == "exit":
        print("Program closed.")
        break

    # convert input

    new_movie = [description]
    new_movie = tfidf.transform(new_movie)

    # predict genre

    prediction = model.predict(new_movie)
    genre = prediction[0]

    print("\nPredicted Genre:", genre)

    # recommend movies

    movies = df[
        df["GENRE"].str.lower() == genre.lower()
    ]["TITLE"].drop_duplicates()

    if len(movies) > 0:
        print("\nRecommended Movies:")

        for movie in movies.head(10):
            print("-", movie)
    else:
        print("No recommendations found.")