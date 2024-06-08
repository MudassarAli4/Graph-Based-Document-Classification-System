import os
import re
import string
import nltk
from nltk.corpus import stopwords

source_folder = (
    "D:\\UNI STUDY\\Study\\Semester 6\\GT\\Project\\Data\\Lifestyle and Hobbies"
)
target_folder = "D:\\UNI STUDY\\Study\\Semester 6\\GT\\Project\\Scrapped Data\\Lifestyle and Hobbies"

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# nltk.download('stopwords')


def preprocess_text(text):
    text = text.lower()

    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    preprocessed_text = " ".join(filtered_tokens)
    return preprocessed_text


for filename in os.listdir(source_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()
            preprocessed_text = preprocess_text(text)

            with open(target_path, "w") as target_file:
                target_file.write(preprocessed_text)
