import requests
from bs4 import BeautifulSoup
import re
import os


def get_text_from_url(url, file_number):
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text(separator=" ")

    text = re.sub(r"\s+", " ", text)

    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    words = []
    word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if word_count + len(sentence_words) <= 520:
            words.extend(sentence_words)
            word_count += len(sentence_words)
        else:
            break

    final_text = " ".join(words[:520])

    filename = f"{file_number}.txt"
    with open(filename, "w") as file:
        file.write(final_text)

    return final_text


def count_words(text):
    words = text.split()
    return len(words)


def get_next_file_number():
    file_number = 1
    while os.path.exists(f"{file_number}.txt"):
        file_number += 1
    return file_number


url = "https://washingtoninst.org/why-you-yes-you-need-a-hobby/"
file_number = get_next_file_number()
scraped_text = get_text_from_url(url, file_number)
print(f"Total words: {count_words(scraped_text)}")
