import os
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def plot_confusion_matrix(conf_matrix, accuracy, labels):
    plt.figure(figsize=(6, 5))
    sns.set(font_scale=0.8)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("\nPredicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.title(
        "Confusion Matrix\n\nAccuracy: {:.2f}%".format(accuracy * 100),
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.gca().tick_params(length=0)
    plt.tight_layout()
    plt.show()


def load_data_from_files(folder):
    data = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                data[filename] = text
    return data


def create_word_graph(text):
    words = nltk.word_tokenize(text.lower())
    graph = nx.Graph()
    unique_words = set(words)
    for word in unique_words:
        graph.add_node(word)
    for i in range(len(words) - 1):
        if not graph.has_edge(words[i], words[i + 1]):
            graph.add_edge(words[i], words[i + 1])
    return graph


# def plot_word_graph(graph, title):
#     plt.figure(figsize=(12, 7))
#     pos = nx.shell_layout(graph)
#     nx.draw(
#         graph, pos, with_labels=True, node_color="skyblue", node_size=500, font_size=7
#     )
#     plt.title(title)
#     plt.show()


def calculate_mcs(graph1, graph2):
    common_nodes = set(graph1.nodes) & set(graph2.nodes)
    mcs_size = len(common_nodes)
    max_size = max(len(graph1.nodes), len(graph2.nodes))
    return 1 - (mcs_size / max_size)


folders = ["Fashion and Beauty", "Health and Fitness", "Lifestyle and Hobbies"]

data = {}
for folder in folders:
    folder_path = os.path.join("Scrapped Data", folder)
    data[folder] = load_data_from_files(folder_path)
# print(data)
graphs = {}
for class_name, files_data in data.items():
    # print(files_data)
    for file_name, file_text in files_data.items():
        graph = create_word_graph(file_text)
        key = (class_name, file_name)
        graphs[key] = graph

X_train = []
y_train = []
# print(graphs.items())
for (class_name, file_name), graph in graphs.items():
    features = []
    # print((class_name, file_name), graph)
    for (_, _), other_graph in graphs.items():
        # print((_, _), other_graph)
        if (class_name, file_name) != (_, _):
            features.append(calculate_mcs(graph, other_graph))
    X_train.append(features)
    y_train.append(class_name)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

X_test = []
y_test = [
    "Fashion and Beauty",
    "Fashion and Beauty",
    "Fashion and Beauty",
    "Health and Fitness",
    "Health and Fitness",
    "Health and Fitness",
    "Lifestyle and Hobbies",
    "Lifestyle and Hobbies",
    "Lifestyle and Hobbies",
]


folder_path = "Testing Data"
folder_data = load_data_from_files(folder_path)
for file_name, file_text in folder_data.items():
    graph = create_word_graph(file_text)
    features = []
    for (_, _), other_graph in graphs.items():
        features.append(calculate_mcs(graph, other_graph))
    X_test.append(features)


y_pred = knn.predict(X_test)
# print(y_pred)
class_results = defaultdict(lambda: {"true": [], "predicted": []})

for file_name, true_label, predicted_label in zip(folder_data.keys(), y_test, y_pred):
    class_results[true_label]["true"].append(true_label)
    class_results[true_label]["predicted"].append(predicted_label)

true_labels = []
predicted_labels = []

for class_name, results in class_results.items():
    true_labels.extend(results["true"])
    predicted_labels.extend(results["predicted"])

print("\nPredicted Labels:")
for file_name, predicted_label in zip(folder_data.keys(), y_pred):
    print(f"\tTest File: {file_name}, Predicted Label: {predicted_label}")

conf_matrix = confusion_matrix(true_labels, predicted_labels)

accuracy = accuracy_score(true_labels, predicted_labels)

print("\nConfusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)


# for file_name, file_text in folder_data.items():
#     graph = create_word_graph(file_text)
#     plot_word_graph(graph, title=f"Word Graph for {file_name}")

plot_confusion_matrix(
    conf_matrix,
    accuracy,
    labels=["Fashion and Beauty", "Health and Fitness", "Lifestyle and Hobbies"],
)
