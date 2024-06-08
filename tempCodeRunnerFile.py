import os
import nltk
import networkx as nx
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# Function to load data from files
def load_data_from_files(folder):
    data = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                data[filename] = text
    return data


# Function to create a graph from text data
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


# Function to calculate the maximum common subgraph (MCS)
def calculate_mcs(graph1, graph2):
    common_nodes = set(graph1.nodes) & set(graph2.nodes)
    mcs_size = len(common_nodes)
    max_size = max(len(graph1.nodes), len(graph2.nodes))
    return 1 - (mcs_size / max_size)


# Set paths to training data folders
folders = ["Fashion and Beauty", "Health and Fitness", "Lifestyle and Hobbies"]

# Load data from folders
data = {}
for folder in folders:
    folder_path = os.path.join(
        "D:\\UNI STUDY\\Study\\Semester 6\\GT\\Project\\Scrapped Data", folder
    )
    data[folder] = load_data_from_files(folder_path)

# Create graphs for each file
graphs = {}
for class_name, files_data in data.items():
    for file_name, file_text in files_data.items():
        graph = create_word_graph(file_text)
        key = (class_name, file_name)
        graphs[key] = graph

# Step 1: Prepare Data
X_train = []
y_train = []

for (class_name, file_name), graph in graphs.items():
    features = []
    for (_, _), other_graph in graphs.items():
        if (class_name, file_name) != (_, _):
            features.append(calculate_mcs(graph, other_graph))
    X_train.append(features)
    y_train.append(class_name)

# Step 2: Train KNN Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 3: Predict Labels
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

# for folder in ["Fashion and Beauty", "Health and Fitness", "Lifestyle and Hobbies"]:
#     folder_path = os.path.join(
#         "D:\\UNI STUDY\\Study\\Semester 6\\GT\\Project\\Testing Data", folder
#     )
folder_path = "D:\\UNI STUDY\\Study\\Semester 6\\GT\\Project\\Testing Data"
folder_data = load_data_from_files(folder_path)
for file_name, file_text in folder_data.items():
    graph = create_word_graph(file_text)
    features = []
    for (_, _), other_graph in graphs.items():
        features.append(calculate_mcs(graph, other_graph))
    X_test.append(features)

# Predict labels for testing data
y_pred = knn.predict(X_test)

# Step 4: Evaluate Performance
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
