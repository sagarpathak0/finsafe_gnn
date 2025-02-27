from flask import Flask, jsonify
import torch
import networkx as nx
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Read the Excel file
df = pd.read_excel("trans.xlsx")  # Replace with your actual file name

# Create a directed graph
G = nx.DiGraph()

# Add edges from the DataFrame
edges = list(zip(df["Source"], df["Destination"]))
G.add_edges_from(edges)
# Convert node labels to indices
node_mapping = {node: i for i, node in enumerate(G.nodes())}
edges_index = torch.tensor([[node_mapping[src], node_mapping[dst]] for src, dst in G.edges()], dtype=torch.long).t()

# Convert to PyG format
data = from_networkx(G)
data.edge_index = edges_index
# 2. Define Node2Vec hyperparameters (must match training)
embedding_dim = 64
walk_length = 20
context_size = 10
walks_per_node = 10
num_negative_samples = 1
sparse = False

# 3. Instantiate the Node2Vec model and load its state.
node2vec = Node2Vec(
    data.edge_index,
    embedding_dim=embedding_dim,
    walk_length=walk_length,
    context_size=context_size,
    walks_per_node=walks_per_node,
    num_negative_samples=num_negative_samples,
    sparse=sparse
)
node2vec.load_state_dict(torch.load("node2vec_model.pt", map_location=torch.device('cpu')))
node2vec.eval()  # Set to evaluation mode

# 4. Load the Isolation Forest model.
iso_forest = joblib.load("isolation_forest_model.pkl")

@app.route('/predict', methods=['GET'])
def predict():
    # Generate node embeddings using the Node2Vec model.
    with torch.no_grad():
        embeddings = node2vec().detach().cpu().numpy()

    # Normalize embeddings.
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Run anomaly detection using Isolation Forest.
    anomaly_scores = iso_forest.decision_function(scaled_embeddings)
    outlier_indices = np.where(anomaly_scores < 0)[0]

    # Prepare results.
    result = {
        "outlier_indices": outlier_indices.tolist(),
        "anomaly_scores": anomaly_scores.tolist()
    }
    return jsonify(result)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
