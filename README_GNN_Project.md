# Node Classification using Graph Neural Networks

## Overview
This project implements a Graph Convolutional Network (GCN) for node classification on a citation network dataset. The objective is to understand and demonstrate how graph-structured data can be effectively processed using Graph Neural Networks (GNNs).

The model is trained on the Cora dataset, where each node represents a research paper and edges represent citation relationships. The task is to classify each paper into one of several categories based on its features and connections.

---

## Objectives
- To understand the fundamentals of Graph Neural Networks  
- To implement a Graph Convolutional Network using PyTorch Geometric  
- To perform semi-supervised node classification on graph data  
- To gain hands-on experience with message passing in graphs  

---

## Technologies Used
- Python  
- PyTorch  
- PyTorch Geometric  

---

## Dataset
The project uses the Cora dataset, a widely used benchmark dataset for node classification tasks in graph learning.

- Nodes: Research papers  
- Edges: Citation links between papers  
- Features: Bag-of-words representation of each paper  
- Classes: Subject categories of papers  

The dataset is automatically downloaded using PyTorch Geometric.

---

## Model Architecture
The implemented model is a two-layer Graph Convolutional Network:

- First Layer: Graph convolution followed by ReLU activation  
- Second Layer: Graph convolution producing class probabilities  

The model leverages neighborhood aggregation (message passing) to learn node representations.

---

## Installation

Install the required dependencies using:

pip install torch torchvision torchaudio  
pip install torch-geometric  

---

## Usage

Run the Python script:

python main.py  

The model will train on the dataset and output the classification accuracy.

---

## Results
The model achieves competitive accuracy on the Cora dataset, demonstrating the effectiveness of Graph Convolutional Networks for semi-supervised learning on graph data.

---

## Learning Outcomes
- Understanding of graph-based data structures  
- Practical implementation of Graph Neural Networks  
- Familiarity with PyTorch Geometric framework  
- Insight into node classification tasks  

---

## Future Work
- Implementation of advanced GNN architectures such as GraphSAGE and GAT  
- Hyperparameter tuning for improved performance  
- Application to real-world datasets  
- Visualization of graph embeddings and predictions  

---

## Author
Muskan Thakur  
CSE (AI & ML) Student  

GitHub: https://github.com/muskan-101  
LinkedIn: https://www.linkedin.com/in/muskan-thakur-6a552a328/  
