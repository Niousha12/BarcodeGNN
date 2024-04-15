import plotly.graph_objects as go
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

from dataset import DNADataset


def visualize_graph(data, node_color=None, size=300, with_labels=True, node_size=50):
    """
    Visualizes a PyTorch Geometric graph using NetworkX and Matplotlib.

    Parameters:
    - data (torch_geometric.data.Data): The PyTorch Geometric graph data object.
    - node_color (str or list): Color of the nodes.
    - size (int): Size of the figure.
    - with_labels (bool): Whether to display node labels.
    - node_size (int): Size of each node.
    """
    G = to_networkx(data, to_undirected=True)  # Convert to undirected graph from PyG data

    # Default colors
    if node_color is None:
        node_color = '#1f78b4'  # A nice blue color

    # Drawing nodes and edges
    plt.figure(figsize=(size // 100, size // 100))
    nx.draw(G, pos=nx.spring_layout(G), with_labels=with_labels, node_color=node_color, node_size=node_size)
    plt.show()


def plotly_visualize_graph(data):
    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Compute the position of each node using one of NetworkX's layout algorithms
    pos = nx.spring_layout(G)  # You can also try other layouts like nx.kamada_kawai_layout or nx.circular_layout

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in pos:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=[], line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()


def plotly_visualize_graph_label(data, labels=None):
    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Compute the position of each node using one of NetworkX's layout algorithms
    pos = nx.spring_layout(G)  # You can also try other layouts like nx.kamada_kawai_layout or nx.circular_layout

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in pos:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Use labels if provided, otherwise use the node index as the label
        if labels is not None:
            node_text.append(labels[node])
        else:
            node_text.append(str(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=[], line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()


def plotly_visualize_graph2(data, labels=None):
    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Compute the position of each node using one of NetworkX's layout algorithms
    pos = nx.spring_layout(G)  # You can also try other layouts like nx.kamada_kawai_layout or nx.circular_layout

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in pos:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Use labels if provided, otherwise use the node index as the label
        if labels is not None:
            node_text.append(labels[node])
        else:
            node_text.append(str(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',  # Combine markers and text
        hoverinfo='text',
        text=node_text,  # Text is the labels
        textposition="middle center",  # Position text in the middle of the markers
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=40,  # Increased size for visibility, originally 20, now twice
            color='lightblue',  # Node color changed to light blue
            line_width=2,
            opacity=0.5  # Set node opacity here
        ),
        textfont=dict(size=18, color='black'))  # Increased text font size, originally 12, now 1.5 times

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()


def plotly_visualize_graph3(data, labels=None):
    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Compute the position of each node using one of NetworkX's layout algorithms
    pos = nx.spring_layout(G)  # You can also try other layouts like nx.kamada_kawai_layout or nx.circular_layout

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in pos:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Use labels if provided, otherwise use the node index as the label
        if labels is not None:
            node_text.append(labels[node])
        else:
            node_text.append(str(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',  # Combine markers and text
        hoverinfo='text',
        text=node_text,  # Text is the labels
        textposition="middle center",  # Position text in the middle of the markers
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=50,  # Increased node size significantly
            color='lightpink',  # Node color changed to light pink
            line_width=2,
            opacity=0.5  # Set node opacity here
        ),
        textfont=dict(size=20, color='black'))  # Slightly increased text font size

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()


if __name__ == '__main__':
    # Example usage
    from torch_geometric.data import Data

    train_dataset = DNADataset('data/supervised_train.csv', k_mer=3, stride=1, data_count=5, truncate=30)

    graph = train_dataset.graphs[3]
    # graph = create_graph_from_sequence(sequence, k=3, stride=4, label=0)
    print(train_dataset.barcodes[3])
    # Visualize the graph
    # visualize_graph(graph, with_labels=False, size=800)  # , graph.node_indices
    plotly_visualize_graph3(graph, labels=graph.node_indices)
