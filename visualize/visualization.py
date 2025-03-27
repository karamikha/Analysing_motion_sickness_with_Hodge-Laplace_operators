import matplotlib.pyplot as plt
import mne
import networkx as nx
import seaborn as sns


def VisualizeMNE(data, channels_names):
    channels_q = len(channels_names)

    info = mne.create_info(ch_names=channels_names, sfreq=500, ch_types=["eeg" for _ in range(channels_q)])
    raw = mne.io.RawArray(data, info)

    raw.plot(scalings=dict(eeg=150000), n_channels=channels_q, butterfly=False)
    plt.show()


def VisualizeMatplotlib(data, channels_names, times):
    channels_q = len(channels_names)

    fig, axes = plt.subplots(channels_q, 1, figsize=(10, 1.5 * channels_q))
    colours = ["green", "blue", "red", "orange", "black", "purple", "brown", "pink"]
    for i in range(channels_q):
        axes[i].plot(times, data[:, i], color=colours[i])
        axes[i].set_ylabel(channels_names[i])
    axes[channels_q - 1].set_xlabel("Время (с)")
    plt.suptitle("ЭЭГ-сигналы")
    plt.tight_layout()
    plt.show()


def VisualizeGraph(corr_matrix, channels_names, label):
    EEG_graph = nx.Graph()
    for i in range(len(channels_names)):
        EEG_graph.add_node(i)
    for i in range(len(channels_names)):
        for j in range(i + 1, len(channels_names)):
            if corr_matrix[i, j] != 0:
                EEG_graph.add_edge(i, j, weight=corr_matrix[i, j],
                                                    node_names=(channels_names[i], channels_names[j]))

    # drawing graph
    plt.figure(1)
    pos = nx.spring_layout(EEG_graph, k=1.7)
    edges = EEG_graph.edges(data=True)
    nx.draw_networkx_nodes(EEG_graph, pos, node_size=500, node_color='green', edgecolors='black')
    nx.draw_networkx_edges(EEG_graph, pos, edges, edge_color='gray')
    nx.draw_networkx_labels(EEG_graph, pos, {i: channels_names[i] for i in range(len(channels_names))},
                            font_size=12, font_color='black')
    nx.draw_networkx_edge_labels(EEG_graph, pos, {(u, v): f"{d['weight']:.2f}" for u, v, d in edges},
                                 font_color='red')
    plt.title(label)
    plt.show()


def VisualizeFuncConnectome(corr_matrix, channels_names, label):
    plt.figure(2, (5, 4))
    sns.heatmap(corr_matrix, cmap="winter", xticklabels=channels_names, yticklabels=channels_names)
    plt.title(label)
    plt.show()
