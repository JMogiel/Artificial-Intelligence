from sklearn.metrics import accuracy_score
from id3_tree_classifier import ID3TreeClassifier
from data_preprocessing import load_and_prepare_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from graphviz import Digraph


def plot_validation_accuracy_vs_depth(depths, accuracies, setups):
    """
    Plots validation accuracy vs max_depth for ID3TreeClassifier.

    :param depths: A list of max_depth values.
    :param accuracies: A list of validation accuracies corresponding to the max_depth values.
    :param setups: A list of strings indicating the different setups for each accuracy.
    """
    unique_setups = set(setups)
    colors = plt.cm.get_cmap('tab10', len(unique_setups))

    for i, setup in enumerate(unique_setups):
        setup_depths = [d for j, d in enumerate(depths) if setups[j] == setup]
        setup_accuracies = [a for j, a in enumerate(accuracies) if setups[j] == setup]
        plt.plot(setup_depths, setup_accuracies, label=setup, color=colors(i))

    plt.xlabel("Max Depth")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Max Depth for ID3TreeClassifier")
    plt.legend()
    plt.show()


# def plot_confusion_matrix(y_true, y_pred, title):
#     cm = confusion_matrix(y_true, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(title)
#     plt.show()
#
#
# def visualize_tree(tree, node_id=0, parent=None, edge_label=None, graph=None):
#     if graph is None:
#         graph = Digraph(name="ID3TreeClassifier")
#
#     if "leaf" in tree:
#         graph.node(f"{node_id}", label=f"{tree['leaf']}", shape="ellipse", style="filled", fillcolor="lightblue")
#     else:
#         attribute = tree["attribute"]
#         graph.node(f"{node_id}", label=f"{attribute}")
#         if parent is not None:
#             graph.edge(f"{parent}", f"{node_id}", label=f"{edge_label}")
#
#         for key, value in tree.items():
#             if key not in ["attribute", "leaf"]:
#                 new_node_id = len(graph.body)
#                 visualize_tree(value, node_id=new_node_id, parent=node_id, edge_label=key, graph=graph)
#
#     return graph


def main():
    file_path = "cardio_train.csv"
    depths = range(0, 12)
    test_sizes = [0.12, 0.33, 0.51]#, 0.01]
    val_sizes = [0.14, 0.36, 0.57]#, 0.05]
    markers = ["o", "s", "^", "o"]
    colors = ["r", "g", "b", "y"]
    i = 0

    # Calculate the number of plots
    n_plots = sum(1 for test_size in test_sizes for val_size in val_sizes if test_size + val_size < 0.9)

    # Create a new figure for the combined plot
    plt.figure(figsize=(15, 8))

    for test_size in test_sizes:
        for val_size in val_sizes:
            if test_size + val_size >= 0.9:
                continue

            X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(file_path, test_size, val_size)

            val_accuracies = []

            for max_depth in depths:
                clf = ID3TreeClassifier(max_depth=max_depth)
                clf.fit(X_train, y_train)

                y_pred_val = clf.predict(X_val)
                val_accuracy = accuracy_score(y_val, y_pred_val)
                val_accuracies.append(val_accuracy)

                # Print validation accuracy for each set
                print(f"test_size={test_size:.4f}, val_size={val_size:.4f}, max_depth={max_depth}, accuracy={val_accuracy:.4f}")

            # Create subplot
            plt.subplot(2, n_plots // 2, i + 1)  # Adjust the number of rows and columns of subplots
            plt.plot(depths, val_accuracies, color=colors[i % len(colors)], marker=markers[i % len(markers)], label=f"test_size={test_size:.4f}, val_size={val_size:.4f}")
            plt.xlabel("Depth")
            plt.ylabel("Validation Accuracy")
            plt.title(f"Validation Accuracy vs Depth for ID3TreeClassifier\n(test_size={test_size:.4f}, val_size={val_size:.4f})")
            plt.legend()

            # Plot on the combined figure
            plt.figure(1)
            plt.plot(depths, val_accuracies, color=colors[i % len(colors)], marker=markers[i % len(markers)], label=f"test_size={test_size:.4f}, val_size={val_size:.4f}")
            i += 1

    # Show the combined plot
    plt.xlabel("Depth")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Depth for ID3TreeClassifier (Combined)")
    plt.legend()
    plt.show()


main()

