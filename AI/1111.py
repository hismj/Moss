import matplotlib.pyplot as plt
import numpy as np

# Confusion matrix data (replace with your actual confusion matrices)
conf_matrix_val = np.array([[35, 4],
                            [6, 47]])

conf_matrix_test = np.array([[32, 12],
                             [6, 43]])

# Define function to plot confusion matrix
def plot_confusion_matrix(matrix, title):
    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    # Add labels
    classes = ['Benign', 'Scam']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add numerical values within each cell
    thresh = matrix.max() / 2.
    for i, j in np.ndindex(matrix.shape):
        plt.text(j, i, format(matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Plotting confusion matrices
plot_confusion_matrix(conf_matrix_val, title='Validation Confusion Matrix')
plt.show()

plot_confusion_matrix(conf_matrix_test, title='Test Confusion Matrix')
plt.show()
