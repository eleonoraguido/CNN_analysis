import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns

def evaluate_model(model, data_test):
    """
    Evaluate the performance of a trained CNN model using test data.

    Parameters:
    -----------
    model : tf.keras.Model
        Trained CNN model.
    data_test : tuple
        Tuple containing test data (traces, distances, event info, azimuth, Stot) and labels.

    Returns:
    --------
    scores : list
        A list containing the evaluation scores (loss value and accuracy) of the model on the test data.
    
    """
    scores = model.evaluate([data_test.traces, data_test.dist, data_test.event, data_test.azimuth, data_test.Stot], data_test.label, verbose=1)
    print("\nEvaluate the model on the test sample:")
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    return scores 


def return_labels(model, data_test):
    """
    Predict labels using a trained model and return both predicted and true labels.

    Parameters:
    -----------
    model : tf.keras.Model
        Trained model for prediction.
    data_test : tuple
        Tuple containing test data (traces, distances, event info, azimuth, Stot) and labels.

    Returns:
    --------
    y_pred : numpy.ndarray
        Predicted labels by the model.
    y_true : numpy.ndarray
        True labels from the test data.

    Example:
    --------
    y_pred, y_true = return_labels(trained_model, test_data)
    """
    y_pred = model.predict([data_test.traces, data_test.dist, data_test.event, data_test.azimuth, data_test.Stot])  
    y_true = np.array(data_test.label)
    return y_pred, y_true


def plot_labels(y_true, y_pred, filename="scoreCNN_test.pdf"):
    """
    Plot the predicted labels distribution of a trained CNN model.

    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels for the test data.
    y_pred : numpy.ndarray
        Predicted labels for the test data.
    filename : str, optional
        Filename to save the plot. Default is 'scoreCNN_test.pdf'.

    Example:
    --------
    plot_labels(trained_model, true_labels, predicted_labels)
    """
    fig, ax = plt.subplots()
    photon = np.array([x for x, y in zip(y_pred, y_true) if y == 1]) # Predicted labels for photons
    proton = np.array([x for x, y in zip(y_pred, y_true) if y == 0]) # Predicted labels for protons
    hist1, bin_edges = np.histogram(photon, bins=10)
    hist2, bin_edges = np.histogram(proton, bins=10)
    ax.hist(proton, 40, [0, 1], color='orange', alpha=0.7, label='Background', ec="darkorange", zorder=2)
    ax.hist(photon, 40, [0, 1], color='green', alpha=0.3, label='Photons', ec="darkgreen", zorder=2)
    ax.vlines(0.5, 1.e0, 1.e5, color='darkblue')
    ax.set_ylabel('Number of events', fontsize='13')
    ax.set_xlabel('$y_{\mathrm{pred}}$', fontsize='13')
    ax.set_title('')
    ax.set_ylim([1.e0, 1.e5])
    ax.set_xlim([0, 1])
    ax.set_yscale('log')
    legend = ax.legend(prop={'size': 13}, edgecolor="black")
    legend.get_frame().set_alpha(0.2)
    ax.grid(zorder=1)
    fig.savefig(filename, transparent=True)
    plt.close(fig)  # Close the figure to release memory




def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrices.pdf"):
    """
    Plot the confusion matrices side by side on the same plot.

    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels for the test data.
    y_pred : numpy.ndarray
        Predicted labels for the test data.
    filename : str, optional
        Filename to save the plot. Default is 'confusion_matrices.pdf'.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    confusion_matrix = metrics.confusion_matrix(y_true, np.rint(y_pred))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Background", "Photon"])
    cm_display.plot(ax=axs[0])
    axs[0].set_title('Confusion matrix')

    confusion_matrix_norm = metrics.confusion_matrix(y_true, np.rint(y_pred), normalize="true")
    cm_display_norm = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_norm, display_labels=["Background", "Photon"])
    cm_display_norm.plot(ax=axs[1])
    axs[1].set_title('Normalized confusion matrix')

    # Remove grid lines
    for ax in axs:
        ax.grid(False)

    # Adjust layout and spacing
    plt.subplots_adjust(wspace=0.5)

    plt.savefig(filename, transparent=True)
    plt.close(fig)  # Close the figure to release memory



def plot_ROC(y_true, y_pred, filename="roc_curve.pdf"):
    """
    Plot the ROC curve.

    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels for the test data.
    y_pred : numpy.ndarray
        Predicted labels for the test data.
    filename : str, optional
        Filename to save the plot. Default is 'roc_curve.pdf'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax1 = axes[0]
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    ax1.set_title('Receiver Operating Characteristic', fontsize=15)
    ax1.plot(fpr, tpr, color='darkblue', label='AUC = %0.4f' % roc_auc)
    ax1.legend(loc='lower right', prop={'size': 12})
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(zorder=0)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_xlabel('False Positive Rate', fontsize=12)

    # Precision-Sensitivity Curve
    ax2 = axes[1]
    prec, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    ax2.set_title('Precision-sensitivity curve', fontsize=15)
    ax2.plot(recall, prec, color='darkblue')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([0.95, 1.01])
    ax2.grid(zorder=0)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_xlabel('Sensitivity', fontsize=12)

    # Save the figure
    fig.savefig(filename, transparent=True)
    plt.close(fig)  # Close the figure to release memory.

    return tpr, threshold


def plot_confusion_matrix_50sigeff(y_true, y_pred, tpr, threshold, prob =0.5, filename="confusion_matrix_50sigeff.pdf"):
    """
    Plot confusion matrix with annotated values at 50% signal efficiency.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    tpr : array-like of shape (n_thresholds,)
        True Positive Rate values corresponding to different thresholds.
    threshold : array-like of shape (n_thresholds,)
        Threshold values corresponding to different True Positive Rates.
    prob : float, optional
        True Positive Rate value for which the threshold is calculated. Default is 0.5.
    filename : str, optional
        Filename to save the plot. Default is "confusion_matrix_50sigeff.pdf".

    Returns:
    --------
    None
    """
    # Find threshold for a given true positive rate
    thres = threshold[np.argmin(np.abs(tpr - prob))]
    
    # Calculate confusion matrix values at 50% signal efficiency
    y_pred_sel = np.where(y_pred > thres, 1, 0)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_sel, normalize='true').ravel()
    fp_50 = fp * 100
    
    # Calculate background rejection
    background_rejection = 100 - fp_50
    
    # Print information
    print("Threshold for 50% signal efficiency:", thres)
    print("False Positive Rate with a True Positive Rate of 0.5:", f"{fp_50:.2f}%")
    print("Background Rejection with a True Positive Rate of 0.5:", f"{background_rejection:.2f}%")
    
    tn = tn*100
    fp = fp*100
    fn = fn*100
    tp = tp*100
    # Plot confusion matrix
    cf_matrix = np.array([[tn, fp], [fn, tp]])
    group_names = [[f"{tn:.2f}%", f"{fp:.2f}%"], [f"{fn:.0f}%", f"{tp:.0f}%"]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_matrix, annot=group_names, fmt='', cmap='viridis', linewidths=0.5, linecolor='black', cbar=False, annot_kws={"size": 14})
    plt.xlabel('Predicted label', fontsize=15)
    plt.ylabel('True label', fontsize=15)
    plt.xticks(ticks=[0.5, 1.5], labels=['Background', 'Photons'], fontsize=13)
    plt.yticks(ticks=[0.5, 1.5], labels=['Background', 'Photons'], fontsize=13)
    plt.title('Confusion Matrix', fontsize=15)
    
    plt.tight_layout()
    plt.savefig(filename)


