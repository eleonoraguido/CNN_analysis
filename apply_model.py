import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns
import collections
import utils
from config import PDF_SAVE_PATH

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
    y_pred = np.squeeze(y_pred)
    return y_pred, y_true


def plot_labels(y_true, y_pred, filename="scoreCNN_test.pdf", save_path=PDF_SAVE_PATH):
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
    save_path : str, optional
        Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.
    Example:
    --------
    plot_labels(trained_model, true_labels, predicted_labels)
    """
    # Generate PDF and save it to the specified path
    full_path = os.path.join(save_path, filename)

    fig, ax = plt.subplots()
    photon = np.array([x for x, y in zip(y_pred, y_true) if y == 1]) # Predicted labels for photons
    proton = np.array([x for x, y in zip(y_pred, y_true) if y == 0]) # Predicted labels for protons
    hist1, bin_edges = np.histogram(photon, bins=10)
    hist2, bin_edges = np.histogram(proton, bins=10)
    ax.hist(proton, 50, [0, 1], color='orange', alpha=0.7, label='Background', ec="darkorange", zorder=2)
    ax.hist(photon, 50, [0, 1], color='green', alpha=0.3, label='Photons', ec="darkgreen", zorder=2)
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
    fig.savefig(full_path, transparent=True)
    plt.close(fig)  # Close the figure to release memory




def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrices.pdf", save_path=PDF_SAVE_PATH):
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
    save_path : str, optional
        Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.
    """

    # Generate PDF and save it to the specified path
    full_path = os.path.join(save_path, filename)

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

    plt.savefig(full_path, transparent=True)
    plt.close(fig)  # Close the figure to release memory



def plot_ROC(y_true, y_pred, filename="roc_curve.pdf", save_path=PDF_SAVE_PATH):
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
    save_path : str, optional
        Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.
    """
    # Generate PDF and save it to the specified path
    full_path = os.path.join(save_path, filename)

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
    fig.savefig(full_path, transparent=True)
    plt.close(fig)  # Close the figure to release memory.

    return tpr, threshold


def plot_confusion_matrix_50sigeff(y_true, y_pred, tpr, threshold, prob =0.5, filename="confusion_matrix_50sigeff.pdf", save_path=PDF_SAVE_PATH):
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
    save_path : str, optional
        Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.

    Returns:
    --------
    thres: float
        Threshold corresponding to 50% signal efficiency
    """
    # Generate PDF and save it to the specified path
    full_path = os.path.join(save_path, filename)

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
    plt.savefig(full_path)

    return thres


def get_background_rejection (y_true, y_pred, threshold):
    """
    Compute background rejection rate based on true and predicted labels.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        threshold (float): Threshold value for converting probabilities to binary predictions.

    Returns:
        misclassified_indices (numpy.ndarray): Index positions of misclassified background events.

    """
    # Calculate confusion matrix values at 50% signal efficiency
    y_pred_sel = np.where(y_pred > threshold, 1, 0)
    
    # Calculate True Negatives (TN) and False Positives (FP)
    tn_indices = np.where((y_true == 0) & (y_pred_sel == 0))[0]  # Index positions of true negatives
    fp_indices = np.where((y_true == 0) & (y_pred_sel == 1))[0]  # Index positions of false positives
    
    # Calculate background rejection
    tn = len(tn_indices)
    fp = len(fp_indices)
    background_rejection = 100 * tn / (tn + fp)
    
    # Print information
    print("The decision threshold has been set to ", threshold)
    print("The corresponding background rejection is ", f"{background_rejection:.2f}%")
    print("Number of background events misclassified (False Positives):", fp, " out of ", tn+fp)

    return fp_indices
    


def print_events_info(indices, test_data, output_file = 'selected_events.pdf', save_path=PDF_SAVE_PATH):
    """
    Print information for selected events based on their indices and loaded data.
    Plot the values of S1000, theta and Nstat for such events, comapred with the whole distributions.

    Parameters:
    -----------
    indices : numpy.ndarray
        Array containing the indices of events.
    test_data : DataSets
        Object containing the test dataset.
    output_file : str, optional
        File path to save the output plot (default is 'selected_events.pdf').
    save_path : str, optional
        Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.

    Returns:
    -----------
    None
    """
    # Generate PDF and save it to the specified path
    full_path = os.path.join(save_path, output_file)

    # Print table header
    print("Selected events:")
    print("{:<10}\t {:<10}\t {:<10}".format('S(1000)', 'Theta', 'Nstat'))
    print("-" * 70)

    # Lists to store values for histograms
    s1000_all_values = []
    theta_all_values = []
    nstat_all_values = []

    s1000_selected_values = []
    theta_selected_values = []
    nstat_selected_values = []
    
    # Iterate over misclassified indices
    for idx in range(len(test_data.event)):
        #dist = test_data.dist[idx]
        #stot = test_data.Stot[idx]
        #azimuth = test_data.azimuth[idx]
        info_event = test_data.event[idx]
        #label = test_data.label[idx]

        # Extract values from the list and convert them to strings
        s1000 = 10**info_event[1][0]
        theta = info_event[0][0]
        nstat = 10**info_event[2][0]
        
        # Append values to lists for histograms
        s1000_all_values.append(s1000)
        theta_all_values.append(theta)
        nstat_all_values.append(nstat)

        # Check if the index is in the given indices
        if idx in indices:
            s1000_selected_values.append(s1000)
            theta_selected_values.append(theta)
            nstat_selected_values.append(nstat)
        
            # Print information for each misclassified event
            print("{:.2f}\t {:.2f}\t {:<10}".format(s1000, theta, nstat))

    #--------------------------------------------------------------------
    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.rcParams.update({'text.usetex': True})

    # Histogram 1
    axes[0].hist(s1000_all_values, bins=30, color='blue', alpha=0.7)
    axes[0].set_title('S(1000) Distribution')
    axes[0].set_xlabel('S(1000)')
    axes[0].set_ylabel('Frequency')
    # Add vertical lines for selected values
    value_counts = collections.Counter(s1000_selected_values)
    for value, count in value_counts.items():
        axes[0].axvline(x=value, color='darkblue', linestyle='--')
        if count > 1:
            axes[0].text(value, axes[0].get_ylim()[1]*0.95, f'{count}', color='darkblue', ha='left', rotation=0)


    # Histogram 2 
    axes[1].hist(theta_all_values, bins=30, color='green', alpha=0.7)
    axes[1].set_title(r'$ \theta $ Distribution')
    axes[1].set_xlabel(r'$ \theta $')
    axes[1].set_ylabel('Frequency')

    # Add vertical lines for selected values
    value_counts = collections.Counter(theta_selected_values)
    for value, count in value_counts.items():
        axes[1].axvline(x=value, color='darkgreen', linestyle='--')
        if count > 1:
            axes[1].text(value, axes[1].get_ylim()[1]*0.95, f'{count}', color='darkblue', ha='left', rotation=0)


    # Histogram 3
    # Calculate the bin edges
    bin_edges = np.arange(1, 21, 1)

    # Plot histogram with custom bins
    _, bins, _ = axes[2].hist(nstat_all_values, bins=bin_edges, color='red', alpha=0.7)
    axes[2].set_title('$N_{stat}$ Distribution')
    axes[2].set_xlabel('$N_{stat}$')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xticks(np.arange(min(bins), max(bins)+1, 1))

    # Add vertical lines for selected values
    value_counts = collections.Counter(nstat_selected_values)
    for value, count in value_counts.items():
        axes[2].axvline(x=value, color='darkred', linestyle='--')
        if count > 1:
            axes[2].text(value+0.1, axes[2].get_ylim()[1]*0.95, f'{count}', color='black', ha='left', rotation=0)

    plt.tight_layout()
    plt.savefig(full_path)




def plot_fpr_vs_S1000(test_data, y_pred, y_test, num_bins=5, pdf_name="fpr_vs_S1000_binning_equal.pdf", save_path=PDF_SAVE_PATH):
    """
    Plot the false positive rate (FPR) versus S1000/VEM with equal binning.

    Parameters:
        test_data (array-like): Array containing the test data.
        y_pred (array-like): Predicted labels.
        y_test (array-like): True labels.
        num_bins (int, optional): Number of bins for binning S1000/VEM. Default is 5.
        pdf_name (str, optional): Name of the PDF file to save the plot. Default is "fpr_vs_S1000_binning_equal.pdf".
        save_path (str, optional): Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.

    Returns:
        None
    """
    # Generate PDF and save it to the specified path
    full_path = os.path.join(save_path, pdf_name)

    # Iterate over events
    s1000 = np.array([test_data.event[idx][1][0] for idx in range(len(test_data.event))])

    # Compute the bin edges with an external function (same number of events in each bin)
    x_edges_equal = utils.compute_bin_edges(s1000, num_bins)

    min_S1000 = s1000.min()
    max_S1000 = s1000.max()
    width = (max_S1000 - min_S1000) / num_bins

    # Compute the bin edges
    x_edges = np.arange(start=min_S1000, stop=max_S1000 + 0.05, step=width)

    fpr_arr = np.empty(x_edges.size-1) 
    bkg_arr = np.empty(x_edges.size-1) 

    fp_sum = 0

    for i, x in enumerate(x_edges_equal[:-1]):
        lab_true = np.array([lab for Sref, lab in zip(s1000, y_test) if Sref > x and Sref < x_edges_equal[i+1]])
        lab_pred = np.array([lab for Sref, lab in zip(s1000, y_pred) if Sref > x and Sref < x_edges_equal[i+1]])
        fp = np.logical_and(lab_pred >= 0.5, lab_true == False)
        if(lab_true.size == 0):
            fpr = 0 
        else:
            fpr = fp.sum() / (lab_true.size - np.count_nonzero(lab_true))
        fpr_arr[i] = fpr
        bkg_arr[i] = lab_true.size - np.count_nonzero(lab_true)
        
    x_axis = np.round(x_edges, 2)
    x_axis_equal = np.round(x_edges_equal, 2)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

    fig, ax = plt.subplots()
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_axis_equal)
    ax.grid(zorder=3)
    ax.hist(x_axis[:-1], x_axis, weights=fpr_arr, rwidth = 0.95, color = "lightblue", ec="blue", zorder=4)
    ax.set_ylabel('FPR', fontsize='13')
    ax.set_xlabel('$\mathrm{log_{10}(S1000/VEM)}$', fontsize='13')

    rects = ax.patches
    for rect, bin_value in zip(rects, bkg_arr):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, f"{bin_value:.0f}\n{height*100:.4f}\%", ha='center', va='bottom')
    ax.tick_params(axis='y', which='minor')
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    s1000_values = 10 ** x_axis_equal
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{val:.1f}" for val in s1000_values])
    ax2.set_xlabel('$S1000 (VEM)$', fontsize='13')

    fig.savefig(full_path)


def plot_tpr_vs_S1000(test_data, y_pred, y_test, num_bins=5, pdf_name="tpr_vs_S1000_binning_equal.pdf", save_path=PDF_SAVE_PATH):
    """
    Plot the True Positive Rate (TPR) versus S1000/VEM with equal binning.

    Parameters:
        test_data (array-like): Array containing the test data.
        y_pred (array-like): Predicted labels.
        y_test (array-like): True labels.
        num_bins (int, optional): Number of bins for binning S1000/VEM. Default is 5.
        pdf_name (str, optional): Name of the PDF file to save the plot. Default is "tpr_vs_S1000_binning_equal.pdf".
        save_path (str, optional): Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.
        
    Returns:
        None
    """
    # Generate PDF and save it to the specified path
    full_path = os.path.join(save_path, pdf_name)

    plt.rcParams.update({'text.usetex': True})
    # Iterate over events
    s1000 = np.array([test_data.event[idx][1][0] for idx in range(len(test_data.event))])

    # Compute the bin edges
    x_edges_equal = utils.compute_bin_edges(s1000, num_bins)

    min_S1000 = s1000.min()
    max_S1000 = s1000.max()
    width = (max_S1000 - min_S1000) / num_bins

    # Compute the bin edges
    x_edges = np.arange(start=min_S1000, stop=max_S1000 + 0.05, step=width)

    tpr_arr = np.empty(x_edges.size - 1) 
    signal_count_arr = np.empty(x_edges.size - 1) 

    for i, x in enumerate(x_edges_equal[:-1]):
        lab_true = np.array([lab for Sref, lab in zip(s1000, y_test) if Sref > x and Sref < x_edges_equal[i + 1]])
        lab_pred = np.array([lab for Sref, lab in zip(s1000, y_pred) if Sref > x and Sref < x_edges_equal[i + 1]])
        tp = np.logical_and(lab_pred >= 0.5, lab_true == True)
        tpr = tp.sum() / np.count_nonzero(lab_true)
        tpr_arr[i] = tpr
        signal_count_arr[i] = np.count_nonzero(lab_true)
        
    x_axis = np.round(x_edges, 2)
    x_axis_equal = np.round(x_edges_equal, 2)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

    fig, ax = plt.subplots()
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_axis_equal)
    ax.grid(zorder=3)
    ax.hist(x_axis[:-1], x_axis, weights=tpr_arr, rwidth=0.95, color="lightblue", ec="blue", zorder=4)
    ax.set_ylabel('TPR', fontsize='13')
    ax.set_xlabel('$\mathrm{log_{10}(S1000/VEM)}$', fontsize='13')

    rects = ax.patches
    for rect, bin_value in zip(rects, signal_count_arr):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, f"{bin_value:.0f}\n{height*100:.4f}\%", ha='center', va='bottom')
    ax.tick_params(axis='y', which='minor')
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    s1000_values = 10 ** x_axis_equal
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{val:.1f}" for val in s1000_values])
    ax2.set_xlabel('$S1000 (VEM)$', fontsize='13')

    fig.savefig(full_path)

