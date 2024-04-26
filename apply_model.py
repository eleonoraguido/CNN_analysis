import os, sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn import metrics
import seaborn as sns
import collections
import utils
from config import PDF_SAVE_PATH


def evaluate_model(models, datasets):
    """
    Evaluate the performance of each model in the dictionary using the corresponding test data.

    Parameters:
    -----------
    - models : dict
        Dictionary containing trained models.
    - datasets : dict
        Dictionary containing datasets used for testing, training, and validation.

    Returns:
    --------
    scores_dict : dict
        Dictionary containing evaluation scores (loss value and accuracy) of each model.
    """
    scores_dict = {}

    for (data_key, dataset), (model_key, trained_model) in zip(datasets.items(), models.items()):
        # Check if the keys match
        if data_key != model_key:
            print("Error: Keys in datasets and models dictionaries don't match.")
            sys.exit(1)

        test_dataset = dataset['test']
        scores = trained_model.evaluate([test_dataset.traces, test_dataset.dist, test_dataset.event, test_dataset.azimuth, test_dataset.Stot], test_dataset.label, verbose=1)
        print("Evaluate the model on the test sample:")
        print("%s: %.2f%% \n" % (trained_model.metrics_names[1], scores[1]*100))
        scores_dict[model_key] = scores

    return scores_dict


def return_labels(models, datasets):
    """
    Predict labels using trained models and return both predicted and true labels.

    Parameters:
    -----------
    models : dict
        Dictionary containing model names as keys and trained models as values.
    datasets : dict
        Dictionary containing dataset names as keys and test data as values.

    Returns:
    --------
    results : dict
        Dictionary containing predicted and true labels for each model and dataset pair.

    Example:
    --------
    results = return_labels(trained_models_dict, test_datasets_dict)
    """
    results = {}

    for (data_key, dataset), (model_key, trained_model) in zip(datasets.items(), models.items()):
        # Check if the keys match
        if data_key != model_key:
            print("Error: Keys in datasets and models dictionaries don't match.")
            sys.exit(1)

        test_dataset = dataset['test']

        y_pred = trained_model.predict([test_dataset.traces, test_dataset.dist, test_dataset.event, test_dataset.azimuth, test_dataset.Stot])  
        y_true = np.array(test_dataset.label)
        y_pred = np.squeeze(y_pred)

        results[model_key] = {'y_pred': y_pred, 'y_true': y_true}

    return results




def plot_labels(results, filename="scoreCNN_test.pdf", save_path=PDF_SAVE_PATH):
    """
    Plot the predicted labels distribution of one or multiple trained CNN models.

    Parameters:
    -----------
    results : dict
        Dictionary containing predicted and true labels for each model.
    filename : str, optional
        Filename to save the plot. Default is 'scoreCNN_test.pdf'.
    save_path : str, optional
        Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.
    """

    full_path = os.path.join(save_path, filename)
    fig, ax = plt.subplots()

    if len(results) == 1:
        data = next(iter(results.values()))
        y_true = data['y_true']
        y_pred = data['y_pred']
        photon = np.array([x for x, y in zip(y_pred, y_true) if y == 1]) # Predicted labels for photons
        proton = np.array([x for x, y in zip(y_pred, y_true) if y == 0]) # Predicted labels for protons
        _, bin_edges = np.histogram(photon, bins=10)
        _, bin_edges = np.histogram(proton, bins=10)
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

    else:
        bins = 50
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        mean_photon = np.zeros(bins)
        mean_proton = np.zeros(bins)
        std_photon = np.zeros(bins)
        std_proton = np.zeros(bins)

        num_models = len(results)

        for data in results.values():
            y_true = data['y_true']
            y_pred = data['y_pred']
            photon = np.array([x for x, y in zip(y_pred, y_true) if y == 1])
            proton = np.array([x for x, y in zip(y_pred, y_true) if y == 0])

            photon_hist, _ = np.histogram(photon, bins=bin_edges)
            proton_hist, _ = np.histogram(proton, bins=bin_edges)

            mean_photon += photon_hist
            mean_proton += proton_hist

        mean_photon /= num_models
        mean_proton /= num_models


        for data in results.values():
            y_true = data['y_true']
            y_pred = data['y_pred']
            photon = np.array([x for x, y in zip(y_pred, y_true) if y == 1])
            proton = np.array([x for x, y in zip(y_pred, y_true) if y == 0])

            photon_hist, _ = np.histogram(photon, bins=bin_edges)
            proton_hist, _ = np.histogram(proton, bins=bin_edges)

            std_photon += (photon_hist - mean_photon) ** 2
            std_proton += (proton_hist - mean_proton) ** 2

        std_photon = np.sqrt(std_photon / (num_models - 1))
        std_proton = np.sqrt(std_proton / (num_models - 1))
        error_on_mean_photon = std_photon / np.sqrt(num_models)
        error_on_mean_proton = std_proton / np.sqrt(num_models)

        # Plot mean values with errors on the mean
        ax.errorbar(bin_centers, mean_proton, yerr=error_on_mean_proton, fmt='o', color='orange', label='Background', markersize=5, capsize=4)
        ax.errorbar(bin_centers, mean_photon, yerr=error_on_mean_photon, fmt='o', color='green', label='Photons', markersize=5, capsize=4)
        ax.vlines(0.5, 1.e0, 1.e5, color='darkblue')
        ax.set_ylabel('Number of events', fontsize='13')
        ax.set_xlabel('$y_{\mathrm{pred}}$', fontsize='13')
        ax.set_title('Mean values of number of predictions')
        ax.set_ylim([1.e0, 1.e5])
        ax.set_xlim([0, 1])
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='both', direction='in', width=1, length=4)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        legend = ax.legend(prop={'size': 13}, edgecolor="black")
        legend.get_frame().set_alpha(0.2)
        
    fig.savefig(full_path, transparent=True)
    plt.close(fig)  # Close the figure to release memory




def conf_matrix_plot(cf_matrix_mean, group_names, ax):
    """
    Plot confusion matrix.

    Parameters:
    -----------
    - cf_matrix_mean : numpy array
        Values of the confusion matrix. Format: [[tn, fp], [fn, tp]]
    - group_names : list of lists
        Labels for confusion matrix cells.
    - ax : matplotlib Axes object
        Axes to plot the confusion matrix.
    """
    sns.heatmap(cf_matrix_mean, annot=group_names, fmt='', cmap='viridis', linewidths=0.5,
                linecolor='black', clip_on=False, annot_kws={"size": 14}, ax=ax)
    ax.set_xlabel('Predicted label', fontsize='15')
    ax.set_ylabel('True label', fontsize='15')



def plot_confusion_matrix(results, filename="confusion_matrices.pdf", save_path=PDF_SAVE_PATH):
    """
    Plot the confusion matrices side by side on the same plot.

    Parameters:
    -----------
    results : dict
        Dictionary containing predicted and true labels for each model and dataset pair.
    filename : str, optional
        Filename to save the plot. Default is 'confusion_matrices.pdf'.
    save_path : str, optional
        Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.
    """
    
    # Generate PDF and save it to the specified path
    full_path = os.path.join(save_path, filename)

    # If there's only one model, plot its confusion matrix directly
    if len(results) == 1:
        data = next(iter(results.values()))
        y_true = data['y_true']
        y_pred = data['y_pred']


        confusion_matrix = metrics.confusion_matrix(y_true, np.rint(y_pred))
        confusion_matrix_norm = metrics.confusion_matrix(y_true, np.rint(y_pred), normalize="true")

        tn, fp, fn, tp = confusion_matrix.ravel()
        tnr, fpr, fnr, tpr = confusion_matrix_norm.ravel()

        tnr = tnr * 100
        fpr = fpr * 100
        fnr = fnr * 100
        tpr = tpr * 100

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        cf_matrix = np.array([[tn, fp], [fn, tp]])
        cf_matrix_norm = np.array([[tnr, fpr], [fnr, tpr]])
        group_names_norm = [[f"{tnr:.2f}%", f"{fpr:.2f}%"], [f"{fnr:.2f}%", f"{tpr:.2f}%"]]
        group_names = [[f"{tn:.0f}", f"{fp:.0f}"], [f"{fn:.0f}", f"{tp:.0f}"]]
            
        conf_matrix_plot(cf_matrix, group_names, axs[1])
        axs[1].set_xticklabels(['Background', 'Photon'], fontsize='13')
        axs[1].set_yticklabels(['Background', 'Photon'], fontsize='13')
        cbar = axs[1].collections[0].colorbar
        cbar.ax.tick_params(labelsize=13)
        axs[1].set_title('Confusion Matrix', fontsize=15)

        conf_matrix_plot(cf_matrix_norm, group_names_norm, axs[0])
        axs[0].set_xticklabels(['Background', 'Photon'], fontsize='13')
        axs[0].set_yticklabels(['Background', 'Photon'], fontsize='13')
        cbar = axs[0].collections[0].colorbar
        cbar.ax.tick_params(labelsize=13)
        axs[0].set_title('Normalized Confusion Matrix', fontsize=15)
            
        # Remove grid lines
        for ax in axs:
            ax.grid(False)

        plt.tight_layout()
        fig.savefig(full_path, transparent=True)
        plt.close(fig)
         

    else:
        # Initialize lists to store confusion matrix values for all models
        tn_list, fp_list, fn_list, tp_list = [], [], [], []

        # Iterate through results to collect confusion matrix values for each model
        for _, data in results.items():
            y_true = data['y_true']
            y_pred = data['y_pred']

            confusion_matrix = metrics.confusion_matrix(y_true, np.rint(y_pred), normalize="true")

            tn, fp, fn, tp = confusion_matrix.ravel()
            tn_list.append(tn*100)
            fp_list.append(fp*100)
            fn_list.append(fn*100)
            tp_list.append(tp*100)

        # Compute mean and standard error of the mean for each entry in the confusion matrix
        n_models = len(results)
        cf_matrix_mean = np.array([[np.mean(tn_list), np.mean(fp_list)],
                                   [np.mean(fn_list), np.mean(tp_list)]])

        # Create group names with mean and standard error of the mean
        str_tp = "({0:.2f} ± {1:.2f})%".format(np.mean(tp_list), np.std(tp_list) / np.sqrt(n_models))
        str_fn = "({0:.2f} ± {1:.2f})%".format(np.mean(fn_list), np.std(fn_list) / np.sqrt(n_models))
        str_fp = "({0:.2f} ± {1:.2f})%".format(np.mean(fp_list), np.std(fp_list) / np.sqrt(n_models))
        str_tn = "({0:.2f} ± {1:.2f})%".format(np.mean(tn_list), np.std(tn_list) / np.sqrt(n_models))

        group_names = [[str_tn, str_fp], [str_fn, str_tp]]

        # Plot confusion matrix with error bars
        fig, ax = plt.subplots()
        conf_matrix_plot(cf_matrix_mean, group_names, ax)
        ax.set_xticklabels(['Background', 'Photon'], fontsize='13')
        ax.set_yticklabels(['Background', 'Photon'], fontsize='13')
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=13)

        plt.tight_layout()
        fig.savefig(full_path, transparent=True)
        plt.close(fig)
        



def plot_ROC(results, filename="roc_curve.pdf", save_path=PDF_SAVE_PATH):
    """
    Plot the ROC curve and the Precision-Sensitivity curve.

    Parameters:
    -----------
    results : dict
        Dictionary containing model names as keys and their corresponding results as values.
        Each value should be a dictionary containing 'y_true' and 'y_pred' keys.
    filename : str, optional
        Filename to save the plot. Default is 'roc_curve.pdf'.
    save_path : str, optional
        Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.
        
    Returns:
    --------
     tuple of two dictionaries:
        - First dictionary (roc_results) contains ROC curve information:
            - Keys: Model names
            - Values: Dictionary containing the following keys:
                - 'tpr': List of true positive rates (TPR)
                - 'threshold': List of threshold values
                - 'auc': Area under the ROC curve (AUC)
            - Accessing example: roc_results[model_name]['tpr']

        - Second dictionary (precision_recall_results) contains Precision-Sensitivity curve information:
            - Keys: Model names
            - Values: Dictionary containing the following keys:
                - 'precision': List of precision values
                - 'recall': List of recall values
            - Accessing example: precision_recall_results[model_name]['precision']
    """
    # Generate PDF and save it to the specified path
    full_path = os.path.join(save_path, filename)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    roc_results = {}
    precision_recall_results = {}

    for model_name, data in results.items():
        y_true = data['y_true']
        y_pred = data['y_pred']

        label_name = model_name.replace("_", " ")

        # ROC Curve
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        roc_results[model_name] = {
            'tpr': tpr.tolist(),
            'threshold': threshold.tolist(),
            'auc': roc_auc
        }

        # Precision-Sensitivity Curve
        prec, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
        precision_recall_results[model_name] = {
            'precision': prec.tolist(),
            'recall': recall.tolist()
        }

        # Plot ROC Curve
        ax1 = axes[0]
        ax1.set_title('Receiver Operating Characteristic', fontsize=15)
        ax1.plot(fpr, tpr, label=f'{label_name} (AUC = %0.4f)' % roc_auc)

        # Plot Precision-Sensitivity Curve
        ax2 = axes[1]
        ax2.set_title('Precision-sensitivity', fontsize=15)
        ax2.plot(recall, prec, label=label_name)

    if len(results) > 1:
        ax1.legend(loc='lower right', prop={'size': 12})
        ax2.legend(loc='lower left', prop={'size': 12})
    else:
        ax1.text(0.5, 0.5, f'AUC = %0.4f' % roc_auc, fontsize=12, horizontalalignment='center', verticalalignment='center')

    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(zorder=0)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_xlabel('False Positive Rate', fontsize=12)

    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([0.95, 1.01])
    ax2.grid(zorder=0)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_xlabel('Sensitivity', fontsize=12)

    # Save the figure
    fig.savefig(full_path, transparent=True)
    plt.close(fig)  # Close the figure to release memory.

    # Return results as a tuple of dictionaries
    return roc_results, precision_recall_results







def plot_confusion_matrix_50sigeff(results, roc_results, prob =0.5, filename="confusion_matrix_50sigeff.pdf", save_path=PDF_SAVE_PATH):
    """
    Plot confusion matrix with annotated values at 50% signal efficiency.

    This function computes the confusion matrix at a 50% signal efficiency threshold for each model 
    based on the provided true positive rate (TPR) values from ROC curve analysis. It then plots the 
    confusion matrices with annotated values.

    Parameters:
    -----------
    results : dict
        Dictionary containing predicted and true labels for each model and dataset pair.
    roc_results : dict
        Dictionary containing ROC curve information (TPR and threshold) for each model.
    prob : float, optional
        True Positive Rate value for which the threshold is calculated. Default is 0.5.
    filename : str, optional
        Filename to save the plot. Default is "confusion_matrix_50sigeff.pdf".
    save_path : str, optional
        Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.

    Returns:
    --------
    thresholds : dict
        Dictionary containing the selected threshold for each model.

    Saves:
    ------
    PDF file containing the plot of confusion matrices with annotated values at 50% signal efficiency.
    """

    # Generate PDF and save it to the specified path
    full_path = os.path.join(save_path, filename)

    # Initialize lists to store confusion matrix values for all models
    tnr_list, fpr_list, fnr_list, tpr_list = [], [], [], []

    thresholds = {}  # Dictionary to store thresholds for each model

    for model_name, data in results.items():
        y_true = data['y_true']
        y_pred = data['y_pred']

        roc_info = roc_results[model_name]
        tpr = np.array(roc_info['tpr'])
        threshold = np.array(roc_info['threshold'])

        # Find threshold for a given true positive rate
        thres = threshold[np.argmin(np.abs(tpr - prob))]

        # Store threshold for the model
        thresholds[model_name] = thres

        # Calculate confusion matrix values at 50% signal efficiency
        y_pred_sel = np.where(y_pred > thres, 1, 0)
        tnr, fpr, fnr, tpr  = metrics.confusion_matrix(y_true, y_pred_sel, normalize='true').ravel()
        tn, fp, fn, tp  = metrics.confusion_matrix(y_true, y_pred_sel).ravel()
        tnr = tnr*100
        fpr = fpr*100
        fnr = fnr*100
        tpr = tpr*100
        
        # Calculate background rejection
        background_rejection = 100 - fpr
    
        print("\nModel: ", model_name)
        print("Threshold for 50% signal efficiency:", thres)
        print("False Positive Rate with a True Positive Rate of 0.5:", f"{fpr:.2f}%")
        print("Background Rejection with a True Positive Rate of 0.5:", f"{background_rejection:.2f}%")

        # If there's only one model, plot its confusion matrix directly
        if len(results) == 1:
            # Plot confusion matrix
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            cf_matrix = np.array([[tn, fp], [fn, tp]])
            cf_matrix_norm = np.array([[tnr, fpr], [fnr, tpr]])
            group_names_norm = [[f"{tnr:.2f}%", f"{fpr:.2f}%"], ["50%", "50%"]]
            group_names = [[f"{tn:.0f}", f"{fp:.0f}"], [f"{fn:.0f}", f"{tp:.0f}"]]
            

            conf_matrix_plot(cf_matrix, group_names, axs[1])
            axs[1].set_xticklabels(['Background', 'Photon'], fontsize='13')
            axs[1].set_yticklabels(['Background', 'Photon'], fontsize='13')
            cbar = axs[1].collections[0].colorbar
            cbar.ax.tick_params(labelsize=13)
            axs[1].set_title('Confusion Matrix', fontsize=15)

            conf_matrix_plot(cf_matrix_norm, group_names_norm, axs[0])
            axs[0].set_xticklabels(['Background', 'Photon'], fontsize='13')
            axs[0].set_yticklabels(['Background', 'Photon'], fontsize='13')
            cbar = axs[0].collections[0].colorbar
            cbar.ax.tick_params(labelsize=13)
            axs[0].set_title('Normalized Confusion Matrix', fontsize=15)
            
             # Remove grid lines
            for ax in axs:
                ax.grid(False)

            plt.tight_layout()
            fig.savefig(full_path)
            plt.close(fig)  # Close the figure to release memory

        else:
            tnr_list.append(tnr)
            fpr_list.append(fpr)
            fnr_list.append(fnr)
            tpr_list.append(tpr)

    if len(results) > 1:
        # Compute mean and standard error of the mean for each entry in the confusion matrix
        n_models = len(results)
        cf_matrix_mean = np.array([[np.mean(tnr_list), np.mean(fpr_list)],
                                   [np.mean(fnr_list), np.mean(tpr_list)]])

        # Create group names with mean and standard error of the mean
        str_tp = "50%"
        str_fn = "50%"
        str_fp = "({0:.2f} ± {1:.2f})%".format(np.mean(fpr_list), np.std(fpr_list) / np.sqrt(n_models))
        str_tn = "({0:.2f} ± {1:.2f})%".format(np.mean(tnr_list), np.std(tnr_list) / np.sqrt(n_models))

        group_names = [[str_tn, str_fp], [str_fn, str_tp]]

        # Plot confusion matrix with error bars
        fig, ax = plt.subplots()
        conf_matrix_plot(cf_matrix_mean, group_names, ax)
        ax.set_xticklabels(['Background', 'Photon'], fontsize='13')
        ax.set_yticklabels(['Background', 'Photon'], fontsize='13')
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=13)

        plt.tight_layout()
        fig.savefig(full_path)
        plt.close(fig)  # Close the figure to release memory

    return thresholds




def get_background_rejection (results, thresholds):
    """
    Compute background rejection rate based on true and predicted labels.

    Parameters:
    -----------
    results : dict
        Dictionary containing predicted and true labels for each model and dataset pair.
    threshold : float
        Threshold value for converting probabilities to binary predictions.

    Returns:
    --------
    numpy.ndarray or list:
        Index positions of misclassified background events for a single model.
        An empty list if the function is not supported for multiple models.
    """

    if len(results) == 1:

        fp_indices = []
        for (model_name, data), (model_name, threshold) in zip(results.items(), thresholds.items()):
            y_pred = data['y_pred']
            y_true = data['y_true']
            
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

    else:
        print("This function is not supported for multiple models.")
        return []
    

def print_events_info(indices, datasets, output_file='selected_events.pdf', save_path=PDF_SAVE_PATH):
    """
    Print information for selected events based on their indices and loaded data.
    Plot the values of S1000, theta and Nstat for such events, compared with the whole distributions.

    Parameters:
    -----------
    indices : numpy.ndarray
        Array containing the indices of events.
    datasets : dict
        Dictionary containing datasets.
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
    new_path = full_path.replace("events", "events_traces")

    if len(datasets) > 1:
        print("Function 'print_events_info' not supported for multiple models.")
        return
    
    test_data = None
    # Retrieve the test data set for the first model
    for data_key, dataset in datasets.items():
        test_data = dataset['test']
        break

    traces = test_data.traces
    # Define the color for the highlighted traces
    highlight_color = 'darkred'
    # Determine the number of stations
    num_stations = traces.shape[1]

    # Create subplots for each station
    fig, axs = plt.subplots(2, num_stations, figsize=(5*num_stations, 10))

    # Iterate over each station
    for station_idx in range(num_stations):
        # Iterate over each event for the upper plots
        for event_idx in range(len(traces)):
            # Get the traces for the current event and station
            traces_data = traces[event_idx][station_idx,:,0]  # Assuming your traces have shape (150,1)
            
            # Access S1000 for the current event
            info_event = test_data.event[event_idx]
            s1000 = 10 ** info_event[1][0]
            y_value = test_data.label[event_idx]

            # Check if the current event index is in the list of indices to highlight
            if event_idx in indices and s1000 < 20:
                # Plot the highlighted trace
                axs[0, station_idx].plot(traces_data, color=highlight_color, zorder =1)
            elif s1000 < 20 and y_value == 0:
                # Plot the regular trace for protons
                axs[0, station_idx].plot(traces_data, color='darkblue', zorder =0, alpha=0.05)

            # Check if the current event index is in the list of indices to highlight for the lower plots
            if event_idx in indices and s1000 < 20:
                # Plot the highlighted trace
                axs[1, station_idx].plot(traces_data, color=highlight_color, zorder =1)
            elif s1000 < 20 and y_value == 1:
                 # Plot the regular trace for protons
                axs[1, station_idx].plot(traces_data, color='orange', zorder =0,alpha=0.05)

        # Set labels and title for each subplot
        axs[0, station_idx].set_xlabel('Time bin')
        axs[0, station_idx].set_ylabel('Cumulative traces')
        axs[0, station_idx].set_title(r'Traces for Station {} (protons with S(1000) $<$ 20)'.format(station_idx+1))

        axs[1, station_idx].set_xlabel('Time bin')
        axs[1, station_idx].set_ylabel('Cumulative traces')
        axs[1, station_idx].set_title(r'Traces for Station {} (photons with S(1000) $<$ 20)'.format(station_idx+1))


    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(new_path)
    print(new_path, full_path)
    plt.close(fig)

    
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
        info_event = test_data.event[idx]

        # Extract values from the list and convert them to strings
        s1000 = 10 ** info_event[1][0]
        theta = info_event[0][0]
        nstat = 10 ** info_event[2][0]
        
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
            axes[0].text(value, axes[0].get_ylim()[1] * 0.95, f'{count}', color='darkblue', ha='left', rotation=0)

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
            axes[1].text(value, axes[1].get_ylim()[1] * 0.95, f'{count}', color='darkblue', ha='left', rotation=0)

    # Histogram 3
    # Calculate the bin edges
    bin_edges = np.arange(1, 21, 1)

    # Plot histogram with custom bins
    _, bins, _ = axes[2].hist(nstat_all_values, bins=bin_edges, color='red', alpha=0.7)
    axes[2].set_title('$N_{stat}$ Distribution')
    axes[2].set_xlabel('$N_{stat}$')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xticks(np.arange(min(bins), max(bins) + 1, 1))

    # Add vertical lines for selected values
    value_counts = collections.Counter(nstat_selected_values)
    for value, count in value_counts.items():
        axes[2].axvline(x=value, color='darkred', linestyle='--')
        if count > 1:
            axes[2].text(value + 0.1, axes[2].get_ylim()[1] * 0.95, f'{count}', color='black', ha='left', rotation=0)

    plt.tight_layout()
    plt.savefig(full_path)
    plt.close(fig)  # Close the figure to release memory





def plot_tpr_vs_S1000(results, datasets, num_bins=5, pdf_name="tpr_vs_S1000_binning_equal.pdf", save_path=PDF_SAVE_PATH):
    """
    Plot the True Positive Rate (TPR) versus S1000/VEM with equal binning.

    Parameters:
    -----------
    results : dict
        Dictionary containing predicted and true labels for each model and dataset pair.
    datasets : dict
        Dictionary containing datasets.
    num_bins : int, optional
        Number of bins for binning S1000/VEM. Default is 5.
    pdf_name : str, optional
        Name of the PDF file to save the plot. Default is "tpr_vs_S1000_binning_equal.pdf".
    save_path : str, optional
        Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.

    Returns:
    --------
    None
    """
    tpr_per_model = {}

    min_tot_S1000 = 100

    # Generate PDF and save it to the specified path
    if len(results) == 1:
        model_name = next(iter(results.keys()))  # Get the model name if there's only one model
        filename = pdf_name
    else:
        filename = f"{pdf_name.split('.')[0]}_{{model_name}}.pdf"  # Add model name to the filename if there are multiple models

    plt.rcParams.update({'text.usetex': True})

    for model_name, data in results.items():
        y_true = data['y_true']
        y_pred = data['y_pred']

        test_dataset = datasets[model_name]['test']

        s1000 = np.array([test_dataset.event[idx][1][0] for idx in range(len(test_dataset.event))])

        # Compute the bin edges
        x_edges_equal = utils.compute_bin_edges(s1000, num_bins)

        min_S1000 = s1000.min()
        max_S1000 = s1000.max()
        width = (max_S1000 - min_S1000) / num_bins

        if(min_tot_S1000 > min_S1000):
            min_tot_S1000 = min_S1000

        # Compute the bin edges
        x_edges = np.arange(start=min_S1000, stop=max_S1000 + 0.05, step=width)

        tpr_arr = np.empty(x_edges.size - 1)
        signal_count_arr = np.empty(x_edges.size - 1)

        for i, x in enumerate(x_edges_equal[:-1]):
            lab_true = np.array([lab for Sref, lab in zip(s1000, y_true) if x < Sref < x_edges_equal[i + 1]])
            lab_pred = np.array([lab for Sref, lab in zip(s1000, y_pred) if x < Sref < x_edges_equal[i + 1]])
            tp = np.logical_and(lab_pred >= 0.5, lab_true == True)
            tpr = tp.sum() / np.count_nonzero(lab_true)
            tpr_arr[i] = tpr
            signal_count_arr[i] = np.count_nonzero(lab_true)
            tpr_per_model[model_name] = tpr_arr

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
            ax.text(rect.get_x() + rect.get_width() / 2, height, f"{bin_value:.0f}\n{height * 100:.2f}\%", ha='center',
                    va='bottom')
        ax.tick_params(axis='y', which='minor')
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        s1000_values = 10 ** x_axis_equal
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels([f"{val:.1f}" for val in s1000_values])
        ax2.set_xlabel('$S1000 (VEM)$', fontsize='13')

        full_path = os.path.join(save_path, filename.format(model_name=model_name))
        fig.savefig(full_path)
        plt.close(fig)  # Close the figure to release memory

    # Compute mean and standard deviation across models for each bin
    tpr_mean = np.mean(list(tpr_per_model.values()), axis=0)
    tpr_std = np.std(list(tpr_per_model.values()), axis=0)
    tpr_emean = tpr_std / np.sqrt(len(tpr_per_model))

    # Getting the bin centers
    x_axis_equal[0] = min_tot_S1000 
    bin_centers = 0.5 * (x_axis[:-1] + x_axis[1:])

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

    fig, ax = plt.subplots()
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_axis_equal)
    ax.grid(zorder=3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.errorbar(bin_centers, tpr_mean, yerr=tpr_emean, fmt='o', color="blue", ecolor='darkblue', capsize=4, label='Mean TPR')
    ax.set_ylabel('TPR', fontsize='13')
    ax.set_xlabel('$\mathrm{log_{10}(S1000/VEM)}$', fontsize='13')

    ax.legend(prop={'size': 13}, edgecolor="black")
    ax.tick_params(axis='y', which='minor')
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)
    ax.set_xlim(x_axis_equal[0], x_axis_equal[-1])

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    s1000_values = 10 ** x_axis_equal
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{val:.1f}" for val in s1000_values])
    ax2.set_xlabel('$S1000 (VEM)$', fontsize='13')

    full_path = os.path.join(save_path, os.path.splitext(pdf_name)[0] + "_mean.pdf")
    fig.savefig(full_path)
    plt.close(fig)



def plot_fpr_vs_S1000(results, datasets, num_bins=5, pdf_name="fpr_vs_S1000_binning_equal.pdf", save_path=PDF_SAVE_PATH):
    """
    Plot the False Positive Rate (FPR) versus S1000/VEM with equal binning.

    Parameters:
    -----------
    results : dict
        Dictionary containing predicted and true labels for each model and dataset pair.
    datasets : dict
        Dictionary containing datasets.
    num_bins : int, optional
        Number of bins for binning S1000/VEM. Default is 5.
    pdf_name : str, optional
        Name of the PDF file to save the plot. Default is "fpr_vs_S1000_binning_equal.pdf".
    save_path : str, optional
        Path to the directory where the plots will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.

    Returns:
    --------
    None
    """
    plt.rcParams.update({'text.usetex': True})

    fpr_per_model = {}

    min_tot_S1000 = 100

    if len(results) == 1:
        model_name = next(iter(results.keys()))  # Get the model name if there's only one model
        filename = pdf_name
    else:
        filename = f"{pdf_name.split('.')[0]}_{{model_name}}.pdf"  # Add model name to the filename if there are multiple models

    for model_name, data in results.items():
        y_true = data['y_true']
        y_pred = data['y_pred']

        test_dataset = datasets[model_name]['test']

        s1000 = np.array([test_dataset.event[idx][1][0] for idx in range(len(test_dataset.event))])
        


        # Compute the bin edges
        x_edges_equal = utils.compute_bin_edges(s1000, num_bins)

        min_S1000 = s1000.min()
        max_S1000 = s1000.max()
        width = (max_S1000 - min_S1000) / num_bins

        if(min_tot_S1000 > min_S1000):
            min_tot_S1000 = min_S1000

        # Compute the bin edges
        x_edges = np.arange(start=min_S1000, stop=max_S1000 + 0.05, step=width)

        fpr_arr = np.empty(x_edges.size - 1)
        background_count_arr = np.empty(x_edges.size - 1)

        for i, x in enumerate(x_edges_equal[:-1]):
            lab_true = np.array([lab for Sref, lab in zip(s1000, y_true) if x < Sref < x_edges_equal[i + 1]])
            lab_pred = np.array([lab for Sref, lab in zip(s1000, y_pred) if x < Sref < x_edges_equal[i + 1]])
            fp = np.logical_and(lab_pred >= 0.5, lab_true == False)  # False positives
            fpr = fp.sum() / np.count_nonzero(lab_true == False)  # False Positive Rate
            fpr_arr[i] = fpr
            background_count_arr[i] = np.count_nonzero(lab_true == False)
            fpr_per_model[model_name] = fpr_arr

        x_axis = np.round(x_edges, 2)
        x_axis_equal = np.round(x_edges_equal, 2)

        fig, ax = plt.subplots()
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_axis_equal)
        ax.grid(zorder=3)
        ax.hist(x_axis[:-1], x_axis, weights=fpr_arr, rwidth=0.95, color="lightblue", ec="blue", zorder=4)
        ax.set_ylabel('FPR', fontsize='13')
        ax.set_xlabel('$\mathrm{log_{10}(S1000/VEM)}$', fontsize='13')

        rects = ax.patches
        for rect, bin_value in zip(rects, background_count_arr):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height, f"{bin_value:.0f}\n{height * 100:.2f}\%", ha='center',
                    va='bottom')
        ax.tick_params(axis='y', which='minor')
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        s1000_values = 10 ** x_axis_equal
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels([f"{val:.1f}" for val in s1000_values])
        ax2.set_xlabel('$S1000 (VEM)$', fontsize='13')

        full_path = os.path.join(save_path, filename.format(model_name=model_name))
        fig.savefig(full_path)
        plt.close(fig)  # Close the figure to release memory

    # Compute mean and standard deviation across models for each bin
    fpr_mean = np.mean(list(fpr_per_model.values()), axis=0)
    fpr_std = np.std(list(fpr_per_model.values()), axis=0)
    fpr_emean = fpr_std / np.sqrt(len(fpr_per_model))

    # Getting the bin centers
    x_axis_equal[0] = min_tot_S1000 
    bin_centers = 0.5 * (x_axis[:-1] + x_axis[1:])

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

    fig, ax = plt.subplots()
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_axis_equal)
    ax.grid(zorder=3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.errorbar(bin_centers, fpr_mean, yerr=fpr_emean, fmt='o', color="blue", ecolor='darkblue', capsize=4, label='Mean FPR')
    ax.set_ylabel('FPR', fontsize='13')
    ax.set_xlabel('$\mathrm{log_{10}(S1000/VEM)}$', fontsize='13')

    ax.legend(prop={'size': 13}, edgecolor="black")
    ax.tick_params(axis='y', which='minor')
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)
    ax.set_xlim(x_axis_equal[0], x_axis_equal[-1])

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    s1000_values = 10 ** x_axis_equal
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{val:.1f}" for val in s1000_values])
    ax2.set_xlabel('$S1000 (VEM)$', fontsize='13')

    full_path = os.path.join(save_path, os.path.splitext(pdf_name)[0] + "_mean.pdf")
    fig.savefig(full_path)
    plt.close(fig)




def plot_fpr_vs_theta(results, datasets, num_bins=5, pdf_name="fpr_vs_theta.pdf", save_path=PDF_SAVE_PATH):
    """
    Plot the False Positive Rate (FPR) versus zenith angle theta.

    Parameters:
    -----------
    results : dict
        Dictionary containing predicted and true labels for each model and dataset pair.
    datasets : dict
        Dictionary containing datasets.
    num_bins : int, optional
        Number of bins for binning theta. Default is 5.
    pdf_name : str, optional
        Name of the PDF file to save the plot. Default is "fpr_vs_theta_binning_equal_size.pdf".
    save_path : str, optional
        Path to the directory where the plots will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.

    Returns:
    --------
    None
    """
    plt.rcParams.update({'text.usetex': True})

    min_theta = 0.0
    max_theta = 60
    width = (max_theta - min_theta) / num_bins

    fpr_per_model = {}

    if len(results) == 1:
        model_name = next(iter(results.keys()))  # Get the model name if there's only one model
        filename = pdf_name
    else:
        filename = f"{pdf_name.split('.')[0]}_{{model_name}}.pdf"  # Add model name to the filename if there are multiple models

    for model_name, data in results.items():
        y_true = data['y_true']
        y_pred = data['y_pred']

        test_dataset = datasets[model_name]['test']

        theta = np.array([test_dataset.event[idx][0][0] for idx in range(len(test_dataset.event))])

        # Compute the bin edges
        x_edges = np.arange(start=min_theta, stop=max_theta + 0.05, step=width)

        fpr_arr = np.empty(x_edges.size - 1)
        background_count_arr = np.empty(x_edges.size - 1)

        for i, x in enumerate(x_edges[:-1]):
            lab_true = np.array([lab for theta_val, lab in zip(theta, y_true) if x < theta_val < x_edges[i + 1]])
            lab_pred = np.array([lab for theta_val, lab in zip(theta, y_pred) if x < theta_val < x_edges[i + 1]])
            fp = np.logical_and(lab_pred >= 0.5, lab_true == False)  # False positives
            fpr = fp.sum() / np.count_nonzero(lab_true == False)  # False Positive Rate
            fpr_arr[i] = fpr
            background_count_arr[i] = np.count_nonzero(lab_true == False)
            fpr_per_model[model_name] = fpr_arr

        x_axis = np.round(x_edges, 2)

        fig, ax = plt.subplots()
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_axis)
        ax.grid(zorder=3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.hist(x_axis[:-1], x_axis, weights=fpr_arr, rwidth=0.95, color="lightblue", ec="blue", zorder=4)
        ax.set_ylabel('FPR', fontsize='13')
        ax.set_xlabel(r'$\theta$ (°)', fontsize='13')

        rects = ax.patches
        for rect, bin_value in zip(rects, background_count_arr):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height, f"{bin_value:.0f}\n{height * 100:.2f}\%", ha='center',
                    va='bottom')
        ax.tick_params(axis='y', which='minor')
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)

        x_edges_radians = np.deg2rad(x_edges)

        # Set the x-axis ticks to display angles in radians on the upper axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(np.linspace(x_edges[0], x_edges[-1], len(x_edges)))
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax2.set_xticklabels([f"{val:.1f}" for val in x_edges_radians])
        ax2.set_xlabel(r'$\theta$ (rad)', fontsize='13')

        full_path = os.path.join(save_path, filename.format(model_name=model_name))
        fig.savefig(full_path)
        plt.close(fig)  # Close the figure to release memory


    # Compute mean and standard deviation across models for each bin
    fpr_mean = np.mean(list(fpr_per_model.values()), axis=0)
    fpr_std = np.std(list(fpr_per_model.values()), axis=0)
    fpr_emean = fpr_std / np.sqrt(len(fpr_per_model))

    # Getting the bin centers
    bin_centers = 0.5 * (x_axis[:-1] + x_axis[1:])

    bin_centers = np.round(bin_centers, 2)

    fig, ax = plt.subplots()
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_axis)
    ax.grid(zorder=3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.errorbar(bin_centers, fpr_mean, yerr=fpr_emean, fmt='o', color="blue", ecolor='darkblue', capsize=4, label='Mean FPR')
    ax.set_ylabel('FPR', fontsize='13')
    ax.set_xlabel(r'$\theta$ (°)', fontsize='13')

    ax.legend(prop={'size': 13}, edgecolor="black")
    ax.tick_params(axis='y', which='minor')
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)
    ax.set_xlim(left=min_theta, right=max_theta)
    x_edges_radians = np.deg2rad(x_edges)

    # Set the x-axis ticks to display angles in radians on the upper axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.linspace(x_edges[0], x_edges[-1], len(x_edges)))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax2.set_xticklabels([f"{val:.1f}" for val in x_edges_radians])
    ax2.set_xlabel(r'$\theta$ (rad)', fontsize='13')

    full_path = os.path.join(save_path, os.path.splitext(pdf_name)[0] + "_mean.pdf")
    fig.savefig(full_path)
    plt.close(fig)






def plot_tpr_vs_theta(results, datasets, num_bins=5, pdf_name="tpr_vs_theta.pdf", save_path=PDF_SAVE_PATH):
    """
    Plot the True Positive Rate (TPR) versus zenith angle theta.

    Parameters:
    -----------
    results : dict
        Dictionary containing predicted and true labels for each model and dataset pair.
    datasets : dict
        Dictionary containing datasets.
    num_bins : int, optional
        Number of bins for binning theta. Default is 5.
    pdf_name : str, optional
        Name of the PDF file to save the plot. Default is "tpr_vs_theta_binning_equal_size.pdf".
    save_path : str, optional
        Path to the directory where the plots will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.

    Returns:
    --------
    None
    """
    plt.rcParams.update({'text.usetex': True})

    tpr_per_model = {}


    min_theta = 0.0
    max_theta = 60
    width = (max_theta - min_theta) / num_bins


    if len(results) == 1:
        model_name = next(iter(results.keys()))  # Get the model name if there's only one model
        filename = pdf_name
    else:
        filename = f"{pdf_name.split('.')[0]}_{{model_name}}.pdf"  # Add model name to the filename if there are multiple models

    for model_name, data in results.items():
        y_true = data['y_true']
        y_pred = data['y_pred']

        test_dataset = datasets[model_name]['test']

        theta = np.array([test_dataset.event[idx][0][0] for idx in range(len(test_dataset.event))])
    
        # Compute the bin edges
        x_edges = np.arange(start=min_theta, stop=max_theta + 0.05, step=width)

        tpr_arr = np.empty(x_edges.size - 1)
        signal_count_arr = np.empty(x_edges.size - 1)

        for i, x in enumerate(x_edges[:-1]):
            lab_true = np.array([lab for theta_val, lab in zip(theta, y_true) if x < theta_val < x_edges[i + 1]])
            lab_pred = np.array([lab for theta_val, lab in zip(theta, y_pred) if x < theta_val < x_edges[i + 1]])
            tp = np.logical_and(lab_pred >= 0.5, lab_true == True)  # True positives
            tpr = tp.sum() / np.count_nonzero(lab_true == True)  # True Positive Rate
            tpr_arr[i] = tpr
            signal_count_arr[i] = np.count_nonzero(lab_true)

        tpr_per_model[model_name] = tpr_arr

        x_axis = np.round(x_edges, 2)

        fig, ax = plt.subplots()
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_axis)
        ax.grid(zorder=3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.hist(x_axis[:-1], x_axis, weights=tpr_arr, rwidth=0.95, color="lightblue", ec="blue", zorder=4)
        ax.set_ylabel('TPR', fontsize='13')
        ax.set_xlabel(r'$\theta$ (°)', fontsize='13')

        rects = ax.patches
        for rect, bin_value in zip(rects, signal_count_arr):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height, f"{bin_value:.0f}\n{height * 100:.2f}\%", ha='center',
                    va='bottom')
        ax.tick_params(axis='y', which='minor')
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)

        x_edges_radians = np.deg2rad(x_edges)

        # Set the x-axis ticks to display angles in radians on the upper axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(np.linspace(x_edges[0], x_edges[-1], len(x_edges)))
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax2.set_xticklabels([f"{val:.1f}" for val in x_edges_radians])
        ax2.set_xlabel(r'$\theta$ (rad)', fontsize='13')

        full_path = os.path.join(save_path, filename.format(model_name=model_name))
        fig.savefig(full_path)
        plt.close(fig)  # Close the figure to release memory

    # Compute mean and standard deviation across models for each bin
    tpr_mean = np.mean(list(tpr_per_model.values()), axis=0)
    tpr_std = np.std(list(tpr_per_model.values()), axis=0)
    tpr_emean = tpr_std / np.sqrt(len(tpr_per_model))

    # Getting the bin centers
    bin_centers = 0.5 * (x_axis[:-1] + x_axis[1:])

    bin_centers = np.round(bin_centers, 2)

    fig, ax = plt.subplots()
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_axis)
    ax.grid(zorder=3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.errorbar(bin_centers, tpr_mean, yerr=tpr_emean, fmt='o', color="blue", ecolor='darkblue', capsize=4, label='Mean TPR')
    ax.set_ylabel('TPR', fontsize='13')
    ax.set_xlabel(r'$\theta$ (°)', fontsize='13')

    ax.legend(prop={'size': 13}, edgecolor="black")
    ax.tick_params(axis='y', which='minor')
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)
    ax.set_xlim(left=min_theta, right=max_theta)
    x_edges_radians = np.deg2rad(x_edges)

    # Set the x-axis ticks to display angles in radians on the upper axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.linspace(x_edges[0], x_edges[-1], len(x_edges)))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax2.set_xticklabels([f"{val:.1f}" for val in x_edges_radians])
    ax2.set_xlabel(r'$\theta$ (rad)', fontsize='13')

    full_path = os.path.join(save_path, os.path.splitext(pdf_name)[0] + "_mean.pdf")
    fig.savefig(full_path)
    plt.close(fig)