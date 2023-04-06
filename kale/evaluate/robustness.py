import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def relative_robustness(robustness_result, task):
    """Compute the relative robustenss metric given the performance of the method on the task."""
    return get_robustness_metric(robustness_result, task, 'relative')


def effective_robustness(robustness_result, task):
    """Compute the effective robustenss metric given the performance of the method on the task."""
    return get_robustness_metric(robustness_result, task, 'effective')


def get_robustness_metric(robustness_result, task, metric):
    """
    Compute robustness metric given specific method performance and the task.
    :param robustness_result: Performance of the method on datasets applied with different level of noises.
    :param task: Name of the task on which the method is evaluated.
    :param metric: Type of robustness metric to be computed. ( "effective" / "relative" )
    """
    if metric == 'effective' and task not in robustness['LF']:
        return "Invalid example name!"
    else:
        result = dict()
        if metric == 'relative':
            helper = relative_robustness_helper
        elif metric == 'effective':
            helper = effective_robustness_helper
        my_method = helper(robustness_result, task)
        for method in list(robustness.keys()):
            if not method.endswith('Transformer'):
                for t in list(robustness[method].keys()):
                    if t == task:
                        if (method == 'EF' or method == 'LF') and task in robustness[method+'-Transformer']:
                            result[method] = helper((np.array(
                                robustness[method][task])+np.array(robustness[method+'-Transformer'][task]))/2, task)
                        else:
                            result[method] = helper(
                                robustness[method][task], task)
        result['my method'] = my_method
        return maxmin_normalize(result, task)


def relative_robustness_helper(robustness_result, task):
    """
    Helper function that computes the relative robustness metric as the area under the performance curve.
    :param robustness_result: Performance of the method on datasets applied with different level of noises.
    """
    area = 0
    for i in range(len(robustness_result)-1):
        area += (robustness_result[i] + robustness_result[i+1]) * 0.1 / 2
    return area


def effective_robustness_helper(robustness_result, task):
    """
    Helper function that computes the effective robustness metric as the performance difference compared to late fusion method.
    :param robustness_result: Performance of the method on datasets applied with different level of noises.
    :param task: Name of the task on which the method is evaluated.
    """
    f = np.array(robustness_result)
    lf = np.array(robustness['LF'][task])
    beta_f = lf + (f[0] - lf[0])
    return np.sum(f - beta_f)


def maxmin_normalize(result, task):
    """
    Normalize the metric for robustness comparison across all methods.
    :param result: Un-normalized robustness metrics of all methods on the given task.
    :param task: Name of the task.
    """
    tmp = []
    method2idx = dict()
    for i, method in enumerate(list(result.keys())):
        method2idx[method] = i
        tmp.append(result[method])
    tmp = np.array(tmp)
    if task.startswith('finance'):
        tmp = -1 * tmp
    tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
    return tmp[method2idx['my method']]


def single_plot(robustness_result, task, xlabel, ylabel, fig_name, method):
    """
    Produce performance vs. robustness plot of a single method.
    :param robustness_result: Performance of the method on dataset applied with different level of noises.
    :param task: Name of the task on which the method is evaluated.
    :param xlabel: Label of x-axis to be appeared in the plot.
    :param ylabel: Label of y-axis to be appeared in the plot.
    :param fig_name: Name of plot to be saved.
    :param method: Name of the method.
    """
    fig, axs = plt.subplots()
    if task.startswith('gentle push') or task.startswith('robotics image') or task.startswith('robotics force'):
        robustness_result = list(np.log(np.array(robustness_result)))
        plt.ylabel('log '+ylabel, fontsize=20)
    axs.plot(np.arange(len(robustness_result)) / 10,
             robustness_result, label=method, linewidth=2.5)
    plt.xlabel(xlabel, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Uncomment the line below to show legends
    # fig.legend(fontsize=15, loc='upper left', bbox_to_anchor=(0.92, 0.94))
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close(fig)
