import numpy as np


def evaluate(predictions, labels, metrics=tuple(), final=False):
    """
    :param final:
    :param predictions:
    :param labels:
    :param metrics: a tuple of metrics
    :return: string result of evaluation
    """
    if len(metrics) == 0:
        return ''

    result = '\n```\n'
    result += '          -------------------\n'

    for metric in metrics:
        if (final or not metric.final_validation) and not metric.sort_needed:
            result += str(metric(predictions, labels)) + '\n'
            result += '          -------------------\n'

    predictions = np.argsort(-predictions, axis=1)

    for metric in metrics:
        if (final or not metric.final_validation) and metric.sort_needed:
            result += str(metric(predictions, labels)) + '\n'
            result += '          -------------------\n'

    result += '```\n\n'
    return result
