import numpy as np

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
            # print "INDEX", i, "SCORE", score, "NUMBER OF HITS", num_hits, "PREDICTED", p

    if not actual:
        return 0.0
    #Use score variable if you want ranked MAP otherwise use num_hits
    # return score / min(len(actual), k)
    return num_hits / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    array : list
            All the values which are used in the mean calculation

    """
    all_average_prec = []
    for a,p in zip(actual, predicted):
        average_precision_val = apk(a,p,k)
        all_average_prec.append(average_precision_val)
    return np.mean(all_average_prec), all_average_prec

