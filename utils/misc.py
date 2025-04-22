import pandas as pd

def identify_regimes(params, decimals=4):
    """Identify bull and bear regimes from a fitted Markov switching model's parameters.

    Regime identification:
    - Bull regime: positive beta and lower variance
    - Bear regime: negative beta and higher variance
      
    If conditions are mixed, the function prints a message and returns None.
    
    The decimals argument (default=4) specifies the number of decimals to round each output value.
    Set decimals to None to disable rounding.
    
    returns:
    A dictionary with keys 'bull' and 'bear', each containing a dictionary with:
        - 'beta': const
        - 'var': variance (sigma2)
        - Transition probabilities (for bull: p00, p01; for bear: p10, p11)
    """

    try:
        p11 = params['p[0->0]']
        p21 = params['p[1->0]']
        b1  = params['const[0]']
        b2  = params['const[1]']
        var1 = params['sigma2[0]']
        var2 = params['sigma2[1]']
    except KeyError as e:
        print(f"Missing parameter: {e}")
        return None

    p12 = 1 - p11
    p22 = 1 - p21

    if decimals is not None:
        p11 = round(p11, decimals)
        p21 = round(p21, decimals)
        p12 = round(p12, decimals)
        p22 = round(p22, decimals)
        b1  = round(b1, decimals)
        b2  = round(b2, decimals)
        var1 = round(var1, decimals)
        var2 = round(var2, decimals)
    
    regime1 = {'beta0': b1, 'var': var1, 'p11': p11, 'p12': p12}
    regime2 = {'beta0': b2, 'var': var2, 'p22': p22, 'p21': p21}

    if regime1['beta0']>0 and regime2['beta0']<0 and regime1['var']<regime2['var']:
        bull = regime1
        bear = regime2
    elif regime2['beta0']>0 and regime1['beta0']<0 and regime2['var']<regime1['var']:
        bull = regime2
        bear = regime1
    else:
        print("Mixed regimes: Cannot clearly identify bull and bear regimes.")
        return None

    result_dict = {
        'bull': bull,
        'bear': bear
    }
    return result_dict


def detect_tsm_classifications(
    target_df: pd.DataFrame,
    ground_truth: pd.Series
) -> pd.DataFrame:
    gt_broadcast = pd.concat([ground_truth] * target_df.shape[1], axis=1)
    gt_broadcast.columns = target_df.columns

    result = pd.DataFrame(index=target_df.index, columns=target_df.columns, dtype='int')

    result[(target_df == 1) & (gt_broadcast == 1)] = 1
    result[(target_df == -1) & (gt_broadcast == -1)] = 2
    result[(target_df == 1) & (gt_broadcast == -1)] = 3
    result[(target_df == -1) & (gt_broadcast == 1)] = 4
    return result