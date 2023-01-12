
class Main:

    def __init__(self):

        self.inputs = None

class Labels:

    def set_bintarget(df, label='price', alpha=0., check_iftarget=True,
                      remove_nan=True, print_info=True):
        """ Sets binary target"""

        target_set = False
        if 'Target' not in df.columns:
            df['Target'] = np.where(df[label].shift(-1) > (1 - alpha) * df[label], 1, 0)
            target_set = True
        elif not check_iftarget:
            df['Target'] = np.where(df[label].shift(-1) > (1 - alpha) * df[label], 1, 0)
            target_set = True
        if remove_nan and target_set:
            df = df[:-1]
        if print_info:
            df['Target'].count
        return df
