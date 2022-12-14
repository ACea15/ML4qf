def apply_scaler(X, scaler_name='MinMaxScaler', **fit_params):

    scaler = getattr(preprocessing, scaler_name)
    if isinstance(X, (np.ndarray, list)):
        X_new = scaler().fit_transform(X, **fit_params)
        return X_new
    elif isinstance(X, pd.DataFrame):
        X_new = scaler().fit_transform(X.values, **fit_params)
        df_new = pd.DataFrame(X_new, index=X.index, columns=X.columns)
        return df_new

def split_data(X, y, data_frame=None, **kwargs):

    # Splitting the datasets into training and testing data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)

    # Output the train and test data size
    print(f"Train and Test Size {len(X_train)}, {len(X_test)}")
    if data_frame is not None:
        data_frame1 = data_frame[:len(X_train)]
        data_frame2 = data_frame[len(X_train):]
        assert len(data_frame2) == len(X_test), "mismatch in length of X and df"
        return X_train, X_test, y_train, y_test, data_frame1, data_frame2
    else:
        return X_train, X_test, y_train, y_test

def split_timedata(X, y, output_splitdata=True, **kwargs):

    tscv = TimeSeriesSplit(**kwargs)
    if output_splitdata:
        Xtrain = []
        Xtest = []
        ytrain = []
        ytest = []
        # Cross-validation
        for train_index, test_index in tscv.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            Xtrain.append(X_train)
            Xtest.append(X_test)
            ytrain.append(y_train)
            ytest.append(y_test)
        return tscv, Xtrain, Xtest, ytrain, ytest
    else:
        return tscv

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
