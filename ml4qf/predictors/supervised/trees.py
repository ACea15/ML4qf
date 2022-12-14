
def tree_pruning(clf, X_train, y_train, X_test, y_test):

    #clf = DecisionTreeClassifier()
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    # plt.figure(figsize=(10, 6))
    # plt.plot(ccp_alphas, impurities)
    # plt.xlabel("effective alpha")
    # plt.ylabel("total impurity of leaves")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ccp_alphas,
                             y=impurities,
                             mode='lines',
                             name='lines'))
    fig.update_layout(legend_title_text = "Contestant")
    fig.update_xaxes(title_text="effective alpha")
    fig.update_yaxes(title_text="total impurity of leaves")
    fig.show()

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    tree_depths = [clf.tree_.max_depth for clf in clfs]
    # plt.figure(figsize=(10,  6))
    # plt.plot(ccp_alphas[:-1], tree_depths[:-1])
    # plt.xlabel("effective alpha")
    # plt.ylabel("total depth")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ccp_alphas[:-1],
                             y=tree_depths[:-1],
                             mode='lines',
                             name='lines'))
    #fig.update_layout(legend_title_text = "Contestant")
    fig.update_xaxes(title_text="effective alpha")
    fig.update_yaxes(title_text="total depth")
    fig.show()

    acc_scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]
    tree_depths = [clf.tree_.max_depth for clf in clfs]
    # plt.figure(figsize=(10,  6))
    # plt.grid()
    # plt.plot(ccp_alphas[:-1], acc_scores[:-1])
    # plt.xlabel("effective alpha")
    # plt.ylabel("Accuracy scores")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ccp_alphas[:-1],
                             y=acc_scores[:-1],
                             mode='lines',
                             name='lines'))
    #fig.update_layout(legend_title_text = "Contestant")
    fig.update_xaxes(title_text="effective alpha")
    fig.update_yaxes(title_text="Accuracy scores")
    fig.show()
    #plt.show()
