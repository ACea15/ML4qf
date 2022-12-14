import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             plot_confusion_matrix, plot_roc_curve, RocCurveDisplay,
                             ConfusionMatrixDisplay)
from sklearn.inspection import DecisionBoundaryDisplay#, ConfusionMatrixDisplay.

from xgboost import to_graphviz


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

class PlotMetrics:

    def __init__(self, predictor):

        self.pred = predictor

    def confusion_matrix(self, X_test, y_test, y_pred=None, return_values=True,
                         **kwargs):
        # Plot confusion matrix
        kwargs['cmap'] = kwargs.get('cmap', 'Blues')
        fig, ax = plt.subplots(figsize=kwargs.pop('figsize'))
        # plot_confusion_matrix(self.pred, X_test, y_test,
        #                       ax=ax, cmap='Blues', values_format='.4g')
        if y_pred is None:
            disp = ConfusionMatrixDisplay.from_estimator(self.pred, X_test, y_test,
                                                  ax=ax, **kwargs)
        else:
            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                                           ax=ax, **kwargs)
        t0, f1, f0, t1 = disp.confusion_matrix.ravel()
        plt.title('Confusion Matrix')
        plt.grid(False)
        plt.show()
        if return_values:
            return {'t0':t0, 'f1':f1, 'f0':f0, 't1':t1}

    def roc_curve(self, X_test, y_test, y_pred=None, **kwargs):

        fig, ax = plt.subplots(figsize=kwargs.pop('figsize'))
        #plot_roc_curve(self.pred, X_test, y_test, ax=ax, color='crimson')
        if y_pred is None:
            disp = RocCurveDisplay.from_estimator(self.pred, X_test, y_test,
                                                  ax=ax, **kwargs)
        else:
            disp = RocCurveDisplay.from_predictions(y_test, y_pred,
                                                    ax=ax, **kwargs)
        ax.plot([0,1], [0,1], linestyle='--')
        #disp.ax_.set_title('ROC Curve')
        plt.show()

    def graphviz(self):

        ## Tree Visualization
        # change tree number to see the corresponding plot
        to_graphviz(self.pred, num_trees=2, rankdir='UT')

    def feature_importance(self, labels=None, return_list=True, **kwargs):
        # Plot feature importance
        # feature importance_type = 'gain'
        if labels is None:
            labels = range(len(self.pred.feature_importances_))
        fig, ax = plt.subplots(figsize=kwargs.pop('figsize'))
        feature_imp = pd.DataFrame({'Importance Score': self.pred.feature_importances_,
                                    'Features': labels}).sort_values(by='Importance Score',
                                                                        ascending=False)
        sns.barplot(x=feature_imp['Importance Score'], y=feature_imp['Features'])
        ax.set_title('Features Importance')
        plt.show()
        if return_list:
            return feature_imp

class PlotTree(PlotMetrics):

    def structure(self):

        n_nodes = self.pred.tree_.node_count
        children_left = self.pred.tree_.children_left
        children_right = self.pred.tree_.children_right
        feature = self.pred.tree_.feature
        threshold = self.pred.tree_.threshold

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        print(
            "The binary tree structure has {n} nodes and has "
            "the following tree structure:\n".format(n=n_nodes)
        )
        for i in range(n_nodes):
            if is_leaves[i]:
                print(
                    "{space}node={node} is a leaf node.".format(
                        space=node_depth[i] * "\t", node=i
                    )
                )
            else:
                print(
                    "{space}node={node} is a split node: "
                    "go to node {left} if X[:, {feature}] <= {threshold} "
                    "else to node {right}.".format(
                        space=node_depth[i] * "\t",
                        node=i,
                        left=children_left[i],
                        feature=feature[i],
                        threshold=threshold[i],
                        right=children_right[i],
                    )
                )

    def tree(self, **kwargs):
        plt.figure(figsize=kwargs.pop('figsize'))
        plt.tight_layout()
        tree.plot_tree(self.pred, **kwargs)
        plt.show()

    def boundaries(self, X, y, plot_colors="rybkg", **kwargs):
        # Plot the decision boundary
        ax = plt.subplot()
        plt.tight_layout()
        disp = DecisionBoundaryDisplay.from_estimator(
            self.pred,
            X,
            #cmap=plt.cm.RdYlBu,
            response_method="predict",
            ax=ax,
            **kwargs)
        disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
        # n_classes = len(self.pred.classes_)
        # # Plot the training points
        # for i, color in zip(range(n_classes), plot_colors[:n_classes]):
        #     idx = np.where(y == i)
        #     plt.scatter(
        #         X[idx, 0],
        #         X[idx, 1],
        #         c=color,
        #         label=iris.target_names[i],
        #         cmap=plt.cm.RdYlBu,
        #         edgecolor="black",
        #         s=15,
        #     )

        plt.show()

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


class PlotSeries():

    def __init__(self, df):

        self.df = df

    def line(self, feature: 'str', **kwargs):
        
        fig = px.line(self.df, y=feature,  **kwargs)
        fig.show()
        
    def candlstick(self, open_sym='Open',
                   high_sym='High',
                   low_sym='Low',
                   close_sym='Close',
                   x_sym=None,
                   rangeslider=True,
                   **kwargs):
        
        if x_sym is None:
            x_sym = self.df.index
        fig = go.Figure(data=[go.Candlestick(x=x_sym,
                open=self.df[open_sym],
                high=self.df[high_sym],
                low=self.df[low_sym],
                close=self.df[close_sym])])
        if rangeslider:
            fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()
