import numpy as np
import minisom


class Model:

    def __init__(self, sizeX, sizeY, X_train, y_train=None, initialise_pca=True,
                 sigma=3, learning_rate=0.7, neighborhood_function='gaussian', topology='rectangular',
                 activation_distance='euclidean', num_iter=1000, random_seed=42, verbose=False):

        self.sizeX = sizeX
        self.sizeY = sizeY
        self.X_train = X_train
        self.y_train =y_train
        self.initialise_pca = initialise_pca
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.neighborhood_function = neighborhood_function
        self.topology = topology
        self.activation_distance = activation_distance
        self.num_iter = num_iter
        self.random_seed = random_seed
        self.verbose = verbose
        self.num_features = self.X_train.shape[1]
        self._build_som()

    def _build_som(self):

        self.som = minisom.MiniSom(self.sizeX, self.sizeY, self.num_features, sigma=self.sigma, learning_rate=self.learning_rate, 
                                   neighborhood_function=self.neighborhood_function, topology=self.topology,
                                   activation_distance=self.activation_distance, random_seed=self.random_seed)
        if self.initialise_pca:
            self.som.pca_weights_init(self.X_train)
        self.som.train(self.X_train, self.num_iter, verbose=self.verbose)
        self.W = self.som.get_weights()

    def classify(self, X_test):
        """Classifies each sample in Xtest in one of the classes definited
        using the method labels_map.
        Returns a list of the same length of Xtest where the i-th element
        is the class assigned to Xtest[i].
        """
        winmap = self.som.labels_map(self.X_train, self.y_train)
        default_class = np.sum(list(winmap.values())).most_common()[0][0]
        result = []
        for d in X_test:
            win_position = self.som.winner(d)
            if win_position in winmap:
                result.append(winmap[win_position].most_common()[0][0])
            else:
                result.append(default_class)

        return result

    @staticmethod
    def feature_selection(W, labels, target_index = -1 , a = 0.04):
        """ Performs feature selection based on a self organised map trained with the desired variables

        INPUTS: W = numpy array, the weights of the map (X*Y*N) where X = map's rows, Y = map's columns, N = number of variables
                labels = list, holds the names of the variables in same order as in W
                target_index = int, the position of the target variable in W and labels
                a = float, an arbitary parameter in which the selection depends, values between 0.03 and 0.06 work well

        OUTPUTS: selected_labels = list of strings, holds the names of the selected features in order of selection
                 target_name = string, the name of the target variable so that user is sure he gave the correct input
        """


        W_2d = np.reshape(W, (W.shape[0]*W.shape[1], W.shape[2])) #reshapes W into MxN assuming M neurons and N features
        target_name = labels[target_index]

        Rand_feat = np.random.uniform(low=0, high=1, size=(W_2d.shape[0], W_2d.shape[1] - 1)) # create N -1 random features
        W_with_rand = np.concatenate((W_2d,Rand_feat), axis=1) # add them to the N regular ones
        W_normed = (W_with_rand - W_with_rand.min(0)) / W_with_rand.ptp(0) # normalize each feature between 0 and 1
        Target_feat = W_normed[:,target_index] # column of target feature
        # Two conditions to check against a
        Check_matrix1 = abs(np.vstack(Target_feat) - W_normed)
        Check_matrix2 = abs(np.vstack(Target_feat) + W_normed - 1)
        S = np.logical_or(Check_matrix1 <= a, Check_matrix2 <= a).astype(int) # applie "or" element-wise in two matrices

        S[:,target_index] = 0 #ignore the target feature so that it is not picked

        selected_labels = []
        while True:

            S2 = np.sum(S, axis=0) # add all rows for each column (feature)

            if not np.any(S2 > 0): # if all features add to 0 kill
                break

            selected_feature_index = np.argmax(S2) # feature with the highest sum gets selected first

            if selected_feature_index > (S.shape[1] - (Rand_feat.shape[1] + 1)): # if random feature is selected kill
                break
            if selected_feature_index != target_index:
                selected_labels.append(labels[selected_feature_index])

            # delete all rows where selected feature evaluates to 1, thus avoid selecting complementary features
            rows_to_delete = np.where(S[:,selected_feature_index] == 1)
            S[rows_to_delete, :] = 0

            #     selected_labels = [label for i, label in enumerate(labels) if i in feature_indeces]
        return selected_labels, target_name

    def iterate_som_selection(self, min_num_features, labels, a_range, num_iterations=100,
                              target_index=-1, target_nameX='target'):

        i = 0
        features_len = 0
        som_labels = set()
        while features_len < min_num_features and i < num_iterations:
            som_labeli, target_name = self.feature_selection(self.W, labels=labels, target_index = target_index, a = a_range[i % len(a_range)])
            som_labels = som_labels.union(set(som_labeli))
            features_len = len(som_labels)
            i += 1
        assert target_name == target_nameX, "targets not matching"
        print("Total number of iterations: %s" %i)
        return list(som_labels)

