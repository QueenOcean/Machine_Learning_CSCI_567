import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    
    K = n_cluster
    N, D = np.shape(x)

    # selecting first center randomly
    first = int(generator.randint(0, n, 1, int))
    res = []
    centers = []
    res.append(x[first])
    centers.append(first)
    

    # find k-1 centers
    for k in range(1, K):
        X = np.zeros([k,N])
        for j in range(k):
            X[j] = np.sum((x - res[j]) ** 2, axis=1)
        M = np.amin(X, axis=0)
        B = np.argmax(M/sum(M))
        centers.append(B)
        res.append(x[B])
    
    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        

        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        
        #TODO
        K = self.n_cluster

        # - Initialize means by picking self.n_cluster from N data points
        means = np.zeros([K, D])
        for k in range(K):
            means[k] = x[self.centers[k]]
            
        B = np.zeros(N)
        J = np.sum([np.sum((x[B == k] - means[k]) ** 2) for k in range(K)]) / N

        # - Update means and membership until convergence 
        iter = 0
        while iter < self.max_iter:
            iter += 1
            
            # Find assignment
            dis = np.sum(((x - np.expand_dims(means, axis=1)) ** 2), axis=2)
            B = np.argmin(dis, axis=0)

            # Find distortion
            J_new = np.sum([np.sum((x[B == k] - means[k]) ** 2) for k in range(K)]) / N
            if np.absolute(J - J_new) <= self.e:
                break
            J = J_new

            # Find means
            means_new = np.array([np.mean(x[B == k], axis=0) for k in range(K)])
            index = np.where(np.isnan(means_new))
            means_new[index] = means[index]
            means = means_new
            
        #return (means, B, iter)
        # - return (means, membership, number_of_updates)
        centroids = means
        y = B
        self.max_iter = iter

        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        
        # TODO:
        # Get centroids and memberships
        K = self.n_cluster
        k_means = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, membership, i = k_means.fit(x, centroid_func)

        # assign labels to centroid_labels
        X = [[] for k in range(K)]
        for i in range(N):
            X[membership[i]].append(y[i])
            
        centroid_labels = np.zeros([K])
        for i in range(K):
            counts = np.bincount(X[i])
            centroid_labels[i] = np.argmax(counts)

        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        
        # TODO:
        K = self.n_cluster
        
        # Find nearest centroid 
        X = np.zeros([K, N])
        for k in range(K):
            X[k] = np.sqrt(np.sum((x - self.centroids[k])**2, axis=1))
        B = np.argmin(X, axis=0)

        # Return labels
        labels = [[] for i in range(N)]
        for i in range(N):
            labels[i] = self.centroid_labels[B[i]]
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    X, Y, Z = image.shape
    x = image.reshape(X * Y, Z)
    dis = np.sum(((x-np.expand_dims(code_vectors, axis=1)) ** 2), axis=2)
    B = np.argmin(dis, axis=0)
    new_im = code_vectors[B].reshape(X, Y, Z)


    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

