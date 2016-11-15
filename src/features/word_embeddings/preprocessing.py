"""
Contains methods for preprocessing features.
"""

from sklearn.decomposition import PCA


def apply_pca(matrix, **kwargs):
    """
    Applies pca to given matrix
    :param matrix: a real-number matrix (in numpy format)
    :param kwargs: keyword arguments ("components" - number of components in result matrix - is optional)
    """
    if "components" in kwargs:
        pca = PCA(n_components=kwargs["components"])
    else:
        pca = PCA()
    pca.fit(matrix)
    return pca.transform(matrix)
