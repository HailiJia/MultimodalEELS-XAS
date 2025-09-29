import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ---------------------------
# PCA
# ---------------------------
def reduce_pca(X, n_components=10, scale=True, random_state=0):
    """
    PCA reduction. Optionally standardize features before PCA.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    n_components : int or float
        Number of components (int) or variance ratio (float in (0,1]).
    scale : bool
        If True, standardize features to zero mean / unit variance.
    random_state : int
        Random state for PCA (affects SVD initialization).

    Returns
    -------
    X_red : ndarray, shape (n_samples, n_components)
    pca   : fitted PCA object
    scaler: fitted StandardScaler or None
    """
    X = np.asarray(X)
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_red = pca.fit_transform(X)
    return X_red, pca, scaler


def pca_inverse_transform(X_red, pca, scaler=None):
    """
    Map PCA-reduced data back to the original feature space.
    """
    X_rec = pca.inverse_transform(X_red)
    if scaler is not None:
        # invert standardization
        X_rec = scaler.inverse_transform(X_rec)
    return X_rec


def plot_pca_variance(pca, title="PCA explained variance"):
    """
    Plot explained variance ratio and cumulative curve.
    """
    evr = np.asarray(pca.explained_variance_ratio_)
    cum = np.cumsum(evr)

    plt.figure()
    plt.plot(np.arange(1, len(evr) + 1), evr, marker='o', label='per-component')
    plt.plot(np.arange(1, len(cum) + 1), cum, marker='o', label='cumulative')
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


# ---------------------------
# t-SNE (2D/3D embeddings)
# ---------------------------
def reduce_tsne(X, n_components=2, perplexity=30.0, learning_rate='auto',
                n_iter=1000, random_state=0):
    """
    t-SNE embedding.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    n_components : int
        Target dimension (2 or 3 common).
    perplexity : float
    learning_rate : float or 'auto'
    n_iter : int
    random_state : int

    Returns
    -------
    X_emb : ndarray, shape (n_samples, n_components)
    tsne  : fitted TSNE object (holds params; no transform method)
    """
    X = np.asarray(X)
    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                init="pca",
                random_state=random_state)
    X_emb = tsne.fit_transform(X)
    return X_emb, tsne



# ---------------------------
# Generic plotting helpers
# ---------------------------
def plot_embedding_2d(X_emb, labels=None, title="2D embedding", alpha=0.9):
    """
    Scatter plot for 2D embeddings (PCA/t-SNE/UMAP).
    """
    X_emb = np.asarray(X_emb)
    assert X_emb.shape[1] == 2, "X_emb must be 2D (n_samples, 2)"

    plt.figure()
    if labels is None:
        plt.scatter(X_emb[:, 0], X_emb[:, 1], s=12, alpha=alpha)
    else:
        labels = np.asarray(labels)
        for lab in np.unique(labels):
            m = labels == lab
            plt.scatter(X_emb[m, 0], X_emb[m, 1], s=12, alpha=alpha, label=str(lab))
        plt.legend()
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.title(title)
    plt.tight_layout()


def plot_embedding_3d(X_emb, labels=None, title="3D embedding", alpha=0.9):
    """
    Scatter plot for 3D embeddings.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

    X_emb = np.asarray(X_emb)
    assert X_emb.shape[1] == 3, "X_emb must be 3D (n_samples, 3)"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if labels is None:
        ax.scatter(X_emb[:, 0], X_emb[:, 1], X_emb[:, 2], s=12, alpha=alpha)
    else:
        labels = np.asarray(labels)
        for lab in np.unique(labels):
            m = labels == lab
            ax.scatter(X_emb[m, 0], X_emb[m, 1], X_emb[m, 2], s=12, alpha=alpha, label=str(lab))
        ax.legend()
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")
    ax.set_title(title)
    plt.tight_layout()


if __name__ == "__main__":
    # Put selected featurized X using featurization.py

    # PCA
    X_pca, pca, scaler = reduce_pca(X, n_components=10, scale=True, random_state=0)
    plot_pca_variance(pca, title="PCA variance (demo)")

    # t-SNE (on PCA-reduced)
    X_tsne, _ = reduce_tsne(X_pca, n_components=2, perplexity=30.0, random_state=0)
    plot_embedding_2d(X_tsne, labels=y, title="t-SNE on PCA (demo)")

    plt.show()
