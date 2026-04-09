import numpy as np

# Parameters
m = 72
n = 256

S = 8


sigma = 1/3.0 * np.sqrt(S/float(m))

# X creation
X = np.random.randn(m, n)

n_col = np.linalg.norm(X, axis=0)
X = np.dot(X,np.diag(1/n_col))    # Normalization per column [Get rid of it for the "To go further" part!]

# theta creation
theta = np.zeros(n)
non_null = np.random.choice(n, S)
theta[non_null] = np.random.randn(S)


# y creation
y = np.dot(X,theta) + sigma*np.random.randn(m)
