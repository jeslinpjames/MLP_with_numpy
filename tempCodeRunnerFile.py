def init_params():
    #Hidden Layer parameters
    W1 = np.random.randn(10, 784) * np.sqrt(2/784)
    b1 = np.zeros((10, 1))

    #Output layer parameters
    W2 = np.random.randn(10, 10) * np.sqrt(2/10)
    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2