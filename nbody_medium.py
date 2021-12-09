def getAcc(pos, mass, G, softening):
    """
    Calculate the acceleration on each particle due to Newton's Law
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """

    N = pos.shape[0];
    a = np.zeros((N, 3));

    for i in range(N):
        for j in range(N):
            dx = pos[j, 0] - pos[i, 0];
            dy = pos[j, 1] - pos[i, 1];
            dz = pos[j, 2] - pos[i, 2];
            inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2) ** (-1.5);
            a[i, 0] += G * (dx * inv_r3) * mass[j];
            a[i, 1] += G * (dy * inv_r3) * mass[j];
            a[i, 2] += G * (dz * inv_r3) * mass[j];

    return a