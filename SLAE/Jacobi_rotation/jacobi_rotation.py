import sys
import numpy as np

def find_max_non_diag(matrix: np.ndarray) -> tuple[int, int]:
    n = matrix.shape[0]
    idx, jdx = 0, 1
    max_el = abs(matrix[0, 1])
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i, j]) > max_el:
                max_el = abs(matrix[i, j])
                idx, jdx = i, j
    return idx, jdx
    

def off_diagonal_norm(a: np.ndarray) -> float:
    return np.sqrt(np.sum(a ** 2) - np.sum(np.diagonal(a) ** 2))
    

def jacobi_rotation(a: np.ndarray, eps=1e-6, max_iter: int = 1000, return_iter_count=False) -> tuple[np.ndarray, np.ndarray]:
    if (a != a.T).any():
        raise ValueError("Matrix is not symmetric! Can't use jacobi rotation method!")
    
    n = a.shape[0]
    q = np.identity(n, dtype=float)
    a = a.copy()
    
    iter_count = 0
    done = False
    
    while iter_count < max_iter:
        iter_count += 1
        if off_diagonal_norm(a) < eps:
            done = True
            break
        
        p_idx, q_idx = find_max_non_diag(a)
        a_pq = a[p_idx, q_idx]
        
        tau = (a[q_idx, q_idx] - a[p_idx, p_idx]) / (2 * a_pq)
        t = np.sign(tau) / (abs(tau) + np.sqrt(1 + tau**2)) if tau != 0 else 1.0
        c = 1 / np.sqrt(1 + t**2)
        s = t * c
        
        j = np.identity(n)
        j[p_idx, p_idx] = c
        j[q_idx, q_idx] = c
        j[p_idx, q_idx] = s
        j[q_idx, p_idx] = -s
        
        a = j.T @ a @ j
        q = q @ j
    
    if not done:
        raise ValueError("The solution diverges!")
    
    eigenvalues = np.diag(a)
    eigenvectors = q
    
    res = [eigenvalues, eigenvectors]
    if return_iter_count:
        res.append(iter_count)
    
    return tuple(res)
    
    


def read_input(filename: str) -> np.ndarray:
    with open(filename) as inp:
        data = inp.readlines()
        a = list()
        for line in data:
            a.append([float(x) for x in line.split()])
        
        return np.array(a)
        
    

def main():
    a = read_input(sys.argv[1])
    eps = 1e-6
    print(f'Matrix A:\n{a}')
    vals, vectors, iters = jacobi_rotation(a, eps=eps, max_iter=1000, return_iter_count=True)
    
    print(f'Eigen values:\n{vals}')
    
    print(f'Matrix of eigen vectors:\n{vectors}')
    
    print(f'Amount of iterations: {iters}')
    print('A·V = V·Λ:', ((a @ vectors - vectors @ np.diag(vals)) < eps).all())
    
    
    
if __name__ == '__main__':
    main()