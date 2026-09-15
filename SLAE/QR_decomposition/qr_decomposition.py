import numpy as np
import sys
 
 
def euclidian_norm(v: np.ndarray) -> float:
    return np.sqrt(np.sum(v ** 2))

def get_qr(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = a.shape[0]
    q = np.identity(n, dtype=float)
    r = a.copy()
 
    for j in range(n - 1):
        x = r[j:, j].copy()
        norm_x = euclidian_norm(x)
        if norm_x < 1e-15:
            continue
 
        v = x.copy()
        v[0] += np.sign(x[0]) * norm_x if x[0] != 0 else norm_x
        v_norm_sq = np.inner(v, v)
 
        h_sub = np.identity(n - j) - 2 * np.outer(v, v) / v_norm_sq
        h = np.identity(n, dtype=float)
        h[j:, j:] = h_sub
 
        r = h @ r
        q = q @ h
    
    return q, r
 
def qr_step(a_i: np.ndarray) -> np.ndarray:
    q, r = get_qr(a_i)    
    return r @ q
 
 
def extract_eigenvalues(a_i: np.ndarray, n: int, eps: float) -> np.ndarray:

    eigenvalues = []
    j = 0
    while j < n:
        if j == n - 1 or abs(a_i[j + 1, j]) < eps:
            eigenvalues.append(float(a_i[j, j]))
            j += 1
        else:
            a, b = a_i[j, j], a_i[j, j + 1]
            c, d = a_i[j + 1, j], a_i[j + 1, j + 1]
            trace = a + d
            det = a * d - b * c
            discriminant = trace ** 2 - 4 * det
            if discriminant >= 0:
                sq = np.sqrt(discriminant)
                eigenvalues.append(float((trace + sq) / 2))
                eigenvalues.append(float((trace - sq) / 2))
            else:
                sq = np.sqrt(-discriminant)
                eigenvalues.append(complex(trace / 2, sq / 2))
                eigenvalues.append(complex(trace / 2, -sq / 2))
            j += 2
    return np.array(eigenvalues)
 
 
def qr_decomposition(a: np.ndarray, eps: float = 1e-6, max_iter: int = 1000, return_iter_count: bool = False) -> np.ndarray:
    
    n = a.shape[0]
    a_i = a.copy()
    prev_eigenvalues = None
    curr_eigenvalues = None
    iter_count = 0
    done = False
 
    while iter_count < max_iter:
        iter_count += 1
        a_i = qr_step(a_i)
 
        curr_eigenvalues = extract_eigenvalues(a_i, n, eps)
 
        if prev_eigenvalues is not None:
            diff = np.max(np.abs(
                np.sort_complex(curr_eigenvalues) - np.sort_complex(prev_eigenvalues)
            ))
            if diff < eps:
                done = True
                break
 
        prev_eigenvalues = curr_eigenvalues
 
    if not done:
        raise ValueError("The solution diverges!")
 
    if return_iter_count:
        return curr_eigenvalues, iter_count
    return curr_eigenvalues
 
 
def read_input(filename: str) -> np.ndarray:
    with open(filename) as inp:
        data = inp.readlines()
        a = list()
        for line in data:
            a.append([float(x) for x in line.split()])
        return np.array(a)
 
 
def main():
    eps = 1e-6
    a = read_input(sys.argv[1])
    print(f'Matrix A:\n{a}\n')
    print ('Example of QR decomposition')
    q, r = get_qr(a)
    print(f'Matrix Q:\n{q}')
    print(f'Matrix R:\n{r}')
    print(f'Q * R = A: {(np.abs(q @ r - a) < eps).all()}\n')
    
    vals, i = qr_decomposition(a, eps=eps, max_iter=1000, return_iter_count=True)
    print(f'Eigen vals:\n{vals}')
    print(f'Amount of iterations: {i}\n')
    np_eigvals = np.linalg.eigvals(a)
    print(f'numpy check: {np_eigvals}')
    print(f'Accuracy: {(np.abs(np.sort_complex(vals) - np.sort_complex(np_eigvals)) < eps).all()}')
 
 
if __name__ == '__main__':
    main()
