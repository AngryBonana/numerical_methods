import sys
import numpy as np
from fractions import Fraction


def _lu_decomposition(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = a.shape[0]
    u = a.copy().astype(Fraction)
    l = np.identity(n, dtype=Fraction)
    p = np.identity(n, dtype=Fraction)
    swaps = 0
    
    for k in range(n - 1):
        p_i = k + np.abs(u[k:, k]).argmax(axis=0)
        val = u[p_i, k]
        
        if val == Fraction(0):
            raise ValueError("A matrix is singular! Can't do LU decomposition!")
        
        if k != p_i:
            swaps += 1
            u[[k, p_i]] = u[[p_i, k]]
            p[[k, p_i]] = p[[p_i, k]]
            l[[k, p_i], :k] = l[[p_i, k], :k]
        
        for i in range(k+1, n):
            l[i, k] = u[i, k] / u[k, k]
        
            for j in range(k, n):
                u[i, j] = u[i, j] - l[i, k] * u[k, j]
    
    return l, u, p, swaps


def _forward(l: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = l.shape[0]
    y = np.zeros(n, dtype=Fraction)
    for i in range(n):
        y[i] = b[i] - l[i, :i] @ y[:i]
    return y


def _backward(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = u.shape[0]
    x = np.zeros(n, dtype=Fraction)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - u[i, i+1:] @ x[i+1:]) / u[i, i]
    return x


def _solve_lu(l: np.ndarray, u: np.ndarray, p: np.ndarray, b: np.ndarray):
    b_perm = p @ b
    y = _forward(l, b_perm)
    x = _backward(u, y)
    return x
    

def _reverse_from_lu(l: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
    n = l.shape[0]
    a_rev = np.zeros((n, n), dtype=Fraction)
    for j in range(n):
        e_j = np.zeros(n, dtype=Fraction)
        e_j[j] = Fraction(1)
        x_j = _solve_lu(l, u, p, e_j)
        a_rev[:, j] = x_j
    
    return a_rev
        
    
def _det_from_lu(u: np.ndarray, nswaps: int = 0):
    n = u.shape[0]
    det = Fraction(1)
    for i in range(n):
        det *= u[i, i]
    return det if nswaps % 2 == 0 else -det
    
    
def lu_decomposition(a: np.ndarray, b: np.ndarray, return_p=False, return_a_reversed=False, return_det_a=False)-> tuple[np.ndarray, np.ndarray, np.ndarray ,np.ndarray, np.ndarray, Fraction]:
    l, u, p, swaps = _lu_decomposition(a)
    x = _solve_lu(l, u, p, b)
    res = [x, l, u]
    if return_p:
        res.append(p)
    
    if return_a_reversed:
        res.append(_reverse_from_lu(l, u, p))
        
    if return_det_a:
        res.append(_det_from_lu(u, swaps))
    
    return tuple(res)
    
    


def read_input(filename: str) -> tuple[np.ndarray, np.ndarray]:
    with open(filename) as inp:
        data = inp.readlines()
        data = [line.split() for line in data]
        n = len(data)
        a = list()
        b = list()
        for i in range(n):
            for j in range(n + 1):
                if j == n:
                    b.append(Fraction(data[i][j]))
                else:
                    a.append(Fraction(data[i][j]))
        return np.array(a).reshape((n, n)), np.array(b)


def print_arr(arr: np.ndarray):
    nd = arr.ndim
    if nd == 1:
        print('[\n  ' + '\n  '.join(str(x) for x in arr) + '\n\t]')
    elif nd == 2:
        for line in arr:
            print('[ ' + ' '.join(str(x) for x in line) + ' ]')
    else:
        raise ValueError('Too big ndim for arr. 1 or 2 supported.')

        
def main():
    filename = sys.argv[1]
    a, b = read_input(filename)
    x, l, u, p, a_reversed, det_a = lu_decomposition(a, b, return_p=True, return_a_reversed=True, return_det_a=True)
    
    lu = l @ u
    
    print('L matrix:')
    print_arr(l)
    print('U matrix:')
    print_arr(u)
    
    print('LU matrix:')
    print_arr(lu)
    
    print('x vector:')
    print_arr(x)
    
    print('A^-1 matrix:')
    print_arr(a_reversed)
    
    print(f'det A: {det_a}')
    
    print('A * A^-1:')
    aa_rev = a @ a_reversed
    print_arr(aa_rev)
    print(f'A * A^-1 = E: {(aa_rev == np.identity(aa_rev.shape[0])).all()}')
    
    pa = p @ a
    print(f'LU = PA: {(lu == pa).all()}')
    
    print(f'Ax=b: {(a @ x == b).all()}')
    
    
    

if __name__ == '__main__':
    main()