import numpy as np
import sys
from fractions import Fraction


def tridiagonal_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_count = lambda a_i, b_i, c_i, p_i_prev: -c_i / (b_i + a_i * p_i_prev)
    q_count = lambda a_i, b_i, d_i, p_i_prev, q_i_prev: (d_i - a_i * q_i_prev) / (b_i + a_i * p_i_prev)
    x_count = lambda p_i, q_i, x_i_next: p_i * x_i_next + q_i
    
    n = len(a)
    p = np.array([Fraction(0)] * n)
    q = np.array([Fraction(0)] * n)
    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]
    for i in range(1, n):
        if i == n - 1:
            p[i] = 0
        else:
            p[i] = p_count(a[i], b[i], c[i], p[i - 1])
        
        q[i] = q_count(a[i], b[i], d[i], p[i - 1], q[i - 1])
    
    x = np.array([Fraction(0)] * n)
    x[n - 1] = q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = x_count(p[i], q[i], x[i + 1])
        
    return p, q, x
    
        

def read_input(filename: str) -> np.ndarray:
    with open(filename) as input:
        data = input.read().split()
        arr = np.array([x for x in map(Fraction, data)]).reshape((-1, 4))
        return arr

def print_arr(arr: np.ndarray) -> None:
    print('[')
    for x in arr:
        print(f'  {x}')
    print('\t]')

def main():
    filename = sys.argv[1]
    data = read_input(filename)
    a = data[:, 0]
    print(a)
    b = data[:, 1]
    c = data[:, 2]
    d = data[:, 3]
    
    p, q, x = tridiagonal_algorithm(a, b, c, d)
    
    print('P coefs:')
    print_arr(p)
    print('Q coefs:')
    print_arr(q)
    print('Solution:')
    print_arr(x)


if __name__ == '__main__':
    main()