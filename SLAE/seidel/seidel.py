import sys
import os
import numpy as np
import time


def gauss_seidel(a: np.ndarray, b: np.ndarray, eps: float = 1e-6, max_iter=1000, return_iter_count: bool =False) ->tuple[np.ndarray, int]:
    arr, p = find_permutation(a.copy())
    n = arr.shape[0]
    b_perm = p @ b
    c = (-arr.T / np.diag(arr)).T
    d = b_perm / np.diag(arr)
    for i in range(n):
        c[i, i] = float(0)
        
    x = d.copy()
    x_old = None
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        x_old = x.copy()
        for i in range(n):
            x[i] = (c[i, :i] * x[:i]).sum() + (c[i, i:] * x_old[i:]).sum() + d[i]
        if np.isnan(x).any():
                    print(f'Iteration: {iter_count}')
                    print_arr(x_old)
                    print_arr(x)
                    raise ValueError('Nan in vector!')
        
        diff = x - x_old
        if np.max(np.abs(diff)) < eps:
            if return_iter_count:
                return x, iter_count
            return x
    print_arr(x_old)
    print_arr(x)
    raise ValueError("The solution diverges!")
    
    

def find_permutation(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = arr.shape[0]
    p = np.identity(n, dtype=float)
    for i in range(n):
        if arr[i, i] == 0:
            swap_exist = False
            for j in range(n):
                if j == i:
                    continue
                if arr[j, i] != 0 and arr[i, j] != 0:
                    arr[[i, j], :] = arr[[j, i], :]
                    p[[i, j], :] = p[[j, i], :]
                    
                    swap_exist = True
                    break
            if not swap_exist:
                raise ValueError("Can't to nonzero diagonal!")
    
    return arr, p
            

def simple_iteration(a: np.ndarray, b: np.ndarray, eps: float = 1e-6, max_iter=1000, return_iter_count: bool =False) ->tuple[np.ndarray, int]:
    arr, p = find_permutation(a.copy())
    n = arr.shape[0]
    b_perm = p @ b
    c = (-arr.T / np.diag(arr)).T
    d = b_perm / np.diag(arr)
    
    for i in range(n):
        c[i, i] = float(0)
    
    iter_count = 0
    x_prev = d.copy()
    x_next = None
    convergence = False
    while iter_count < max_iter:
        iter_count += 1
        x_next = c @ x_prev + d
        if np.isnan(x_next).any():
            print(f'Iteration: {iter_count}')
            print_arr(x_prev)
            print_arr(x_next)
            raise ValueError('Nan in vector!')
        diff = x_next - x_prev
        
        if np.max(np.abs(diff)) < eps: # max-норма
            convergence = True
            break
        x_prev = x_next
    
    if not convergence:
        print_arr(x_prev)
        print_arr(x_next)
        raise ValueError("The solution diverges!")
    
    
    if return_iter_count:
        return x_next, iter_count
    
    return x_next
    


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
                    b.append(float(data[i][j]))
                else:
                    a.append(float(data[i][j]))
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
    eps = 1e-6
    a, b = read_input(sys.argv[1])
    
    start_time = time.perf_counter()
    x_true = np.linalg.solve(a, b) # LU decomposition in numpy
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f'Eps = {eps}')
    print('x true:')
    print_arr(x_true)
    print(f'Time: {elapsed_time:.6f}\n')
    
    start_time = time.perf_counter()
    x, i = simple_iteration(a, b, max_iter=1e4, eps=1e-6, return_iter_count=True)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f'Simple Iterations: {i}')
    print_arr(x)
    print(f'Accuracy: {(np.abs(x - x_true) < eps).all()}')
    print(f'Time: {elapsed_time:.6f}\n')
    
    start_time = time.perf_counter()
    x, i = gauss_seidel(a, b, max_iter=1e4, eps=1e-6, return_iter_count=True)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f'Seidel Iterations: {i}')
    print_arr(x)
    print(f'Accuracy: {(np.abs(x - x_true) <= eps).all()}')
    print(f'Time: {elapsed_time:.6f}')


if __name__ == '__main__':
    main()