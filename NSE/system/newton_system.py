from math import sin, cos, exp


def newton(x1:float, x2:float, eps:float = 1e-6, max_iter:int = 100, return_iter=False) -> tuple[tuple[float, float], int]:
    df1_dx1 = 4
    df1_dx2 = lambda x2: sin(x2)
    df2_dx1 = lambda x1: - exp(x1)
    df2_dx2 = 4
    
    x1_prev = x1
    x2_prev = x2
    for i in range(max_iter):
        detJ = df1_dx1 * df2_dx2 - df1_dx2(x2_prev) * df2_dx1(x1_prev)
        detA1 = f1(x1_prev, x2_prev) * df2_dx2 - f2(x1_prev, x2_prev) * df1_dx2(x2_prev)
        detA2 = df1_dx1 * f2(x1_prev, x2_prev) - df2_dx1(x1_prev) * f1(x1_prev, x2_prev)
        
        x1_new = x1_prev - detA1 / detJ
        x2_new = x2_prev - detA2 / detJ
        
        if max(abs(x1_new - x1_prev), abs(x2_new - x2_prev)) < eps:
            if return_iter:
                return (x1_new, x2_new), i + 1
            return (x1_new, x2_new)
                
        x1_prev = x1_new
        x2_prev = x2_new
    
    raise ValueError("Solution diverges!")


def simple_iterations(x1:float, x2:float, eps:float = 1e-6, max_iter:int = 100, return_iter=False) -> tuple[tuple[float, float], int]:
    x1_func = lambda x2: cos(x2) / 4
    x2_func = lambda x1: exp(x1) / 4
    
    x1_prev = x1
    x2_prev = x2
    for i in range(max_iter):
        x1_new = x1_func(x2_prev)
        x2_new = x2_func(x1_prev)
        
        if max(abs(x1_new - x1_prev), abs(x2_new - x2_prev)) < eps:
            if return_iter:
                return (x1_new, x2_new), i + 1
            return (x1_new, x2_new)
        
        x1_prev = x1_new
        x2_prev = x2_new
    
    raise ValueError("Solution diverges!")




def f1(x1: float, x2: float) -> float:
    return 4 * x1 - cos(x2)

def f2(x1: float, x2: float) -> float:
    return 4 * x2 - exp(x1)

def main():
    eps = 1e-9
    sol1, i1 = simple_iterations(0.2, 0.4, eps, return_iter=True)
    sol2, i2 = newton(0.2, 0.4, eps, return_iter=True)
    print(f'Simple iterations solution: {sol1}\nIteration: {i1}\n')
    print(f'Newton method solution: {sol2}\nIteration: {i2}')
    

if __name__ == '__main__':
    main()