from math import pow, log, exp


def newton(f, f_der,
                x_0: float, eps: float = 1e-6,
                max_iter: int = 100,
                return_iter: bool = False) -> tuple[float, int]:
    
    x_prev = x_0
    for i in range(max_iter):
        
        x_next = x_prev - f(x_prev) / f_der(x_prev)
        
        if abs(x_next - x_prev) < eps:
            if return_iter:
                return x_next, i + 1
            return x_next
        
        x_prev = x_next
    
    raise ValueError("Solution diverges!")
        

def simple_iterations(x_func,
                        x_0: float, eps: float = 1e-6,
                        max_iter: int = 100,
                        return_iter: bool = False) -> tuple[float, int]:
    
    x_prev = x_0
    for i in range(max_iter):
        x_next = x_func(x_prev)
            
        if abs(x_next - x_prev) < eps:
            if return_iter:
                return x_next, i + 1
            return x_next
    
        x_prev = x_next
        
    raise ValueError("Solution diverges!")


def f(x: float) -> float:
    return log(x + 1) - pow(x, 3) + 1


def f_der(x: float) -> float:
    return 1.0 / (x + 1) - 3 * pow(x, 2)


def x_func_right(t: float) -> float:
    return pow(log(t + 1) + 1, 1/3)

def x_func_left(t: float) -> float:
    return exp(pow(t, 3) - 1) - 1


def main():
    eps = 1e-9
    
    print('Newton method:')
    x1, i1 = newton(f, f_der, 1.5, eps, return_iter=True)
    print(f'x2: {x1}\nIterations: {i1}\n')
    
    print('Simple iterations method:')
    x2, i2 = simple_iterations(x_func_right, 1.5, eps, return_iter=True)
    print(f'x2: {x2}\nIterations: {i2}\n')

if __name__ == "__main__":
    main()