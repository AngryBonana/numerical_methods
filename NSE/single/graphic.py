import matplotlib.pyplot as plt
import numpy as np
from newton import f

def make_graphic(a: float, b: float, func, dots: int = 100) -> None:
    if a > b:
        raise ValueError("Incorrect borders! a > b")
    x = np.linspace(a, b, dots)
    vect_f = np.vectorize(func)
    y = vect_f(x)
    ox = np.zeros(x.size)
    plt.plot(x, y)
    plt.plot(x, ox)
    plt.grid(True)
    plt.title('Графическое решение уравнения')
    plt.show()
    
    
if __name__ == "__main__":
    make_graphic(-0.9, 5, f)
    