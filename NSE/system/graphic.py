import matplotlib.pyplot as plt
import numpy as np


def make_graphic(a: float, b: float, dots: int = 1000) -> None:
    if a > b:
        raise ValueError("Incorrect borders! a > b")

    t = np.linspace(a, b, dots)

    plt.plot(np.cos(t) / 4, t, label='cos(x2) = 4·x1')

    plt.plot(t, np.exp(t) / 4, label='4·x2 = exp(x1)')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
if __name__ == '__main__':
    make_graphic(0, 1)


