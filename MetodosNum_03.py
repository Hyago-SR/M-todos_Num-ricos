import numpy as np
import matplotlib.pyplot as plt

# Definindo a função f(t, y)
def f(t, y):
    return -y + 2 * np.cos(t)

# Solução exata
def exact_solution(t):
    return np.sin(t) + np.cos(t)

# Método de Euler
def euler_method(f, y0, t, h):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return y

# Método de Taylor
def taylor_method(f, f1, y0, t, h):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i] + h * f(t[i], y[i]) + (h**2 / 2) * f1(t[i], y[i])
    return y

# Método de Euler Melhorado
def improved_euler_method(f, y0, t, h):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h * k1 / 2)
        y[i + 1] = y[i] + h * k2
    return y

# Método de Runge-Kutta de 2ª ordem
def runge_kutta_2(f, y0, t, h):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h, y[i] + h * k1)
        y[i + 1] = y[i] + (h / 2) * (k1 + k2)
    return y

# Método de Runge-Kutta de 3ª ordem
def runge_kutta_3(f, y0, t, h):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + (h / 2) * k1)
        k3 = f(t[i] + h, y[i] + h * k2 - h * k1)
        y[i + 1] = y[i] + (h / 6) * (k1 + 4 * k2 + k3)
    return y

# Método de Runge-Kutta de 4ª ordem
def runge_kutta_4(f, y0, t, h):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y

# Método Previsor-Corretor
def predictor_corrector_method(f, y0, t, h, tol, kmax):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i] + h * f(t[i], y[i])
        for k in range(kmax):
            y0_new = y[i] + (h / 2) * (f(t[i], y[i]) + f(t[i + 1], y[i + 1]))
            e = abs((y[i + 1] - y0_new) / y[i + 1])
            if e < tol:
                y[i + 1] = y0_new
                break
            y[i + 1] = y0_new
    return y

# Método de Passos Múltiplos de 2ª ordem
def multi_step_2(f, y0, t, h):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    y[1] = y[0] + h * f(t[0], y[0])
    for i in range(1, n - 1):
        y[i + 1] = y[i] + (h / 2) * (3 * f(t[i], y[i]) - f(t[i - 1], y[i - 1]))
    return y

# Método de Passos Múltiplos de 4ª ordem
def multi_step_4(f, y0, t, h):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    y[1] = y[0] + h * f(t[0], y[0])
    y[2] = y[1] + (h / 2) * (3 * f(t[1], y[1]) - f(t[0], y[0]))
    y[3] = y[2] + (h / 2) * (3 * f(t[2], y[2]) - f(t[1], y[1]))
    for i in range(3, n - 1):
        l1 = 55 * f(t[i], y[i])
        l2 = -59 * f(t[i - 1], y[i - 1])
        l3 = 37 * f(t[i - 2], y[i - 2])
        l4 = -9 * f(t[i - 3], y[i - 3])
        y[i + 1] = y[i] + (h / 24) * (l1 + l2 + l3 + l4)
    return y

# Parâmetros
h_values = [0.25, 0.125]
y0 = 1
t_max = 5  # Intervalo de tempo para a solução
tol = 0.01
kmax = 10

# Loop sobre os valores de h
for h in h_values:
    t = np.arange(0, t_max + h, h)
    exact = exact_solution(t)

    methods = {
        "Euler": euler_method(f, y0, t, h),
        "Taylor": taylor_method(f, f, y0, t, h),
        "Euler Melhorado": improved_euler_method(f, y0, t, h),
        "Runge-Kutta 2": runge_kutta_2(f, y0, t, h),
        "Runge-Kutta 3": runge_kutta_3(f, y0, t, h),
        "Runge-Kutta 4": runge_kutta_4(f, y0, t, h),
        "Previsor-Corretor": predictor_corrector_method(f, y0, t, h, tol, kmax),
        "Passos Múltiplos 2": multi_step_2(f, y0, t, h),
        "Passos Múltiplos 4": multi_step_4(f, y0, t, h)
    }

    # print(f"h = {h}")
    for method_name, y in methods.items():
        error = np.abs((y - exact) / exact) * 100
        print(f"Método: {method_name} | h = {h}")
        print("="*51)
        print("|t\t |y_numerico\t |y_exato\t |erro (%)")
        print("="*51)
        for i in range(len(t)):
            print(f"|{t[i]:.2f}\t |{y[i]:.6f}\t |{exact[i]:.6f}\t |{error[i]:.6f}")
        print("="*51)
        print("\n")

        # Plotando os resultados
        plt.plot(t, y, label=f'{method_name} (h={h})')

plt.plot(t, exact, label='Solução Exata', linestyle='--')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('Comparação entre Diferentes Métodos e Solução Exata')
plt.grid(True)
plt.show()
