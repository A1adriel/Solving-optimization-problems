import numpy as np
from scipy.optimize import linprog

def solve_lp(c, A_ub, b_ub, description):
    print(f"\n--- {description} ---")
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)], method='highs')
    print("Успех:", result.success)
    print("x =", result.x)
    print("Максимум z =", -result.fun)

# Кейс 1: Пекарня
c1 = np.array([-7, -5])
A1 = np.array([[4, 2], [1, 2]])
b1 = np.array([40, 22])
solve_lp(c1, A1, b1, "Муниципальная пекарня")

# Кейс 2: Ремонтный участок
c2 = np.array([-8, -7])
A2 = np.array([[3, 2], [2, 3]])
b2 = np.array([24, 30])
solve_lp(c2, A2, b2, "Городской ремонтный участок")

# Кейс 3: Полевые рационы
c3 = np.array([-10, -7])
A3 = np.array([[3, 1], [1, 2]])
b3 = np.array([21, 14])
solve_lp(c3, A3, b3, "Комплектование полевых рационов")

# Кейс 4: Ремонтный склад
c4 = np.array([-11, -8])
A4 = np.array([[2, 3], [3, 1]])
b4 = np.array([24, 21])
solve_lp(c4, A4, b4, "Ремонтно-обслуживающий склад")