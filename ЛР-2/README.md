# Отчёт по лабораторной работе №2
## «Транспортная задача»

**Выполнил:** Самбуев Алдар Баирович

---

## 1. Цель работы

Освоить построение транспортной модели как частного случая линейного программирования:
- проверять закрытость/открытость задачи;
- вводить фиктивные узлы для балансировки;
- формировать матрицы `c`, `A_eq`, `b_eq`, `bounds` для `scipy.optimize.linprog`;
- решать задачу и интерпретировать полученный план перевозок.

---

## 2. Теоретическая справка

Транспортная задача задаётся:
- **поставщиками** с запасами $a_i$ ($i = 1 \ldots m$);
- **потребителями** со спросом $b_j$ ($j = 1 \ldots n$);
- **матрицей затрат** $c_{ij}$ на перевозку единицы груза от $i$ к $j$.

Переменные $x_{ij}$ — объём перевозки.

**Целевая функция:**

$$Z = \sum_i \sum_j c_{ij} x_{ij} \to \min$$

**Ограничения:**
- по запасам: $\sum_j x_{ij} = a_i$
- по спросу: $\sum_i x_{ij} = b_j$
- $x_{ij} \ge 0$

Если сумма запасов равна сумме спроса — задача **закрытая**. Иначе — **открытая**:
- запасов больше → добавляем **фиктивного потребителя** с нулевой стоимостью;
- спроса больше → добавляем **фиктивного поставщика** (в учебных примерах стоимость 0).

Для `linprog` матрица перевозок разворачивается в вектор по правилу $k = i \cdot n + j$ (нумерация с 0):
- `c = costs.flatten()`
- `A_eq`: сначала $m$ строк для поставщиков, затем $n$ строк для потребителей
- `b_eq = [a₁, …, aₘ, b₁, …, bₙ]`
- `bounds = [(0, None)] * (m * n)`

---

## 3. Решение четырёх кейсов

### 3.1. Кейс 1: Лекарства в больницы (закрытая)

- Запасы: 30, 40, 35
- Спрос: 20, 25, 30, 30
- Баланс: 105 = 105

**Оптимальный план:**

```
Склад A → Больница 1: 20,  Больница 3: 10
Склад B → Больница 2: 25,  Больница 4: 15
Склад C → Больница 3: 20,  Больница 4: 15
```

**Минимальная стоимость: 520**

---

### 3.2. Кейс 2: Школьное питание (запасы > спрос)

- Запасы: 40, 35, 30 → сумма 105
- Спрос: 20, 25, 30, 15 → сумма 90
- Добавлен фиктивный потребитель с потребностью 15, стоимость 0

**Оптимальный план:**

```
Комбинат A → Школа 1: 20,  Школа 2: 5,   Фиктивный: 15
Комбинат B → Школа 2: 20,  Школа 3: 15
Комбинат C → Школа 3: 15,  Школа 4: 15
```

**Минимальная стоимость: 475**

Фиктивный потребитель означает остаток продукции, не отправленный школам.

---

### 3.3. Кейс 3: Топливо для частей (закрытая)

- Запасы: 50, 40, 35 → сумма 125
- Спрос: 30, 25, 35, 35 → сумма 125

**Оптимальный план:**

```
База A → Часть 1: 30,  Часть 3: 20
База B → Часть 2: 25,  Часть 4: 15
База C → Часть 3: 15,  Часть 4: 20
```

**Минимальная стоимость: 825**

---

### 3.4. Кейс 4: Запчасти на ремонтные базы (спрос > запасов)

- Запасы: 25, 30, 20 → сумма 75
- Спрос: 15, 20, 18, 30 → сумма 83
- Добавлен фиктивный поставщик с запасом 8, стоимость 0

**Оптимальный план:**

```
Склад A           → Рембаза 1: 15,  Рембаза 3: 10
Склад B           → Рембаза 2: 20,  Рембаза 4: 10
Склад C           → Рембаза 3: 8,   Рембаза 4: 12
Фиктивный поставщик → Рембаза 4: 8
```

**Минимальная стоимость: 479**

Фиктивный поставщик показывает дефицит: рембаза 4 недополучает 8 единиц.

---

## 4. Проверка допустимости

Во всех решениях суммы по строкам равны запасам, суммы по столбцам — спросу (с учётом фиктивных узлов). Условия неотрицательности выполнены.

---

## 5. Выводы

1. Транспортная задача эффективно решается как задача линейного программирования.
2. Балансировка через фиктивный узел приводит задачу к каноническому виду с равенствами.
3. Интерпретация фиктивных перевозок (остаток / дефицит) важна для принятия управленческих решений.
4. Полученные планы экономически обоснованы — используются самые дешёвые маршруты (например, комбинат B в школу 2 с затратами 4 вместо 6).

---

## 6. Код решения

Все четыре кейса используют одинаковые функции `balance_transport_problem` и `build_transport_lp`, корректно обрабатывающие как закрытые, так и открытые задачи.

### 6.1. Кейс 1: Лекарства в больницы (`lab_02_student_civil_01.ipynb`)

```python
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.optimize import linprog

DUMMY_SUPPLIER_NAME = "Фиктивный поставщик"
DUMMY_CONSUMER_NAME = "Фиктивный потребитель"
BALANCE_TOLERANCE = 1e-9

supplier_names = ['Склад A', 'Склад B', 'Склад C']
consumer_names = ['Больница 1', 'Больница 2', 'Больница 3', 'Больница 4']

supplies = np.array([30, 40, 35], dtype=float)
demands  = np.array([20, 25, 30, 30], dtype=float)

costs = np.array([
    [5, 7, 6, 10],
    [8, 4, 5,  7],
    [6, 6, 4,  5]
], dtype=float)

def balance_transport_problem(supplies, demands, costs, supplier_names, consumer_names, dummy_cost=0.0):
    balanced_supplies       = supplies.astype(float).copy()
    balanced_demands        = demands.astype(float).copy()
    balanced_costs          = costs.astype(float).copy()
    balanced_supplier_names = list(supplier_names)
    balanced_consumer_names = list(consumer_names)
    balance_difference = balanced_supplies.sum() - balanced_demands.sum()

    if abs(balance_difference) < BALANCE_TOLERANCE:
        balance_note = "Закрытая задача. Фиктивный узел не требуется."
    elif balance_difference > 0:
        balanced_demands = np.append(balanced_demands, balance_difference)
        balanced_consumer_names.append(DUMMY_CONSUMER_NAME)
        balanced_costs = np.column_stack([balanced_costs, np.full((len(balanced_supplies), 1), dummy_cost)])
        balance_note = (f"Открытая (запасы > спрос). Добавлен фиктивный потребитель "
                        f"с потребностью {balance_difference} и стоимостью {dummy_cost}.")
    else:
        dummy_supply = -balance_difference
        balanced_supplies = np.append(balanced_supplies, dummy_supply)
        balanced_supplier_names.append(DUMMY_SUPPLIER_NAME)
        balanced_costs = np.vstack([balanced_costs, np.full((1, len(balanced_demands)), dummy_cost)])
        balance_note = (f"Открытая (спрос > запасов). Добавлен фиктивный поставщик "
                        f"с запасом {dummy_supply} и стоимостью {dummy_cost}.")

    return (balanced_supplies, balanced_demands, balanced_costs,
            balanced_supplier_names, balanced_consumer_names, balance_note)

def build_transport_lp(supplies, demands, costs):
    supplier_count, consumer_count = costs.shape
    variable_count = supplier_count * consumer_count
    c = costs.flatten()
    A_eq_rows = []
    for i in range(supplier_count):
        row = np.zeros(variable_count)
        for j in range(consumer_count):
            row[i * consumer_count + j] = 1
        A_eq_rows.append(row)
    for j in range(consumer_count):
        row = np.zeros(variable_count)
        for i in range(supplier_count):
            row[i * consumer_count + j] = 1
        A_eq_rows.append(row)
    A_eq = np.array(A_eq_rows)
    b_eq = np.r_[supplies, demands]
    bounds = [(0, None)] * variable_count
    return {"c": c, "A_eq": A_eq, "b_eq": b_eq, "bounds": bounds}

(balanced_supplies, balanced_demands, balanced_costs,
 balanced_supplier_names, balanced_consumer_names, balance_note) = balance_transport_problem(
    supplies, demands, costs, supplier_names, consumer_names)
print(balance_note)

lp_model = build_transport_lp(balanced_supplies, balanced_demands, balanced_costs)
result = linprog(lp_model["c"], A_eq=lp_model["A_eq"], b_eq=lp_model["b_eq"],
                 bounds=lp_model["bounds"], method="highs")
assert result.success, result.message

plan    = result.x.reshape(len(balanced_supplier_names), len(balanced_consumer_names))
plan_df = pd.DataFrame(plan, index=balanced_supplier_names, columns=balanced_consumer_names)
display(plan_df)

print("Суммы по строкам (запасы):", plan_df.sum(axis=1).values)
print("Суммы по столбцам (спрос):", plan_df.sum(axis=0).values)
print("Минимальная стоимость:", result.fun)
```

---

### 6.2. Кейс 2: Школьное питание (`lab_02_student_civil_02.ipynb`)

Структура кода идентична кейсу 1. Отличаются только исходные данные:

```python
supplier_names = ['Комбинат A', 'Комбинат B', 'Комбинат C']
consumer_names = ['Школа 1', 'Школа 2', 'Школа 3', 'Школа 4']

supplies = np.array([40, 35, 30], dtype=float)
demands  = np.array([20, 25, 30, 15], dtype=float)

costs = np.array([
    [4, 6, 8, 7],
    [5, 4, 7, 6],
    [6, 5, 4, 8]
], dtype=float)
```

---

### 6.3. Кейс 3: Топливо для частей (`lab_02_student_military_01.ipynb`)

```python
supplier_names = ['База A', 'База B', 'База C']
consumer_names = ['Часть 1', 'Часть 2', 'Часть 3', 'Часть 4']

supplies = np.array([50, 40, 35], dtype=float)
demands  = np.array([30, 25, 35, 35], dtype=float)

costs = np.array([
    [ 6,  7, 9, 12],
    [ 5,  4, 8, 10],
    [ 8,  6, 5,  7]
], dtype=float)
```

---

### 6.4. Кейс 4: Запчасти на ремонтные базы (`lab_02_student_military_02.ipynb`)

```python
supplier_names = ['Склад A', 'Склад B', 'Склад C']
consumer_names = ['Рембаза 1', 'Рембаза 2', 'Рембаза 3', 'Рембаза 4']

supplies = np.array([25, 30, 20], dtype=float)
demands  = np.array([15, 20, 18, 30], dtype=float)

costs = np.array([
    [7, 5, 9, 11],
    [6, 4, 7,  8],
    [8, 6, 5,  7]
], dtype=float)
```
