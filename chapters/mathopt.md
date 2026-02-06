## MathOpt як шар моделювання

<a name="chapters-mathopt"></a>

Нещодавно Google OR-Tools представив новий шар моделювання під назвою
[MathOpt](https://developers.google.com/optimization/math_opt). Він надає
агностичний до розв’язувачів API для формулювання та розв’язання задач
математичної оптимізації, зокрема лінійних програм (LP) і змішаних
цілочисельних програм (MIP). Він задуманий як простіша і сучасніша альтернатива
старішому інтерфейсу `pywraplp`, з більшим фокусом на зручність використання та
продуктивність.

Це важливо, адже багато оптимізаційних задач природно формулюються як LP або MIP,
і CP-SAT не завжди є правильним інструментом для них. Як уже обговорювали,
працювати з неперервними змінними та коефіцієнтами в CP-SAT не дуже зручно, адже
їх потрібно ретельно дискретизувати. З MathOpt ви можете напряму використовувати
неперервні змінні та коефіцієнти з плаваючою комою, якщо обираєте розв’язувач,
який їх підтримує (наприклад, HiGHS або Gurobi). Для певних типів задач
LP/MIP-розв’язувачі також можуть бути значно ефективнішими за CP-SAT. Моделюючи
задачу в MathOpt, ви легко перемикаєтеся між різними розв’язувачами та можете
подивитися, який найкраще підходить саме вам.

Який підвох під час використання MathOpt? Він додає невеликий оверхед у порівнянні
з прямим викликом нативного API розв’язувача й наразі підтримує лише лінійні
обмеження — тобто ті, які ви додаєте в CP-SAT через метод `add`, виключаючи
оператори `!=`, `<` і `>`. Сам MathOpt підтримує неперервні та необмежені змінні,
а також коефіцієнти з плаваючою комою, але ці можливості недоступні, коли ви
обираєте CP-SAT як бекенд. Якщо ваша модель містить неперервні змінні, ви
отримаєте помилку, адже CP-SAT підтримує лише цілочисельні змінні та
коефіцієнти.

Якщо ви хочете порівняти продуктивність CP-SAT з MIP-розв’язувачем, вам і надалі
потрібно дискретизувати неперервні змінні та коефіцієнти, коли цього вимагає
задача. На щастя, багато практичних задач є суто цілочисельними, тож на практиці
ви можете навіть не помітити цього обмеження.

> [!NOTE]
>
> Інша мова моделювання, орієнтована на спільноту програмування з обмеженнями, —
> [CPMpy](https://cpmpy.readthedocs.io/en/latest/index.html), яка напряму
> підтримує CP-SAT як бекенд-розв’язувач. Натомість інструменти моделювання,
> поширені в спільноті математичної оптимізації — такі як
> [Pyomo](https://pyomo.org/), [GAMS](https://www.gams.com/) та
> [AMPL](https://ampl.com/) — не підтримують CP-SAT.

Далі наведено короткий огляд API MathOpt.

### Імпорт і створення моделі

Як і в CP-SAT, спершу потрібно імпортувати модуль MathOpt і створити екземпляр
моделі.

```python
from ortools.math_opt.python import mathopt as mo

model = mo.Model(name="optional_name")
```

### Змінні

Далі можна почати додавати змінні до моделі. MathOpt підтримує неперервні,
булеві та цілочисельні змінні. За замовчуванням неперервні та цілочисельні
змінні є невід’ємними (тобто мають нижню межу 0). Ви можете задати інші межі
через параметри `lb` та `ub`.

```python
x = model.add_variable(lb=0.0, ub=10.0, name="x")  # Неперервна змінна в [0, 10]
# На відміну від CP-SAT, імена змінним задавати не обов’язково
nameless_x = model.add_variable(lb=0.0, ub=10.0)  # Змінна без імені
x_unbounded = model.add_variable(lb=-math.inf, ub=math.inf, name="x_unbounded")  # Необмежена неперервна змінна
```

Щоб створити булеві або цілочисельні змінні, використовуйте відповідні методи:

```python
y = model.add_binary_variable(name="y")  # Булева змінна з {0, 1}
z = model.add_integer_variable(lb=0, ub=100, name="z")  # Цілочисельна змінна в [0, 100]
# Цілочисельна змінна без верхньої межі
z_unbounded = model.add_integer_variable(name="z_unbounded")
# або явно
z_unbounded = model.add_integer_variable(lb=0, ub=math.inf, name="z_unbounded")
```

> [!WARNING]
>
> Нижня межа за замовчуванням дорівнює 0 для всіх змінних, а верхня межа —
> нескінченність для неперервних і цілочисельних змінних. Якщо вам потрібна
> змінна, що може набувати від’ємних значень, потрібно явно задати нижню межу
> `-math.inf` або інше від’ємне число.

### Обмеження

Лінійні обмеження можна додавати в модель через метод `add_linear_constraint`.
Є два способи: форма нерівності/рівності та «boxed» форма.

```python
# Форма нерівності/рівності з використанням операторів Python
model.add_linear_constraint(2 * x + 3 * y <= 10, name="constraint1")
model.add_linear_constraint(x + y + z == 5, name="constraint2")
model.add_linear_constraint(z - 2 * x >= 0, name="constraint3")

# Boxed форма
model.add_linear_constraint(lb=0, expr=2 * x + 3 * y, ub=10, name="constraint1_boxed")
model.add_linear_constraint(lb=5, expr=x + y + z, ub=5, name="constraint2_boxed")  # рівність
model.add_linear_constraint(lb=0, expr=z - 2 * x, ub=math.inf, name="constraint3_boxed")  # нерівність

# Обмеження не обов’язково мають бути іменованими
model.add_linear_constraint(4 * x + y <= 20)
```

Якщо потрібно сумувати список термінів, можна використовувати звичайну суму в
Python, як і в CP-SAT. Але MathOpt надає більш ефективну функцію `mo.fast_sum`,
яка оптимізована під продуктивність.

```python
terms = [i * x for i in range(1, 11)]  # Створюємо список термінів
model.add_linear_constraint(mo.fast_sum(terms) <= 100, name="sum_constraint")
```

Також можна використовувати «хендли» виразів, що робить модель читабельнішою та
простішою в підтримці.

```python
in_vars = [mo.variable(name=f"in_{i}") for i in range(1, 6)]
out_vars = [mo.variable(name=f"out_{i}") for i in range(1, 6)]
incoming_flow = mo.fast_sum(in_vars)
outgoing_flow = mo.fast_sum(out_vars)
model.add_linear_constraint(incoming_flow == outgoing_flow, name="flow_conservation")
```

### Цільова функція

Цільову функцію можна задати через метод `set_objective` і вказати, чи потрібно
максимізувати або мінімізувати.

```python
model.set_objective(3 * x + 4 * y + 2 * z, is_maximize=True)  # Максимізувати
model.set_objective(x + 2 * z, is_maximize=False)  # Мінімізувати
```

### Розв’язання

Щоб розв’язати модель, використовуйте функцію `mo.solve`, яка приймає модель,
тип розв’язувача та необов’язкові параметри розв’язувача.

```python
params = mo.SolveParameters(
    time_limit=timedelta(seconds=30),  # Ліміт часу 30 секунд
    relative_gap_tolerance=0.01,  # 1% відносний допуск за розривом
    enable_output=True  # Увімкнути вивід розв’язувача
)
result = mo.solve(model, solver_type=mo.SolverType.HIGHS, params=params)
```

Наразі MathOpt підтримує такі розв’язувачі: `GSCIP`, `GUROBI`, `GLOP`,
`CP_SAT`, `PDLP`, `GLPK`, `OSQP`, `ECOS`, `SCS`, `HIGHS` та `SANTORINI`. Якщо у
вас немає ліцензії Gurobi, рекомендую використовувати HiGHS або GSCIP — обидва
це open-source MIP-розв’язувачі.

### Перегляд результатів

Після розв’язання моделі можна переглянути результати через об’єкт `SolveResult`,
який повертає `mo.solve`.

Перевіряємо причину завершення:

```python
term = result.termination
print(f"Termination: {term.reason.name}")
if term.detail:
    print(f"Detail: {term.detail}")
```

Перевіряємо, чи знайдено допустиме первинне рішення:

```python
if result.has_primal_feasible_solution():
    print(f"Objective value: {result.objective_value()}")
    values = result.variable_values()
    print(f"x: {values.get(x)}, y: {values.get(y)}, z: {values.get(z)}")
else:
    print("No primal feasible solution found.")
```

Метод `variable_values()` повертає відображення зі змінних на їхні значення.
Також можна передати список змінних, щоб отримати значення в тому ж порядку, що
зручно, якщо потрібна лише підмножина.

```python
values = result.variable_values([x, y, z])
print(f"x: {values[0]}, y: {values[1]}, z: {values[2]}")
```

> [!NOTE]
>
> Якщо ви розв’язуєте LP і базовий розв’язувач підтримує двоїсті значення, ви
> також можете отримати двоїсті значення обмежень за допомогою методу
> `dual_values()` об’єкта `SolveResult`. Пам’ятайте, що в такому разі потрібно
> зберігати хендл на обмеження під час створення, наприклад,
> `constraint = model.add_linear_constraint(...)`. Існують і додаткові
> можливості, як-от callbacks і ліниві обмеження, які ми тут не розглядаємо.
> Однак у
> [прикладах](https://github.com/google/or-tools/tree/stable/ortools/math_opt/samples/python)
> можна знайти гарні кейси.

### Приклади

Розгляньмо два приклади, що демонструють, як моделювати та розв’язувати задачі
оптимізації за допомогою MathOpt.

#### Спрощена задача дієти Стиглера

**Спрощена задача дієти Стиглера** — це класична оптимізаційна задача, де потрібно
вибрати невід’ємні порції різних продуктів, щоб мінімізувати загальну вартість і
водночас задовольнити мінімальні харчові вимоги за калоріями, білками та кальцієм.
Ця модель слугує невеликим ілюстративним прикладом застосування математичної
оптимізації в плануванні харчування. Одиниці виміру та значення тут наведено лише
для демонстрації і їх не слід сприймати як дієтичні факти.

```python
from ortools.math_opt.python import mathopt as mo
from datetime import timedelta
# --- Дані -----------------------------------------------------------------
foods = ["Wheat Flour", "Milk", "Cabbage", "Beef"]

# Вартість на порцію (EUR)
cost = {
    "Wheat Flour": 0.36,
    "Milk": 0.23,
    "Cabbage": 0.10,
    "Beef": 1.20,
}

# Поживні речовини на порцію (приблизно / ілюстративно)
#               Calories  Protein(g)  Calcium(mg)
calories =     {"Wheat Flour": 364.0, "Milk": 150.0, "Cabbage": 25.0,  "Beef": 250.0}
protein =      {"Wheat Flour": 10.0,  "Milk": 8.0,   "Cabbage": 1.3,   "Beef": 26.0}
calcium =      {"Wheat Flour": 15.0,  "Milk": 285.0, "Cabbage": 40.0,  "Beef": 20.0}

# Мінімальні вимоги
req = {
    "Calories": 2000.0,   # kcal
    "Protein": 55.0,      # g
    "Calcium": 800.0,     # mg
}

# --- Модель ---------------------------------------------------------------
model = mo.Model(name="stigler_diet")

# Рішення: порції кожного продукту (неперервні, ≥ 0)
# За бажанням можна перейти на цілі змінні, щоб отримати MIP-варіант.
servings = {f: model.add_variable(lb=0.0, name=f"servings[{f}]") for f in foods}

# Необов’язкові верхні межі, щоб результати виглядали охайно
for f in foods:
    model.add_linear_constraint(servings[f] <= 20.0, name=f"cap[{f}]")

# --- Обмеження на поживні речовини ----------------------------------------
model.add_linear_constraint(
    mo.fast_sum(calories[f] * servings[f] for f in foods) >= req["Calories"],
    name="nutrients[Calories]",
)
model.add_linear_constraint(
    mo.fast_sum(protein[f] * servings[f] for f in foods) >= req["Protein"],
    name="nutrients[Protein]",
)
model.add_linear_constraint(
    mo.fast_sum(calcium[f] * servings[f] for f in foods) >= req["Calcium"],
    name="nutrients[Calcium]",
)

# --- Ціль: мінімізувати загальну вартість ---------------------------------
model.set_objective(
    mo.fast_sum(cost[f] * servings[f] for f in foods),
    is_maximize=False,
)

# --- Розв’язання -----------------------------------------------------------
params = mo.SolveParameters(
    time_limit=timedelta(seconds=10),
    relative_gap_tolerance=1e-6,  # LP, тож можемо бути строгими
    enable_output=False,
)

result = mo.solve(model, solver_type=mo.SolverType.HIGHS, params=params)

# --- Звіт ------------------------------------------------------------------
term = result.termination
print(f"Termination: {term.reason.name}")

if not result.has_primal_feasible_solution():
    print("No feasible solution found.")
    return

print(f"Minimum cost: €{result.objective_value():.2f}")

# Витягуємо вибрані порції (ігноруємо майже нульові)
values = result.variable_values()
print("\nServings:")
for f in foods:
    qty = values.get(servings[f], 0.0)
    if abs(qty) > 1e-9:
        print(f"  {f:12s}: {qty:8.3f}")

# Обчислюємо фактичні підсумки
tot_cal = sum(calories[f] * values.get(servings[f], 0.0) for f in foods)
tot_pro = sum(protein[f]  * values.get(servings[f], 0.0) for f in foods)
tot_calcium = sum(calcium[f] * values.get(servings[f], 0.0) for f in foods)

print("\nNutrient totals (minimum required in parentheses):")
print(f"  Calories: {tot_cal:.1f}  ({req['Calories']})")
print(f"  Protein : {tot_pro:.1f}  ({req['Protein']})")
print(f"  Calcium : {tot_calcium:.1f}  ({req['Calcium']})")
```

#### Покриття множинами

**Задача покриття множинами (Set Cover)** — це класична комбінаторна
оптимізаційна задача. Дано універсум елементів $U$ і набір підмножин
$S \subseteq 2^U$, кожна з яких має цілочисельну вартість; потрібно вибрати
мінімально-вартісну колекцію підмножин так, щоб кожен елемент з $U$ був
покритий принаймні однією обраною підмножиною.

Ця задача трапляється в розподілі ресурсів, плануванні та проєктуванні мереж.
Стандартне цілочисельне формулювання таке:

- **Змінні:** для кожної підмножини $S$ бінарна змінна $z[S] \in \{0, 1\}$
  показує, чи обрано підмножину $S$.
- **Обмеження:** для кожного елемента $u \in U$ сума $z[S]$ по всіх підмножинах
  $S$, що містять $u$, має бути не меншою за 1.
- **Ціль:** мінімізувати загальну вартість $\sum_{S} \text{cost}[S] \cdot z[S]$.

Оскільки всі змінні цілочисельні, цю задачу можна моделювати у CP-SAT і напряму
порівнювати з MIP-розв’язувачами, такими як Gurobi або HiGHS.

```python
from ortools.math_opt.python import mathopt as mo
from datetime import timedelta

U = {1, 2, 3, 4, 5, 6}
subsets = {
    "S1": {1, 2, 3},
    "S2": {2, 4},
    "S3": {3, 5, 6},
    "S4": {1, 4, 6},
    "S5": {2, 5},
    "S6": {4, 5, 6},
}
cost = {"S1": 4, "S2": 2, "S3": 3, "S4": 3, "S5": 2, "S6": 4}

model = mo.Model(name="set_cover_cp_sat")

# Змінні рішення
z = {s: model.add_binary_variable(name=f"pick[{s}]") for s in subsets}

# Обмеження покриття
for u in U:
    model.add_linear_constraint(
        mo.fast_sum(z[s] for s in subsets if u in subsets[s]) >= 1,
        name=f"cover[{u}]",
    )

# Ціль
model.set_objective(
    mo.fast_sum(cost[s] * z[s] for s in subsets),
    is_maximize=False,
)

# Розв’язання з CP-SAT
params = mo.SolveParameters(time_limit=timedelta(seconds=10), enable_output=True)
result = mo.solve(model, solver_type=mo.SolverType.CP_SAT, params=params)

print(f"[SetCover] Termination: {result.termination.reason.name}")
if not result.has_primal_feasible_solution():
    print("[SetCover] No feasible solution found.")
    return

vals = result.variable_values()
chosen = [s for s in subsets if int(round(vals.get(z[s], 0.0)))]
total_cost = sum(cost[s] for s in chosen)
print(f"[SetCover] Minimum total cost: {total_cost}")
print("[SetCover] Chosen subsets:", ", ".join(chosen))
```
