<a name="04B-advanced-modelling"></a>

## Просунуте моделювання

<!-- START_SKIP_FOR_README -->

![Обкладинка «Моделювання»](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/logo_complex_assembly.webp)

<!-- STOP_SKIP_FOR_README -->

Після знайомства з базовими елементами CP-SAT у цьому розділі ми перейдемо до
складніших обмежень. Вони вже орієнтовані на конкретні задачі — наприклад,
маршрутизацію чи планування — але всередині свого домену дуже універсальні й
потужні. Водночас вони потребують більш детального пояснення щодо правильного
використання.

- [Обмеження турів](#04-modelling-circuit): `add_circuit`,
  `add_multiple_circuit`, `add_reservoir_constraint_with_active`
- [Інтервали](#04-modelling-intervals): `new_interval_var`,
  `new_interval_var_series`, `new_fixed_size_interval_var`,
  `new_optional_interval_var`, `new_optional_interval_var_series`,
  `new_optional_fixed_size_interval_var`,
  `new_optional_fixed_size_interval_var_series`,
  `add_no_overlap`,`add_no_overlap_2d`, `add_cumulative`
- [Обмеження автомата](#04-modelling-automaton): `add_automaton`
- [Обмеження резервуара](#04-modelling-reservoir): `add_reservoir_constraint`
- [Кусково-лінійні обмеження](#04-modelling-pwl): офіційно не частина CP-SAT,
  але ми надаємо безкоштовний код для копіювання.

<a name="04-modelling-circuit"></a>

### Обмеження circuit/турів

Маршрути й тури важливі для розв’язання оптимізаційних задач у багатьох сферах,
далеко за межами класичної маршрутизації. Наприклад, у секвенуванні ДНК
оптимізація порядку збирання фрагментів критична, а в наукових дослідженнях
методичне впорядкування переналаштувань експериментів може суттєво зменшити
операційні витрати та простої. Обмеження `add_circuit` і `add_multiple_circuit`
в CP-SAT дозволяють легко моделювати різні сценарії. Вони виходять за межі
класичної
[задачі комівояжера (TSP)](https://en.wikipedia.org/wiki/Travelling_salesman_problem),
дозволяючи розв’язки, де не потрібно відвідувати всі вершини, а також
підтримуючи кілька неперетинних підтурів. Така адаптивність робить їх
неоціненними для широкого кола практичних задач, де порядок і організація
операцій критично впливають на ефективність і результат.

|                         ![TSP Example](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/optimal_tsp.png)                         |
| :-------------------------------------------------------------------------------------------------------------------------------------------------: |
| Задача комівояжера (TSP) шукає найкоротший маршрут, який відвідує кожну вершину рівно один раз і повертається до стартової вершини. |

Задача комівояжера — одна з найвідоміших і найкраще досліджених комбінаторних
оптимізаційних задач. Це класичний приклад задачі, яку легко зрозуміти,
поширеної на практиці, але складної для розв’язання. Вона також займає особливе
місце в історії оптимізації, адже багато загальних технік спершу були розроблені
саме для TSP. Якщо ще не робили цього, рекомендую переглянути
[цю доповідь Bill Cook](https://www.youtube.com/watch?v=5VjphFYQKj8) або навіть
прочитати книгу
[In Pursuit of the Traveling Salesman](https://press.princeton.edu/books/paperback/9780691163529/in-pursuit-of-the-traveling-salesman).

> [!TIP]
>
> Якщо ваша задача — саме TSP, можливо, вам буде корисним
> [розв’язувач Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html).
> Для задач, близьких до TSP, більш відповідним може бути MIP-розв’язувач,
> оскільки багато варіантів TSP дають сильні лінійні релаксації, які MIP
> ефективно використовують. Також зверніть увагу на
> [OR-Tools Routing](https://developers.google.com/optimization/routing), якщо
> маршрутизація — значна частина вашої задачі. Але коли варіанти TSP — лише
> компонент більшої задачі, CP-SAT із `add_circuit` або `add_multiple_circuit`
> може бути дуже корисним.

|                                                                                                                                                        ![TSP BnB Example](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/tsp_bnb_improved.png)                                                                                                                                                         |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Цей приклад показує, чому MIP-розв’язувачі такі сильні для TSP. Лінійна релаксація (вгорі) вже дуже близька до оптимального розв’язку. Розгалужуючись, тобто пробуючи 0 і 1, лише для двох дробових змінних, ми не лише знаходимо оптимальний розв’язок, а й доводимо оптимальність. Приклад згенеровано за допомогою [DIY TSP Solver](https://www.math.uwaterloo.ca/tsp/D3/bootQ.html). |

#### `add_circuit`

Обмеження `add_circuit` використовується для розв’язання задач про цикли у
орієнтованих графах і навіть дозволяє петлі. Воно приймає список трійок
`(u,v,var)`, де `u` і `v` — початкова і кінцева вершини, а `var` — булева
змінна, яка показує, чи включене ребро в розв’язок. Обмеження гарантує, що
ребра з `True` формують один цикл, який відвідує кожну вершину рівно один раз,
за винятком вершин із петлею, встановленою в `True`. Індекси вершин мають
починатися з 0 і не можуть мати пропусків, інакше це призведе до ізоляції та
недопустимості циклу.

Ось приклад використання CP-SAT для орієнтованої задачі комівояжера:

```python
from ortools.sat.python import cp_model

# Орієнтований граф із вагами ребер
dgraph = {(0, 1): 13, (1, 0): 17, ...(2, 3): 27}

# Ініціалізуємо модель CP-SAT
model = cp_model.CpModel()

# Булеві змінні для кожного ребра
edge_vars = {(u, v): model.new_bool_var(f"e_{u}_{v}") for (u, v) in dgraph.keys()}

# Обмеження circuit для одного туру
model.add_circuit([(u, v, var) for (u, v), var in edge_vars.items()])

# Цільова функція — мінімізувати сумарну вартість
model.minimize(sum(dgraph[(u, v)] * x for (u, v), x in edge_vars.items()))

# Розв’язуємо модель
solver = cp_model.CpSolver()
status = solver.solve(model)
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    tour = [(u, v) for (u, v), x in edge_vars.items() if solver.value(x)]
    print("Tour:", tour)

# Output: [(0, 1), (2, 0), (3, 2), (1, 3)], тобто 0 -> 1 -> 3 -> 2 -> 0
```

Це обмеження можна адаптувати для шляхів, додавши віртуальне ребро, яке
замикає шлях у цикл, наприклад `(3, 0, 1)` для шляху від вершини 0 до вершини 3.

#### Креативне використання `add_circuit`

`add_circuit` можна творчо адаптувати для різних споріднених задач. Хоча для
задачі найкоротшого шляху існують ефективніші алгоритми, покажімо, як
адаптувати `add_circuit` в освітніх цілях.

```python
from ortools.sat.python import cp_model

# Задаємо зважений орієнтований граф із вартістю ребер
dgraph = {(0, 1): 13, (1, 0): 17, ...(2, 3): 27}

source_vertex = 0
target_vertex = 3

# Додаємо нульові петлі для вершин, які не є джерелом або ціллю
for v in [1, 2]:
    dgraph[(v, v)] = 0

# Ініціалізуємо модель CP-SAT і змінні
model = cp_model.CpModel()
edge_vars = {(u, v): model.new_bool_var(f"e_{u}_{v}") for (u, v) in dgraph}

# Визначаємо цикл із псевдоребром від цілі до джерела
circuit = [(u, v, var) for (u, v), var in edge_vars.items()] + [
    (target_vertex, source_vertex, 1)
]
model.add_circuit(circuit)

# Мінімізуємо сумарну вартість
model.minimize(sum(dgraph[(u, v)] * x for (u, v), x in edge_vars.items()))

# Розв’язуємо та отримуємо шлях
solver = cp_model.CpSolver()
status = solver.solve(model)
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    path = [(u, v) for (u, v), x in edge_vars.items() if solver.value(x) and u != v]
    print("Path:", path)

# Output: [(0, 1), (1, 3)], тобто 0 -> 1 -> 3
```

Цей підхід демонструє гнучкість `add_circuit` для різних задач турів і шляхів.
Ось ще приклад:

- [Budget constrained tours](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_circuit_budget.py):
  оптимізація найбільшого можливого туру в межах заданого бюджету.

#### `add_multiple_circuit`

Для задач із кількома поїздками з депо можна використати `add_multiple_circuit`.
Це обмеження схоже на `add_circuit`, але явно дозволяє відвідувати депо кілька
разів. Як і `add_circuit`, `add_multiple_circuit` підтримує опційні вершини
через петлі.

Це особливо корисно для задач маршрутизації транспорту (VRP), де кілька турів
починаються з одного депо. Зазвичай VRP має додаткові обмеження, бо інакше
повернення в депо без потреби є неоптимальним. Альтернативою є дублювання графа
та застосування `add_circuit` на кожній копії, але `add_multiple_circuit`
дозволяє уникнути копіювання графа і створення кількох наборів змінних, залишаючи
єдиний набір змінних та ребер.

Недоліком методу є те, що деякі обмеження, наприклад заборона відвідувати дві
вершини в одному турі, стають складнішими, адже всі тури спільно використовують
змінні. Водночас багато обмежень усе ще можна ефективно моделювати, наприклад
обмеження місткості в задачі CVRP. CVRP — класична задача в операційному
дослідженні та логістиці: потрібно знайти найкоротший набір маршрутів для
флоту однакових транспортів, що стартують і завершують у єдиному депо (це може
бути і той самий транспорт, що робить кілька поїздок). Кожного клієнта треба
відвідати рівно один раз, з обмеженням, що сумарний попит на кожному турі не
перевищує місткість транспортного засобу.

Нижче наведено приклад реалізації CVRP із `add_multiple_circuit` і додатковою
змінною для відстеження місткості на кожній вершині.

```python
from typing import Hashable
import networkx as nx
from ortools.sat.python import cp_model


class CvrpMultiCircuit:
    """CVRP через multi-circuit обмеження CP-SAT."""

    def __init__(
        self,
        graph: nx.Graph,
        depot: Hashable,
        capacity: int,
        demand_label: str = "demand",
        model: cp_model.CpModel | None = None,
    ):
        self.graph, self.depot = graph, depot
        self.model = model or cp_model.CpModel()
        self.capacity = capacity
        self.demand_label = demand_label

        # Список вершин з депо на початку
        self.vertices = [depot] + [v for v in graph.nodes() if v != depot]
        self.index = {v: i for i, v in enumerate(self.vertices)}

        # Булеві змінні дуг для обох напрямків
        self.arc_vars = {
            (i, j): self.model.new_bool_var(f"arc_{i}_{j}")
            for u, v in graph.edges
            for i, j in ((self.index[u], self.index[v]), (self.index[v], self.index[u]))
        }
        arcs = [(i, j, var) for (i, j), var in self.arc_vars.items()]

        # Обмеження multi-circuit
        self.model.add_multiple_circuit(arcs)

        # Змінні місткості та обмеження
        self.cap_vars = [
            self.model.new_int_var(0, capacity, f"cap_{i}")
            for i in range(len(self.vertices))
        ]
        for i, j, var in arcs:
            if j == 0:
                continue
            demand = graph.nodes[self.vertices[j]].get(demand_label, 0)
            self.model.add(
                self.cap_vars[j] >= self.cap_vars[i] + demand
            ).only_enforce_if(var)

    def is_arc_used(self, u, v) -> cp_model.BoolVarT:
        return self.arc_vars[(self.index[u], self.index[v])]

    def weight(self, label: str = "weight") -> cp_model.LinearExprT:
        return sum(
            var * self.graph[self.vertices[i]][self.vertices[j]][label]
            for (i, j), var in self.arc_vars.items()
        )

    def minimize_weight(self, label: str = "weight"):
        self.model.minimize(self.weight(label=label))

    def extract_tours(self, solver: cp_model.CpSolver) -> list[list]:
        # Будуємо орієнтований граф обраних дуг
        dg = nx.DiGraph(
            [
                (self.vertices[i], self.vertices[j])
                for (i, j), var in self.arc_vars.items()
                if solver.value(var)
            ]
        )

        # Ейлерів цикл і розбиття за депо
        euler = nx.eulerian_circuit(dg, source=self.depot)
        tours, curr = [], [self.depot]
        for u, v in euler:
            curr.append(v)
            if v == self.depot:
                tours.append(curr)
                curr = [self.depot]
        if len(curr) > 1:
            tours.append(curr)
        return tours
```

|                                                                                                                 ![CVRP Example](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/cvrp_example.png)                                                                                                                  |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Задача CVRP шукає найкоротші маршрути, що відвідують кожну вершину рівно один раз і повертаються до стартової вершини. Депо є початком і кінцем кожного туру. Граф у прикладі зважений приблизною геометричною відстанню, а місткість транспортного засобу встановлена на 15. |

> [!WARNING]
>
> Хоча `add_multiple_circuit` дозволяє додаткові LNS-стратегії та може
> покращувати нижні межі, стандартна формулювання Miller-Tucker-Zemlin (MTZ)
> іноді ефективніша для CVRP. Обидва підходи кращі за використання `add_circuit`
> на кількох копіях графа. Реалізації всіх трьох підходів моделювання CVRP є
> [тут](https://github.com/d-krupke/cpsat-primer/blob/main/examples/cvrp/).

#### Продуктивність `add_circuit` для TSP

Таблиця нижче показує продуктивність CP-SAT на різних інстансах TSPLIB із
використанням `add_circuit` та лімітом 90 секунд. Продуктивність можна вважати
достатньою, але MIP-розв’язувач легко її перевершує.

| Instance | # vertices | runtime | lower bound | objective | opt. gap |
| :------- | ---------: | ------: | ----------: | --------: | -------: |
| att48    |         48 |    0.47 |       33522 |     33522 |        0 |
| eil51    |         51 |    0.69 |         426 |       426 |        0 |
| st70     |         70 |     0.8 |         675 |       675 |        0 |
| eil76    |         76 |    2.49 |         538 |       538 |        0 |
| pr76     |         76 |   54.36 |      108159 |    108159 |        0 |
| kroD100  |        100 |    9.72 |       21294 |     21294 |        0 |
| kroC100  |        100 |    5.57 |       20749 |     20749 |        0 |
| kroB100  |        100 |     6.2 |       22141 |     22141 |        0 |
| kroE100  |        100 |    9.06 |       22049 |     22068 |        0 |
| kroA100  |        100 |    8.41 |       21282 |     21282 |        0 |
| eil101   |        101 |    2.24 |         629 |       629 |        0 |
| lin105   |        105 |    1.37 |       14379 |     14379 |        0 |
| pr107    |        107 |     1.2 |       44303 |     44303 |        0 |
| pr124    |        124 |    33.8 |       59009 |     59030 |        0 |
| pr136    |        136 |   35.98 |       96767 |     96861 |        0 |
| pr144    |        144 |   21.27 |       58534 |     58571 |        0 |
| kroB150  |        150 |   58.44 |       26130 |     26130 |        0 |
| kroA150  |        150 |   90.94 |       26498 |     26977 |       2% |
| pr152    |        152 |   15.28 |       73682 |     73682 |        0 |
| kroA200  |        200 |   90.99 |       29209 |     29459 |       1% |
| kroB200  |        200 |   31.69 |       29437 |     29437 |        0 |
| pr226    |        226 |   74.61 |       80369 |     80369 |        0 |
| gil262   |        262 |   91.58 |        2365 |      2416 |       2% |
| pr264    |        264 |   92.03 |       49121 |     49512 |       1% |
| pr299    |        299 |   92.18 |       47709 |     49217 |       3% |
| linhp318 |        318 |   92.45 |       41915 |     52032 |      19% |
| lin318   |        318 |   92.43 |       41915 |     52025 |      19% |
| pr439    |        439 |   94.22 |      105610 |    163452 |      35% |

Існують дві основні формулювання для моделювання TSP без `add_circuit`:
[формулювання Данцига—Фалкерсона—Джонсона (DFJ)](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Dantzig%E2%80%93Fulkerson%E2%80%93Johnson_formulation)
та
[формулювання Міллера—Такера—Земліна (MTZ)](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Miller%E2%80%93Tucker%E2%80%93Zemlin_formulation[21]).
DFJ загалом ефективніше через сильнішу лінійну релаксацію. Однак воно потребує
«ледачих» обмежень, які CP-SAT не підтримує. Без них продуктивність DFJ у CP-SAT
порівнянна з MTZ. Попри це, обидва формулювання значно гірші за `add_circuit`.
Це підкреслює перевагу `add_circuit` для турів і шляхів. На відміну від
користувача, `add_circuit` може використовувати lazy constraints всередині,
що дає суттєву перевагу.

<a name="04-modelling-intervals"></a>

### Планування і пакування з інтервалами

Особливий тип змінних — інтервальні змінні, що дозволяють моделювати інтервали,
тобто відрізок певної довжини зі стартом і кінцем. Існують інтервали фіксованої
довжини, змінної довжини та опційні інтервали для різних сценаріїв. Вони
особливо корисні в поєднанні з обмеженнями відсутності перекриття у 1D та 2D.
Це підходить для задач геометричного пакування, планування й інших задач, де
потрібно уникати накладень інтервалів. Ці змінні особливі тим, що фактично це
не змінна, а контейнер, який обмежує окремо задані змінні старту, довжини й
кінця.

Є чотири типи інтервальних змінних: `new_interval_var`,
`new_fixed_size_interval_var`, `new_optional_interval_var` та
`new_optional_fixed_size_interval_var`. `new_optional_interval_var` є
найвиразнішим, але й найдорожчим, тоді як `new_fixed_size_interval_var` —
найпростішим і найефективнішим. Усі типи приймають `start=` змінну. Інтервали
з `fixed_size` вимагають константний `size=`, що задає довжину. Інакше `size=`
може бути змінною в парі з `end=`, що ускладнює розв’язання. Інтервали з
`optional` мають аргумент `is_present=`, булеву змінну, що показує, чи інтервал
присутній. Обмеження no-overlap застосовуються лише до присутніх інтервалів,
що дозволяє моделювати задачі з кількома ресурсами чи опційними задачами. Замість
цілочисельної змінної всі аргументи можуть приймати афінні вирази, наприклад
`start=5*start_var+3`.

```python
model = cp_model.CpModel()

start_var = model.new_int_var(0, 100, "start")
length_var = model.new_int_var(10, 20, "length")
end_var = model.new_int_var(0, 100, "end")
is_present_var = model.new_bool_var("is_present")

# інтервал із довжиною, яку можна змінювати (дорожчий)
flexible_interval = model.new_interval_var(
    start=start_var, size=length_var, end=end_var, name="flexible_interval"
)

# інтервал фіксованої довжини
fixed_interval = model.new_fixed_size_interval_var(
    start=start_var,
    size=10,  # має бути константою
    name="fixed_interval",
)

# опційний інтервал зі змінною довжиною (найдорожчий)
optional_interval = model.new_optional_interval_var(
    start=start_var,
    size=length_var,
    end=end_var,
    is_present=is_present_var,
    name="optional_interval",
)

# опційний інтервал фіксованої довжини
optional_fixed_interval = model.new_optional_fixed_size_interval_var(
    start=start_var,
    size=10,  # має бути константою
    is_present=is_present_var,
    name="optional_fixed_interval",
)
```

Ці інтервальні змінні самі по собі не корисні, адже те саме можна зробити
простими лінійними обмеженнями. Проте CP-SAT має спеціальні обмеження для
інтервалів, які важко моделювати вручну і які значно ефективніші.

CP-SAT пропонує три обмеження для інтервалів:
`add_no_overlap`, `add_no_overlap_2d`, `add_cumulative`. `add_no_overlap`
забороняє перекриття в одному вимірі (наприклад, час). `add_no_overlap_2d`
забороняє перекриття у двох вимірах (наприклад, пакування прямокутників).
`add_cumulative` моделює ресурсне обмеження, де сума попитів перекривних
інтервалів не перевищує місткість ресурсу.

`add_no_overlap` приймає список (опційних) інтервалів і гарантує, що жодні два
присутні інтервали не перекриваються.

```python
model.add_no_overlap(
    interval_vars=[
        flexible_interval,
        fixed_interval,
        optional_interval,
        optional_fixed_interval,
        # ...
    ]
)
```

`add_no_overlap_2d` приймає два списки (опційних) інтервалів і забезпечує, що для
кожної пари `i` та `j` інтервали `x_intervals[i]` і `x_intervals[j]` або
`y_intervals[i]` і `y_intervals[j]` не перекриваються. Отже, обидва списки мають
мати однакову довжину, а `x_intervals[i]` і `y_intervals[i]` вважаються парою.
Якщо `x_intervals[i]` або `y_intervals[i]` опційні, то весь об’єкт є опційним.

```python
model.add_no_overlap_2d(
    x_intervals=[
        flexible_interval,
        fixed_interval,
        optional_interval,
        optional_fixed_interval,
        # ...
    ],
    y_intervals=[
        flexible_interval,
        fixed_interval,
        optional_interval,
        optional_fixed_interval,
        # ...
    ],
)
```

`add_cumulative` використовується для ресурсних обмежень, де сума попитів
перекривних інтервалів не може перевищувати місткість ресурсу. Наприклад,
планування енергоємних машин, коли сумарне споживання не повинно перевищувати
потужність мережі. Обмеження приймає список інтервалів, список попитів і змінну
місткості. Попити мають ту саму довжину, що й інтервали, бо попит зіставляється
за індексом. Оскільки місткість і попити можуть бути змінними (або афінними
виразами), можна моделювати досить складні ресурсні обмеження.

```python
demand_vars = [model.new_int_var(1, 10, f"demand_{i}") for i in range(4)]
capacity_var = model.new_int_var(1, 100, "capacity")
model.add_cumulative(
    intervals=[
        flexible_interval,
        fixed_interval,
        optional_interval,
        optional_fixed_interval,
    ],
    demands=demand_vars,
    capacity=capacity_var,
)
```

> [!WARNING]
>
> Не переходьте одразу до інтервалів у задачах планування. Інтервали корисні,
> коли у вас є більш-менш неперервний час чи простір. Якщо задача більш
> дискретна, наприклад має фіксовану кількість слотів, часто ефективніше
> змоделювати її простими булевими змінними та обмеженнями. Особливо якщо можна
> використати доменні знання, щоб знайти кластери зустрічей, які не можуть
> перекриватися, це може бути значно ефективніше. Якщо планування домінують
> переходи, ваша задача може бути радше маршрутизаційною, і тоді більше підходить
> `add_circuit`.

Розгляньмо кілька прикладів використання цих обмежень.

#### Планування конференц-залу з інтервалами

Припустимо, у нас є конференц-зал і треба запланувати кілька зустрічей. Кожна
зустріч має фіксовану тривалість і діапазон можливих стартів. Слоти — по 5
хвилин від 8:00 до 18:00. Отже, є $10 \times 12 = 120$ слотів, і ми можемо
використовувати просту цілочисельну змінну для старту. Для фіксованих тривалостей
зручно використовувати `new_fixed_size_interval_var`. `add_no_overlap` гарантує
відсутність перекриття, а домени стартових змінних задають можливі часові вікна.

Для обробки даних введемо `namedtuple` для зустрічей і дві функції для
перетворення часу в індекс і назад.

```python
# Конвертуємо час у індекс і назад
def t_to_idx(hour, minute):
    return (hour - 8) * 12 + minute // 5


def idx_to_t(time_idx):
    hour = 8 + time_idx // 12
    minute = (time_idx % 12) * 5
    return f"{hour}:{minute:02d}"


# Опис зустрічі
MeetingInfo = namedtuple("MeetingInfo", ["start_times", "duration"])
```

Створімо кілька зустрічей.

```python
# Опис зустрічей
meetings = {
    "meeting_a": MeetingInfo(
        start_times=[
            [t_to_idx(8, 0), t_to_idx(12, 0)],
            [t_to_idx(16, 0), t_to_idx(17, 0)],
        ],
        duration=120 // 5,  # 2 години
    ),
    "meeting_b": MeetingInfo(
        start_times=[
            [t_to_idx(10, 0), t_to_idx(12, 0)],
        ],
        duration=30 // 5,  # 30 хвилин
    ),
    "meeting_c": MeetingInfo(
        start_times=[
            [t_to_idx(16, 0), t_to_idx(17, 0)],
        ],
        duration=15 // 5,  # 15 хвилин
    ),
    "meeting_d": MeetingInfo(
        start_times=[
            [t_to_idx(8, 0), t_to_idx(10, 0)],
            [t_to_idx(12, 0), t_to_idx(14, 0)],
        ],
        duration=60 // 5,  # 1 година
    ),
}
```

Тепер створимо модель CP-SAT і додамо інтервали та обмеження.

```python
# Створюємо модель CP-SAT
model = cp_model.CpModel()

# Створюємо змінні старту для кожної зустрічі
start_time_vars = {
    meeting_name: model.new_int_var_from_domain(
        cp_model.Domain.from_intervals(meeting_info.start_times),
        f"start_{meeting_name}",
    )
    for meeting_name, meeting_info in meetings.items()
}

# Створюємо інтервали для кожної зустрічі
interval_vars = {
    meeting_name: model.new_fixed_size_interval_var(
        start=start_time_vars[meeting_name],
        size=meeting_info.duration,
        name=f"interval_{meeting_name}",
    )
    for meeting_name, meeting_info in meetings.items()
}

# Гарантуємо, що зустрічі не перекриваються
model.add_no_overlap(list(interval_vars.values()))
```

І нарешті, розв’яжемо модель і витягнемо розклад.

```python
# Розв’язуємо модель
solver = cp_model.CpSolver()
status = solver.solve(model)

# Витягуємо та друкуємо розклад
scheduled_times = {}
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    for meeting_name in meetings:
        start_time = solver.value(start_time_vars[meeting_name])
        scheduled_times[meeting_name] = start_time
        print(f"{meeting_name} starts at {idx_to_t(start_time)}")
else:
    print("No feasible solution found.")
```

Трохи магії з matplotlib — і можемо візуалізувати розклад.

|                ![Schedule](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/scheduling_example.png)                |
| :-----------------------------------------------------------------------------------------------------------------------------------: |
| Можливий неперекривний розклад для цього прикладу. Інстанс простий, але можна спробувати додати ще зустрічей. |

#### Планування для кількох ресурсів з опційними інтервалами

Тепер уявімо, що у нас кілька ресурсів, наприклад кілька конференц-залів, і ми
потрібно запланувати зустрічі так, щоб у межах одного залу вони не перекривалися.
Це можна змоделювати опційними інтервалами, які існують лише якщо зустріч
призначено в конкретний зал. `add_no_overlap` гарантує відсутність перекриття
зустрічей у кожному залі.

Оскільки у нас два зали, зробимо задачу складнішою — інакше розв’язувач міг би
обійтися одним залом. Для цього просто додамо більше і довших зустрічей.

```python
# Опис зустрічей
meetings = {
    "meeting_a": MeetingInfo(
        start_times=[
            [t_to_idx(8, 0), t_to_idx(12, 0)],
            [t_to_idx(16, 0), t_to_idx(16, 0)],
        ],
        duration=120 // 5,
    ),
    "meeting_b": MeetingInfo(
        start_times=[[t_to_idx(10, 0), t_to_idx(12, 0)]], duration=240 // 5
    ),
    "meeting_c": MeetingInfo(
        start_times=[[t_to_idx(16, 0), t_to_idx(17, 0)]], duration=30 // 5
    ),
    "meeting_d": MeetingInfo(
        start_times=[
            [t_to_idx(8, 0), t_to_idx(10, 0)],
            [t_to_idx(12, 0), t_to_idx(14, 0)],
        ],
        duration=60 // 5,
    ),
    "meeting_e": MeetingInfo(
        start_times=[[t_to_idx(10, 0), t_to_idx(12, 0)]], duration=120 // 5
    ),
    "meeting_f": MeetingInfo(
        start_times=[[t_to_idx(14, 0), t_to_idx(14, 0)]], duration=240 // 5
    ),
    "meeting_g": MeetingInfo(
        start_times=[[t_to_idx(14, 0), t_to_idx(16, 0)]], duration=120 // 5
    ),
}
```

Тепер треба створити інтервал для кожного залу та зустрічі, а також булеву
змінну, яка показує, чи зустріч призначена в зал. Не можна використовувати
один інтервал для двох залів, інакше він буде присутній одночасно в обох.

```python
# Створюємо модель
model = cp_model.CpModel()

# Створюємо змінні старту та кімнат
start_time_vars = {
    name: model.new_int_var_from_domain(
        cp_model.Domain.from_intervals(info.start_times), f"start_{name}"
    )
    for name, info in meetings.items()
}

rooms = ["room_a", "room_b"]
room_vars = {
    name: {room: model.new_bool_var(f"{name}_in_{room}") for room in rooms}
    for name in meetings
}

# Створюємо інтервали та додаємо no-overlap
interval_vars = {
    name: {
        # Окремий інтервал для кожної кімнати
        room: model.new_optional_fixed_size_interval_var(
            start=start_time_vars[name],
            size=info.duration,
            is_present=room_vars[name][room],
            name=f"interval_{name}_in_{room}",
        )
        for room in rooms
    }
    for name, info in meetings.items()
}
```

Тепер гарантуємо, що кожна зустріч призначена рівно в один зал і що у кожному
залі немає перекриття.

```python
# Кожну зустріч призначаємо рівно в один зал
for name, room_dict in room_vars.items():
    model.add_exactly_one(room_dict.values())

for room in rooms:
    model.add_no_overlap([interval_vars[name][room] for name in meetings])
```

І знову візуалізуємо розклад.

| ![Schedule multiple rooms](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/scheduling_multiple_resources.png) |
| :-------------------------------------------------------------------------------------------------------------------------------: |
| Можливий неперекривний розклад для наведеного прикладу з кількома залами. |

> [!TIP]
>
> Цю модель легко розширити, щоб максимізувати кількість зустрічей через
> цільову функцію. Також можна максимізувати відстань між двома зустрічами,
> використавши інтервал зі змінною довжиною. Це хороша вправа.

#### Пакування прямокутників без перекриття

Розгляньмо, як перевірити, чи можна упакувати набір прямокутників у контейнер
без перекриття. Це поширена задача в логістиці або задачах розкрою.

Спочатку визначимо `namedtuple` для прямокутників і контейнера.

```python
from collections import namedtuple

# Визначаємо namedtuple для прямокутників і контейнера
Rectangle = namedtuple("Rectangle", ["width", "height"])
Container = namedtuple("Container", ["width", "height"])

# Приклад
rectangles = [Rectangle(width=2, height=3), Rectangle(width=4, height=5)]
container = Container(width=10, height=10)
```

Далі створимо змінні для нижніх лівих кутів прямокутників і обмежимо їх, щоб
прямокутники залишалися в межах контейнера.

```python
model = cp_model.CpModel()

# Змінні для нижніх лівих кутів
x_vars = [
    model.new_int_var(0, container.width - box.width, name=f"x1_{i}")
    for i, box in enumerate(rectangles)
]
y_vars = [
    model.new_int_var(0, container.height - box.height, name=f"y1_{i}")
    for i, box in enumerate(rectangles)
]
```

Створимо інтервали для кожного прямокутника. Початок — нижній лівий кут, розмір
— ширина або висота. Використаємо `add_no_overlap_2d`, щоб уникнути перекриттів.

```python
# Інтервали для ширини та висоти прямокутників
x_interval_vars = [
    model.new_fixed_size_interval_var(
        start=x_vars[i], size=box.width, name=f"x_interval_{i}"
    )
    for i, box in enumerate(rectangles)
]
y_interval_vars = [
    model.new_fixed_size_interval_var(
        start=y_vars[i], size=box.height, name=f"y_interval_{i}"
    )
    for i, box in enumerate(rectangles)
]

# Забороняємо перекриття
model.add_no_overlap_2d(x_interval_vars, y_interval_vars)
```

Опційні інтервали зі змінною довжиною дозволяють моделювати повороти та
знаходити найбільше можливе пакування. Код здається складним, але є досить
прямолінійним з огляду на складність задачі.

Спочатку визначимо `namedtuple` для прямокутників і контейнера.

```python
from collections import namedtuple
from ortools.sat.python import cp_model

# Визначаємо namedtuple для прямокутників і контейнера
Rectangle = namedtuple("Rectangle", ["width", "height", "value"])
Container = namedtuple("Container", ["width", "height"])

# Приклад
rectangles = [
    Rectangle(width=2, height=3, value=1),
    Rectangle(width=4, height=5, value=1),
]
container = Container(width=10, height=10)
```

Далі створимо змінні для координат прямокутників, включно з нижніми лівими та
верхніми правими кутами, а також булевою змінною, що показує, чи прямокутник
повернутий.

```python
model = cp_model.CpModel()

# Змінні для нижніх лівих і верхніх правих кутів
bottom_left_x_vars = [
    model.new_int_var(0, container.width, name=f"x1_{i}")
    for i, box in enumerate(rectangles)
]
bottom_left_y_vars = [
    model.new_int_var(0, container.height, name=f"y1_{i}")
    for i, box in enumerate(rectangles)
]
upper_right_x_vars = [
    model.new_int_var(0, container.width, name=f"x2_{i}")
    for i, box in enumerate(rectangles)
]
upper_right_y_vars = [
    model.new_int_var(0, container.height, name=f"y2_{i}")
    for i, box in enumerate(rectangles)
]

# Змінні, що показують поворот
rotated_vars = [model.new_bool_var(f"rotated_{i}") for i in range(len(rectangles))]
```

Тепер створимо змінні ширини та висоти з урахуванням повороту, і обмеження, що
зв’язують їх із поворотом.

```python
# Змінні ширини та висоти з урахуванням повороту
width_vars = []
height_vars = []
for i, box in enumerate(rectangles):
    domain = cp_model.Domain.from_values([box.width, box.height])
    width_vars.append(model.new_int_var_from_domain(domain, name=f"width_{i}"))
    height_vars.append(model.new_int_var_from_domain(domain, name=f"height_{i}"))
    # Два можливі варіанти присвоєння ширини/висоти
    model.add_allowed_assignments(
        [width_vars[i], height_vars[i], rotated_vars[i]],
        [(box.width, box.height, 0), (box.height, box.width, 1)],
    )
```

Далі створимо булеву змінну, що означає, чи прямокутник упакований, і інтервали,
що представляють його займаний простір. Їх використовуємо для `add_no_overlap_2d`.

```python
# Змінні, що вказують, чи прямокутник упакований
packed_vars = [model.new_bool_var(f"packed_{i}") for i in range(len(rectangles))]

# Інтервали для ширини та висоти
x_interval_vars = [
    model.new_optional_interval_var(
        start=bottom_left_x_vars[i],
        size=width_vars[i],
        is_present=packed_vars[i],
        end=upper_right_x_vars[i],
        name=f"x_interval_{i}",
    )
    for i, box in enumerate(rectangles)
]
y_interval_vars = [
    model.new_optional_interval_var(
        start=bottom_left_y_vars[i],
        size=height_vars[i],
        is_present=packed_vars[i],
        end=upper_right_y_vars[i],
        name=f"y_interval_{i}",
    )
    for i, box in enumerate(rectangles)
]

# Забороняємо перекриття
model.add_no_overlap_2d(x_interval_vars, y_interval_vars)
```

Нарешті, максимізуємо кількість упакованих прямокутників через цільову функцію.

```python
# Максимізуємо кількість упакованих прямокутників
model.maximize(sum(box.value * x for x, box in zip(packed_vars, rectangles)))
```

|                       ![./images/dense_packing.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/dense_packing.png)                       |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| Це щільне пакування CP-SAT знайшов менш ніж за 0.3 с, що вражає і виглядає ефективнішим за наївну реалізацію в Gurobi. |

Повний код можна знайти тут:

|                           Варіант задачі                           |                                                                                Код                                                                                 |
| :------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     Перевірка здійсненності пакування без поворотів     |    [./evaluations/packing/solver/packing_wo_rotations.py](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/packing/solver/packing_wo_rotations.py)    |
| Пошук найбільшого пакування без поворотів |   [./evaluations/packing/solver/knapsack_wo_rotations.py](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/packing/solver/knapsack_wo_rotations.py)   |
|      Перевірка здійсненності пакування з поворотами       |  [./evaluations/packing/solver/packing_with_rotations.py](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/packing/solver/packing_with_rotations.py)  |
|  Пошук найбільшого пакування з поворотами   | [./evaluations/packing/solver/knapsack_with_rotations.py](https://github.com/d-krupke/cpsat-primer/blob/main/evaluations/packing/solver/knapsack_with_rotations.py) |

CP-SAT добре знаходить здійсненне пакування, але майже не здатен довести
недопустимість. У варіанті «рюкзака» він усе одно пакує більшість прямокутників
навіть для великих інстансів.

|                           ![./images/packing_plot_solved.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/packing_plot_solved.png)                           |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Кількість розв’язаних інстансів для задачі пакування (ліміт 90 с). Повороти трохи ускладнюють задачу. Жоден з використаних інстансів не було доведено як недопустимий. |
|                            ![./images/packing_percentage.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/packing_percentage.png)                            |
|                                           Однак CP-SAT здатен упакувати майже всі прямокутники навіть для найбільших інстансів.                                            |

#### Роздільна здатність і параметри

У ранніх версіях CP-SAT продуктивність обмежень no-overlap сильно залежала від
роздільної здатності. З часом це змінилося, але вплив залишається непослідовним.
У прикладі в ноутбуці я дослідив, як роздільна здатність впливає на час
виконання `add_no_overlap` у версіях 9.3 і 9.8. Для 9.3 час виконання
помітно зростає зі збільшенням роздільної здатності. Натомість у 9.8 час
виконання зменшується при більшій роздільній здатності, що підтвердили повторні
тести. Це несподіване спостереження свідчить, що продуктивність CP-SAT щодо
no-overlap ще не стабілізувалася і може змінюватися в майбутніх версіях.

| Resolution | Runtime (CP-SAT 9.3) | Runtime (CP-SAT 9.8) |
| ---------- | -------------------- | -------------------- |
| 1x         | 0.02s                | 0.03s                |
| 10x        | 0.7s                 | 0.02s                |
| 100x       | 7.6s                 | 1.1s                 |
| 1000x      | 75s                  | 40.3s                |
| 10_000x    | >15min               | 0.4s                 |

[Цей ноутбук](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_no_overlap_2d.ipynb)
використано для створення таблиці.

Втім, експериментуючи з менш документованими можливостями, я помітив, що
продуктивність у старішій версії можна суттєво покращити такими параметрами:

```python
solver.parameters.use_energetic_reasoning_in_no_overlap_2d = True
solver.parameters.use_timetabling_in_no_overlap_2d = True
solver.parameters.use_pairwise_reasoning_in_no_overlap_2d = True
```

У найновішій версії CP-SAT суттєвого приросту я не помітив.

<a name="04-modelling-automaton"></a>

### Обмеження автомата

Обмеження автомата моделюють скінченні автомати, тобто допустимі переходи між
станами. Це особливо корисно у верифікації ПЗ, де важливо, щоб програма
дотримувалася заданої послідовності станів. З огляду на важливість верифікації
в дослідженнях, ці обмеження мають свою аудиторію, але інші можуть перейти до
наступного розділу.

|                  ![Automaton Example](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/automaton.png)                   |
| :----------------------------------------------------------------------------------------------------------------------------------------: |
| Приклад скінченного автомата з чотирма станами і сімома переходами. Стан 0 — початковий, стан 3 — кінцевий. |

Автомат працює так: маємо список цілочисельних змінних `transition_variables`,
які представляють значення переходів. Починаючи зі `starting_state`, наступний
стан визначається трійкою `(state, transition_value, next_state)`, яка відповідає
першій змінній переходу. Якщо такої трійки немає — модель недопустима. Процес
повторюється для кожної наступної змінної. Важливо, щоб останній перехід вів у
кінцевий стан (можливо, через петлю); інакше модель недопустима.

Автомат із прикладу можна змоделювати так:

```python
model = cp_model.CpModel()

transition_variables = [model.new_int_var(0, 2, f"transition_{i}") for i in range(4)]
transition_triples = [
    (0, 0, 1),  # Якщо стан 0 і значення 0, переходимо в стан 1
    (1, 0, 1),  # Якщо стан 1 і значення 0, залишаємося в стані 1
    (1, 1, 2),  # Якщо стан 1 і значення 1, переходимо в стан 2
    (2, 0, 0),  # Якщо стан 2 і значення 0, переходимо в стан 0
    (2, 1, 1),  # Якщо стан 2 і значення 1, переходимо в стан 1
    (2, 2, 3),  # Якщо стан 2 і значення 2, переходимо в стан 3
    (3, 0, 3),  # Якщо стан 3 і значення 0, залишаємося в стані 3
]

model.add_automaton(
    transition_variables=transition_variables,
    starting_state=0,
    final_states=[3],
    transition_triples=transition_triples,
)
```

Присвоєння `[0, 1, 2, 0]` є допустимим, тоді як `[1, 0, 1, 2]` недопустиме, бо зі
стану 0 немає переходу для значення 1. Так само `[0, 0, 1, 1]` недопустиме, бо
не закінчується у кінцевому стані.

> :reference:
>
> Обмеження автомата, наприклад, використано в цій
> [статті](https://arxiv.org/pdf/2410.11981) для моделювання Parallel Batch
> Scheduling With Incompatible Job Families.

<a name="04-modelling-reservoir"></a>

### Обмеження резервуара

Інколи потрібно підтримувати баланс між притоками та відтоками резервуара.
Найочевидніший приклад — водосховище, де рівень води має бути між мінімумом і
максимумом. Обмеження резервуара приймає список часових змінних, список зміни
рівня та мінімальний/максимальний рівень. Якщо афінний вираз `times[i]`
набуває значення `t`, тоді поточний рівень змінюється на `level_changes[i]`.
Зауважте, що змінні зміни рівня наразі не підтримуються — зміни сталі у момент
`t`. Обмеження гарантує, що рівень завжди між мінімумом і максимумом:
`sum(level_changes[i] if times[i] <= t) in [min_level, max_level]`.

Є багато прикладів, окрім водосховища, де потрібно балансувати попит і
пропозицію: підтримка запасів на складі або забезпечення рівня персоналу в
клініці. `add_reservoir_constraint` у CP-SAT дозволяє легко моделювати такі
задачі.

У наведеному прикладі `times[i]` — час застосування зміни `level_changes[i]`,
тому обидва списки мають однакову довжину. Рівень резервуара стартує з 0, і
мінімальний рівень має бути $\leq 0$, а максимальний — $\geq 0$.

```python
times = [model.new_int_var(0, 10, f"time_{i}") for i in range(10)]
level_changes = [1] * 10

model.add_reservoir_constraint(
    times=times,
    level_changes=level_changes,
    min_level=-10,
    max_level=10,
)
```

Додатково, `add_reservoir_constraint_with_active` дозволяє моделювати резервуар
з _опційними_ змінами. Тут маємо список булевих змінних `actives`, де
`actives[i]` означає, чи відбувається зміна `level_changes[i]`, тобто
`sum(level_changes[i] * actives[i] if times[i] <= t) in [min_level, max_level]`.
Якщо зміна не активна, це як якщо б її не існувало, і рівень залишається
незмінним незалежно від часу й значення зміни.

```python
times = [model.new_int_var(0, 10, f"time_{i}") for i in range(10)]
level_changes = [1] * 10
actives = [model.new_bool_var(f"active_{i}") for i in range(10)]

model.add_reservoir_constraint_with_active(
    times=times,
    level_changes=level_changes,
    actives=actives,
    min_level=-10,
    max_level=10,
)
```

Щоб проілюструвати використання, розгляньмо приклад планування медсестер у
клініці. Повний приклад — у
[ноутбуці](https://github.com/d-krupke/cpsat-primer/blob/main/examples/add_reservoir.ipynb).

Клініці потрібно, щоб завжди було достатньо медсестер без надлишку. Для 12-годинного
робочого дня змоделюємо попит як ціле число для кожної години.

```python
# додатне число означає, що потрібно більше медсестер, від’ємне — менше.
demand_change_at_t = [3, 0, 0, 0, 2, 0, 0, 0, -1, 0, -1, 0, -3]
demand_change_times = list(range(len(demand_change_at_t)))  # [0, 1, ..., 12]
```

Є список медсестер, кожна має власну доступність та максимальну тривалість зміни.

```python
max_shift_length = 5

# початок і кінець доступності кожної медсестри
nurse_availabilities = 2 * [
    (0, 7),
    (0, 4),
    (0, 8),
    (2, 9),
    (1, 5),
    (5, 12),
    (7, 12),
    (0, 12),
    (4, 12),
]
```

Ініціалізуємо змінні моделі: старт і кінець зміни кожної медсестри та булеву
змінну, що показує, чи вона працює.

```python
# булева змінна, що показує, чи медсестру заплановано
nurse_scheduled = [
    model.new_bool_var(f"nurse_{i}_scheduled") for i in range(len(nurse_availabilities))
]

# моделюємо початок і кінець кожної зміни
shifts_begin = [
    model.new_int_var(begin, end, f"begin_nurse_{i}")
    for i, (begin, end) in enumerate(nurse_availabilities)
]

shifts_end = [
    model.new_int_var(begin, end, f"end_nurse_{i}")
    for i, (begin, end) in enumerate(nurse_availabilities)
]
```

Додаємо базові обмеження, щоб зміни були валідними.

```python
for begin, end in zip(shifts_begin, shifts_end):
    model.add(end >= begin)  # кінець після початку
    model.add(end - begin <= max_shift_length)  # зміна не надто довга
```

Рівень резервуара — це кількість запланованих медсестер у будь-який момент
мінус попит до цього моменту. Додаємо обмеження резервуара, щоб завжди вистачало
персоналу, але не було надлишку (рівень між 0 і 2). Маємо три типи змін:

1. Попит змінюється на початку кожної години. Для цього використовуємо фіксовані
   моменти часу і активуємо всі зміни. Попит зі знаком мінус, бо зростання попиту
   знижує рівень резервуара.
2. Початок зміни медсестри підвищує рівень на 1. Час — `shifts_begin`, зміна
   активна лише якщо медсестра запланована.
3. Завершення зміни знижує рівень на 1. Час — `shifts_end`, зміна активна лише
   якщо медсестра запланована.

```python
times = demand_change_times
demands = [
    -demand for demand in demand_change_at_t
]  # зростання попиту знижує резервуар
actives = [1] * len(demand_change_times)

times += list(shifts_begin)
demands += [1] * len(shifts_begin)  # медсестра починає зміну
actives += list(nurse_scheduled)

times += list(shifts_end)
demands += [-1] * len(shifts_end)  # медсестра завершує зміну
actives += list(nurse_scheduled)

model.add_reservoir_constraint_with_active(
    times=times,
    level_changes=demands,
    min_level=0,
    max_level=2,
    actives=actives,
)
```

> [!NOTE]
>
> Обмеження резервуара дозволяють описувати умови, які важко змоделювати
> «вручну». Проте, хоча в мене небагато практики з ними, я не очікував би, що їх
> легко оптимізувати. Напишіть, якщо у вас є позитивний або негативний досвід
> використання та для яких масштабів задач вони працюють добре.

<a name="04-modelling-pwl"></a>

### Нелінійні обмеження / кусково-лінійні функції

На практиці часто трапляються функції витрат, які не є лінійними. Наприклад,
виробнича задача, де ви виробляєте три різні вироби. Кожен виріб має різні
компоненти, які потрібно купувати. Вартість компонентів спочатку зменшується зі
збільшенням обсягу, а потім зростає, коли постачальник вичерпує запаси і
доводиться купувати у дорожчого. Крім того, кількість клієнтів, готових платити
певну ціну, обмежена. Якщо хочете продавати більше, доведеться знижувати ціну,
що зменшить прибуток.

Припустімо, така функція має вигляд $y=f(x)$ на рисунку нижче. На жаль, це
досить складна функція, яку не можна напряму виразити в CP-SAT. Проте можна
наблизити її кусково-лінійною функцією (червона лінія). Такі апроксимації дуже
поширені, а деякі розв’язувачі навіть роблять це автоматично, наприклад Gurobi.
Роздільну здатність можна збільшувати довільно, але чим більше сегментів, тим
складніша модель. Тому зазвичай її роблять лише настільки високою, наскільки
потрібно.

|                                                                                                                     ![./images/pwla.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/pwla.png)                                                                                                                      |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Ми можемо змоделювати довільну неперервну функцію кусково-лінійною. Тут ми розбиваємо оригінал на кілька лінійних сегментів. Точність можна адаптувати під вимоги. Лінійні сегменти далі можна виразити в CP-SAT. Чим менше сегментів, тим простіше моделювати і розв’язувати. |

Використовуючи лінійні обмеження (`model.add`) та реїфікацію (`.only_enforce_if`),
можна змоделювати кусково-лінійну функцію в CP-SAT. Для цього використовуємо
булеві змінні, що вибирають сегмент, і активуємо відповідне лінійне обмеження
через реїфікацію. Однак у CP-SAT виникають дві проблеми, показані на рисунку.

|                                                                                                             ![./images/pwla_problems.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/pwla_problems.png)                                                                                                              |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Навіть якщо f(x) складається з лінійних сегментів, ми не можемо просто реалізувати $y=f(x)$ в CP-SAT. По-перше, для багатьох значень $x$ функція дає нецілі значення, отже модель стає недопустимою. По-друге, канонічне представлення лінійних сегментів часто потребує нецілих коефіцієнтів, що також заборонено в CP-SAT. |

- **Проблема A:** навіть якщо сегмент — лінійна функція, результат може бути
  нецілим. У прикладі $f(5)=3{,}5$, і якщо ми задаємо $y=f(x)$, то значення
  $x=5$ стає забороненим, що не є бажаним. Є два варіанти: або використовувати
  складнішу апроксимацію, яка гарантує цілі значення, або застосувати
  нерівності. Перший варіант може вимагати занадто багато сегментів і бути
  надто дорогим. Другий дає слабше обмеження, адже ми можемо задати лише
  $y<=f(x)$ або $y>=f(x)$, але не $y=f(x)$. Якщо спробувати обидві нерівності,
  отримаємо ту саму недопустимість. Але часто достатньо однієї нерівності.
  Якщо потрібно обмежити $y$ зверху — використовуємо $y<=f(x)$, якщо знизу —
  $y>=f(x)$. Якщо $f(x)$ представляє витрати, то використовуємо $y>=f(x)$ і
  мінімізуємо $y$.

- **Проблема B:** канонічна форма лінійної функції — $y=ax+b$. Часто потрібні
  нецілі коефіцієнти. Їх можна масштабувати до цілих, додаючи масштабний
  множник. Наприклад, нерівність $y=0.5x+0.5$ можна переписати як $2y=x+1$.
  Це робиться через НСК, але масштаб може стати великим і привести до
  переповнень.

Можлива реалізація:

```python
# Ми хочемо задати y=f(x)
x = model.new_int_var(0, 7, "x")
y = model.new_int_var(0, 5, "y")

# Булеві змінні для вибору сегмента
segment_active = [model.new_bool_var("segment_1"), model.new_bool_var("segment_2")]
model.add_at_most_one(segment_active)  # активний лише один сегмент

# Сегмент 1
# якщо 0<=x<=3, тоді y >= 0.5*x + 0.5
model.add(2 * y >= x + 1).only_enforce_if(segment_active[0])
model.add(x >= 0).only_enforce_if(segment_active[0])
model.add(x <= 3).only_enforce_if(segment_active[0])

# Сегмент 2
model.add(_SLIGHTLY_MORE_COMPLEX_INEQUALITY_).only_enforce_if(segment_active[1])
model.add(x >= 3).only_enforce_if(segment_active[1])
model.add(x <= 7).only_enforce_if(segment_active[1])

model.minimize(y)
# якщо б ми максимізували y, використовували б <= замість >=
```

Це може бути доволі громіздко, але я написав невеликий helper-клас, який робить
це автоматично. Він знаходиться в
[./utils/piecewise_functions](https://github.com/d-krupke/cpsat-primer/blob/main/utils/piecewise_functions/).
Просто скопіюйте у свій код.

Цей код робить додаткові оптимізації:

1. Розгляд кожного сегмента як окремого випадку може бути дорогим і
   неефективним. Тому суттєво допомагає, якщо ви можете об’єднати кілька
   сегментів в один випадок. Це можна зробити, виявляючи опуклі ділянки, адже
   обмеження опуклих областей не заважають одне одному.
2. Додавання опуклої оболонки сегментів як надлишкового обмеження, що не
   залежить від `only_enforce_if`, іноді допомагає розв’язувачу краще обмежити
   область. Обмеження з `only_enforce_if` зазвичай погано працюють для лінійної
   релаксації, а незалежна опукла оболонка одразу обмежує простір розв’язків без
   гілкування по випадках.

Застосуймо цей код до задачі вище.

У нас є два продукти, кожен вимагає три компоненти. Перший продукт потребує 3
компоненти 1, 5 компоненти 2 і 2 компоненти 3. Другий продукт потребує 2
компоненти 1, 1 компоненту 2 і 3 компоненти 3. Ми можемо купити до 1500 кожного
компонента за цінами з рисунка нижче. Виробляти можна до 300 одиниць кожного
продукту і продавати їх за цінами з рисунка.

| ![./images/production_example_cost_components.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/production_example_cost_components.png) | ![./images/production_example_selling_price.png](https://github.com/d-krupke/cpsat-primer/blob/main/images/production_example_selling_price.png) |
| :--------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: |
| Витрати на закупівлю компонентів для виробництва. | Ціни продажу продукції. |

Хочемо максимізувати прибуток, тобто дохід мінус витрати на компоненти. Модель:

```python
requirements_1 = (3, 5, 2)
requirements_2 = (2, 1, 3)

from ortools.sat.python import cp_model

model = cp_model.CpModel()
buy_1 = model.new_int_var(0, 1_500, "buy_1")
buy_2 = model.new_int_var(0, 1_500, "buy_2")
buy_3 = model.new_int_var(0, 1_500, "buy_3")

produce_1 = model.new_int_var(0, 300, "produce_1")
produce_2 = model.new_int_var(0, 300, "produce_2")

model.add(produce_1 * requirements_1[0] + produce_2 * requirements_2[0] <= buy_1)
model.add(produce_1 * requirements_1[1] + produce_2 * requirements_2[1] <= buy_2)
model.add(produce_1 * requirements_1[2] + produce_2 * requirements_2[2] <= buy_3)

# Код із ./utils!
from piecewise_functions import PiecewiseLinearFunction, PiecewiseLinearConstraint

# Функції витрат
costs_1 = [(0, 0), (1000, 400), (1500, 1300)]
costs_2 = [(0, 0), (300, 300), (700, 500), (1200, 600), (1500, 1100)]
costs_3 = [(0, 0), (200, 400), (500, 700), (1000, 900), (1500, 1500)]
# PiecewiseLinearFunction — pydantic модель і легко серіалізується
f_costs_1 = PiecewiseLinearFunction(
    xs=[x for x, y in costs_1], ys=[y for x, y in costs_1]
)
f_costs_2 = PiecewiseLinearFunction(
    xs=[x for x, y in costs_2], ys=[y for x, y in costs_2]
)
f_costs_3 = PiecewiseLinearFunction(
    xs=[x for x, y in costs_3], ys=[y for x, y in costs_3]
)

# Функції доходу
gain_1 = [(0, 0), (100, 800), (200, 1600), (300, 2_000)]
gain_2 = [(0, 0), (80, 1_000), (150, 1_300), (200, 1_400), (300, 1_500)]
f_gain_1 = PiecewiseLinearFunction(xs=[x for x, y in gain_1], ys=[y for x, y in gain_1])
f_gain_2 = PiecewiseLinearFunction(xs=[x for x, y in gain_2], ys=[y for x, y in gain_2])

# Обмеження y>=f(x) для витрат
x_costs_1 = PiecewiseLinearConstraint(model, buy_1, f_costs_1, upper_bound=False)
x_costs_2 = PiecewiseLinearConstraint(model, buy_2, f_costs_2, upper_bound=False)
x_costs_3 = PiecewiseLinearConstraint(model, buy_3, f_costs_3, upper_bound=False)

# Обмеження y<=f(x) для доходу
x_gain_1 = PiecewiseLinearConstraint(model, produce_1, f_gain_1, upper_bound=True)
x_gain_2 = PiecewiseLinearConstraint(model, produce_2, f_gain_2, upper_bound=True)

# Максимізуємо дохід мінус витрати
model.Maximize(x_gain_1.y + x_gain_2.y - (x_costs_1.y + x_costs_2.y + x_costs_3.y))

solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True
status = solver.solve(model)
print(f"Buy {solver.value(buy_1)} of component 1")
print(f"Buy {solver.value(buy_2)} of component 2")
print(f"Buy {solver.value(buy_3)} of component 3")
print(f"Produce {solver.value(produce_1)} of product 1")
print(f"Produce {solver.value(produce_2)} of product 2")
print(f"Overall gain: {solver.objective_value}")
```

Отримаємо такий результат:

```
Buy 930 of component 1
Buy 1200 of component 2
Buy 870 of component 3
Produce 210 of product 1
Produce 150 of product 2
Overall gain: 1120.0
```

На жаль, такі задачі швидко стають дуже складними для моделювання і розв’язання.
Це лише доказ того, що теоретично такі задачі можна моделювати в CP-SAT. На
практиці без експертності можна втратити багато часу й нервів.
