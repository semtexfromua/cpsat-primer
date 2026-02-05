<!-- EDIT THIS PART VIA 05_parameters.md -->

<a name="05-parameters"></a>

## Параметри

<!-- START_SKIP_FOR_README -->

![Обкладинка «Параметри»](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/logo_3.webp)

<!-- STOP_SKIP_FOR_README -->

CP-SAT має багато параметрів для керування своєю поведінкою. Ці параметри
реалізовані через
[Protocol Buffers](https://developers.google.com/protocol-buffers) і змінюються
через властивість `parameters`. Щоб переглянути всі доступні опції, дивіться
добре задокументований `proto`-файл в
[офіційному репозиторії](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto).
Нижче я виділю найважливіші параметри, щоб ви могли отримати максимум від CP-SAT.

> :warning: Лише кілька параметрів, таких як `max_time_in_seconds`, підходять
> початківцям. Більшість інших параметрів, як-от стратегії рішень, краще
> залишати за замовчуванням, бо вони добре підібрані, і втручання може лише
> нашкодити оптимізаціям. Для кращої продуктивності зосередьтеся на покращенні
> моделі.

### Логування

Параметр `log_search_progress` особливо важливий на початку. Він вмикає лог
пошуку, що дає уявлення про те, як CP-SAT розв’язує вашу задачу. У продакшені
ви можете його вимкнути, але під час розробки він корисний для розуміння
процесу й реагування на проблеми.

```python
solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True

# Власна функція логування, наприклад через Python logging замість stdout
# Корисно в Jupyter, де stdout може бути не видно
solver.log_callback = print  # (str)->None
# Якщо використовуєте власну функцію логування, можна вимкнути stdout
solver.parameters.log_to_stdout = False
```

Лог містить цінну інформацію для розуміння CP-SAT і вашої задачі. Він показує,
скільки змінних було видалено, які техніки найефективніше покращували нижні та
верхні межі, тощо.

Приклад логу:

```
Starting CP-SAT solver v9.10.4067
Parameters: max_time_in_seconds: 30 log_search_progress: true relative_gap_limit: 0.01
Setting number of workers to 16

Initial optimization model '': (model_fingerprint: 0x1d316fc2ae4c02b1)
#Variables: 450 (#bools: 276 #ints: 6 in objective)
  - 342 Booleans in [0,1]
  - 12 in [0][10][20][30][40][50][60][70][80][90][100]
  - 6 in [0][10][20][30][40][100]
  - 6 in [0][80][100]
  - 6 in [0][100]
  - 6 in [0,1][34][67][100]
  - 12 in [0,6]
  - 18 in [0,7]
  - 6 in [0,35]
  - 6 in [0,36]
  - 6 in [0,100]
  - 12 in [21,57]
  - 12 in [22,57]
#kBoolOr: 30 (#literals: 72)
#kLinear1: 33 (#enforced: 12)
#kLinear2: 1'811
#kLinear3: 36
#kLinearN: 94 (#terms: 1'392)

Starting presolve at 0.00s
  3.26e-04s  0.00e+00d  [DetectDominanceRelations]
  6.60e-03s  0.00e+00d  [PresolveToFixPoint] #num_loops=4 #num_dual_strengthening=3
  2.69e-05s  0.00e+00d  [ExtractEncodingFromLinear] #potential_supersets=44 #potential_subsets=12
[Symmetry] Graph for symmetry has 2'224 nodes and 5'046 arcs.
[Symmetry] Symmetry computation done. time: 0.000374304 dtime: 0.00068988
[Symmetry] #generators: 2, average support size: 12
[Symmetry] 12 orbits with sizes: 2,2,2,2,2,2,2,2,2,2,...
[Symmetry] Found orbitope of size 6 x 2
[SAT presolve] num removable Booleans: 0 / 309
[SAT presolve] num trivial clauses: 0
[SAT presolve] [0s] clauses:570 literals:1152 vars:303 one_side_vars:268 simple_definition:35 singleton_clauses:0
[SAT presolve] [3.0778e-05s] clauses:570 literals:1152 vars:303 one_side_vars:268 simple_definition:35 singleton_clauses:0
[SAT presolve] [4.6758e-05s] clauses:570 literals:1152 vars:303 one_side_vars:268 simple_definition:35 singleton_clauses:0
  1.10e-02s  9.68e-03d  [Probe] #probed=1'738 #new_bounds=12 #new_binary_clauses=1'111
  2.34e-03s  0.00e+00d  [MaxClique] Merged 602(1374 literals) into 506(1960 literals) at_most_ones.
  3.31e-04s  0.00e+00d  [DetectDominanceRelations]
  1.89e-03s  0.00e+00d  [PresolveToFixPoint] #num_loops=2 #num_dual_strengthening=1
  5.45e-04s  0.00e+00d  [ProcessAtMostOneAndLinear]
  8.19e-04s  0.00e+00d  [DetectDuplicateConstraints] #without_enforcements=306
  8.62e-05s  7.21e-06d  [DetectDominatedLinearConstraints] #relevant_constraints=114 #num_inclusions=42
  1.94e-05s  0.00e+00d  [DetectDifferentVariables]
  1.90e-04s  8.39e-06d  [ProcessSetPPC] #relevant_constraints=560 #num_inclusions=24
  2.01e-05s  0.00e+00d  [FindAlmostIdenticalLinearConstraints]
...
```

З огляду на складність логу, я розробив інструмент для візуалізації та
коментування. Ви можете просто вставити лог у цей інструмент — він автоматично
підсвітить найважливіші моменти. Обов’язково перегляньте приклади.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cpsat-log-analyzer.streamlit.app/)
[![d-krupke - CP-SAT Log Analyzer](https://img.shields.io/badge/d--krupke-CP--SAT%20Log%20Analyzer-blue?style=for-the-badge&logo=github)](https://github.com/d-krupke/CP-SAT-Log-Analyzer)

|                                                                                                                       ![Search Progress](https://github.com/d-krupke/cpsat-primer/blob/main/images/search_progress.png)                                                                                                                       |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Графік прогресу пошуку з часом, побудований аналізатором логів (інший лог, ніж вище). Він допомагає зрозуміти, що складніше: знайти гарний розв’язок чи довести його якість. На цій основі можна застосувати відповідні контрзаходи. |

Ми повернемося до логів у наступному розділі.

> [!TIP]
>
> З мого викладацького досвіду, я часто бачив, як студенти вважали, що CP-SAT
> «завис», і лише потім виявлялося, що в їхньому коді побудови моделі є
> зайвий вкладений цикл $O(n^5)$, який виконується днями. Логічно підозрювати
> CP-SAT, бо він розв’язує складну частину задачі. Але навіть «проста» частина
> побудови моделі може забрати багато часу, якщо реалізована невірно. Увімкнувши
> логування, студенти одразу бачили, що проблема у їхньому коді, а не в CP-SAT.
> Це може зберегти багато часу й нервів.

### Ліміт часу і статус

Працюючи з великими або складними моделями, CP-SAT може не знайти оптимальний
розв’язок за розумний час і потенційно працювати безкінечно. Тому важливо
встановлювати ліміт часу, особливо в продакшені. Навіть у межах ліміту CP-SAT
часто знаходить досить хороший розв’язок, хоча він може бути не доведено
оптимальним.

Вибір адекватного ліміту часу залежить від багатьох факторів і потребує
експериментів. Я зазвичай починаю з 60–300 секунд, щоб не чекати надто довго
під час тестування і водночас дати розв’язувачу шанс знайти хороший розв’язок.

Щоб встановити ліміт часу (у секундах), використайте:

```python
solver.parameters.max_time_in_seconds = 60  # 60с ліміт
```

Після запуску важливо перевіряти статус, щоб зрозуміти, чи знайдено оптимальний
або допустимий розв’язок, чи розв’язку немає:

```python
status = solver.solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("We have a solution.")
else:
    print("Help?! No solution available! :( ")
```

Можливі статуси:

- `OPTIMAL` (4): знайдено оптимальний розв’язок.
- `FEASIBLE` (2): знайдено допустимий розв’язок; можна оцінити його якість через
  `solver.best_objective_bound`.
- `INFEASIBLE` (3): жоден розв’язок не задовольняє всі обмеження.
- `MODEL_INVALID` (1): модель CP-SAT задана некоректно.
- `UNKNOWN` (0): розв’язок не знайдено і доказ недопустимості відсутній.
  Проте межа може бути доступна.

Щоб отримати назву статусу за кодом, використовуйте `solver.status_name(status)`.

Окрім ліміту часу, можна задавати допустиму якість розв’язку через
`absolute_gap_limit` та `relative_gap_limit`. Абсолютний ліміт зупиняє
розв’язувач, коли розв’язок у межах заданого значення від межі. Відносний
ліміт зупиняє, коли значення цілі (O) знаходиться у певному відсотку від межі
(B). Щоб зупинитися, коли розв’язок (доведено) в межах 5% від оптимуму:

```python
solver.parameters.relative_gap_limit = 0.05
```

Якщо прогрес зупинився або з інших причин, можна використати callback для
розв’язків, щоб зупиняти пошук за певних умов. З їхньою допомогою ви можете
перевіряти кожен новий розв’язок і вирішувати, чи достатньо він хороший. На
відміну від Gurobi, CP-SAT не підтримує додавання lazy constraints із callback
(і взагалі їх не підтримує), що є важливим обмеженням для задач з динамічними
корекціями моделі.

Щоб додати callback, потрібно успадкувати `CpSolverSolutionCallback`. Документація
доступна
[тут](https://developers.google.com/optimization/reference/python/sat/python/cp_model#cp_model.CpSolverSolutionCallback).

```python
class MySolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, data):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.data = data  # Зберігаємо дані

    def on_solution_callback(self):
        obj = self.objective_value  # Найкраще значення
        bound = self.best_objective_bound  # Найкраща межа
        print(f"The current value of x is {self.Value(x)}")
        if abs(obj - bound) < 10:
            self.StopSearch()  # Зупиняємо пошук
        # ...


solver.solve(model, MySolutionCallback(None))
```

Офіційний приклад із callback:
[тут](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/stop_after_n_solutions_sample_sat.py).

> [!WARNING]
>
> Межі — це палка з двома кінцями. З одного боку, вони показують, наскільки ви
> близько до оптимуму в межах обраної моделі, і дозволяють рано завершити
> оптимізацію. З іншого — вони можуть вводити в оману з двох причин. По-перше,
> межі стосуються моделі і можуть створити хибне відчуття якості, адже ні модель,
> ні дані зазвичай не ідеальні. По-друге, іноді отримати хороші межі за
> розумний час неможливо, хоча сам розв’язок може бути хорошим — ви просто цього
> не знаєте. Це може призвести до марних витрат ресурсів у гонитві за кращими
> межами чи підходами. Межі корисні, але важливо розуміти їхнє походження і
> обмеження.

Окрім значення цілі та межі, можна отримати й внутрішні метрики, як-от
`self.num_booleans`, `self.num_branches`, `self.num_conflicts`. Ці метрики
обговоримо пізніше.

Починаючи з версії 9.10, CP-SAT підтримує callbacks для меж (bound callbacks),
які викликаються, коли доведена межа покращується. На відміну від
solution callbacks, які спрацьовують при знаходженні нового розв’язку, bound
callbacks корисні для зупинки пошуку, коли межа достатньо хороша. Синтаксис
відрізняється: це вільні функції, які напряму працюють із solver.

```python
solver = cp_model.CpSolver()


def bound_callback(bound):
    print(f"New bound: {bound}")
    if bound > 100:
        solver.stop_search()


solver.best_bound_callback = bound_callback
```

Замість простої функції можна використати callable-об’єкт, щоб зберігати
посилання на solver. Це дозволяє винести callback за межі локальної області.

```python
class BoundCallback:
    def __init__(self, solver) -> None:
        self.solver = solver

    def __call__(self, bound):
        print(f"New bound: {bound}")
        if bound > 200:
            print("Abort search due to bound")
            self.solver.stop_search()
```

Цей метод гнучкіший і виглядає більш «пайтонічно».

Крім того, при кожному новому розв’язку або межі генерується лог-повідомлення.
Ви можете під’єднатися до логів і вирішувати, коли зупиняти пошук, через
log callback.

```python
solver.parameters.log_search_progress = True  # Увімкнути логування
solver.log_callback = lambda msg: print("LOG:", msg)  # (str) -> None
```

> [!WARNING]
>
> Будьте обережні з callbacks: вони можуть суттєво сповільнити розв’язувач.
> Вони викликаються часто і змушують повертатися до повільнішого Python-рівня.
> Я часто бачив, як студенти скаржилися на повільну роботу solver-а, а
> більшість часу витрачалася саме у callback. Навіть прості операції в
> callback можуть швидко накопичувати затрати часу.

### Паралелізація

CP-SAT — портфельний розв’язувач, який використовує різні техніки паралельно.
Є певний обмін інформацією між воркерами, але CP-SAT не ділить простір рішень
на частини, як це робить branch-and-bound у MIP. Це може призводити до певної
надлишковості пошуку, але паралельне використання різних технік підвищує шанси
знайти «правильну» стратегію. Передбачити, яка техніка найкраща для конкретної
задачі, часто складно, тому паралелізація може бути дуже корисною.

За замовчуванням CP-SAT використовує всі доступні ядра (включно з
гіпертредингом). Ви можете керувати паралелізацією, встановивши кількість
воркерів.

```python
solver.parameters.num_workers = 8  # використовувати 8 ядер
```

> [!TIP]
>
> Для багатьох моделей продуктивність можна підвищити, зменшивши кількість
> воркерів до кількості фізичних ядер або навіть менше. Це дає змогу іншим
> воркерам працювати на вищій частоті, отримати більше пропускної здатності
> пам’яті та зменшити взаємні перешкоди. Але майте на увазі: менше воркерів
> означає менше напрямків пошуку одночасно, що може знизити шанси прогресу.

Ось які сабсолвери використовує CP-SAT 9.9 на різних рівнях паралелізації для
оптимізаційної задачі без додаткових специфікацій (наприклад, стратегій рішень).
Кожен рядок описує додаткові сабсолвери порівняно з попереднім рядком. Зверніть
увагу, що деякі параметри/обмеження/цілі можуть змінювати стратегію
паралелізації. Також див. офіційну документацію:
[troubleshooting](https://github.com/google/or-tools/blob/main/ortools/sat/docs/troubleshooting.md#improving-performance-with-multiple-workers).

| # Workers | Full Problem Subsolvers                                                        | First Solution Subsolvers                                                                                                                                                                                                                                                                 | Incomplete Subsolvers                                                                                                                                                                                                                                                  | Helper Subsolvers                                                                 |
| --------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **1**     | `default_lp`                                                                   | No solver                                                                                                                                                                                                                                                                                 | No solver                                                                                                                                                                                                                                                              | No solver                                                                         |
| **2**     |                                                                                |                                                                                                                                                                                                                                                                                           | +13 solvers: `feasibility_pump`, `graph_arc_lns`, `graph_cst_lns`, `graph_dec_lns`, `graph_var_lns`, `packing_precedences_lns`, `packing_rectangles_lns`, `packing_slice_lns`, `rins/rens`, `rnd_cst_lns`, `rnd_var_lns`, `scheduling_precedences_lns`, `violation_ls` | +3 solvers: `neighborhood_helper`, `synchronization_agent`, `update_gap_integral` |
| **3**     | +1 solver: `no_lp`                                                             |                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                        |                                                                                   |
| **4**     | +1 solver: `max_lp`                                                            |                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                        |                                                                                   |
| **5**     |                                                                                | +1 solver: `fj_short_default`                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                                        |                                                                                   |
| **6**     | +1 solver: `quick_restart`                                                     |                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                        |                                                                                   |
| **7**     | +1 solver: `reduced_costs`                                                     |                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                        |                                                                                   |
| **8**     | +1 solver: `quick_restart_no_lp`                                               |                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                        |                                                                                   |
| **12**    | +2 solvers: `lb_tree_search`, `pseudo_costs`                                   | +2 solvers: `fj_long_default`, `fs_random`                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                                        |                                                                                   |
| **16**    | +3 solvers: `objective_lb_search`, `objective_shaving_search_no_lp`, `probing` | +1 solver: `fs_random_quick_restart`                                                                                                                                                                                                                                                      |                                                                                                                                                                                                                                                                        |                                                                                   |
| **20**    | +2 solvers: `objective_shaving_search_max_lp`, `probing_max_lp`                | +1 solver: `fj_short_lin_default`                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                        |                                                                                   |
| **32**    | +2 solvers: `objective_lb_search_max_lp`, `objective_lb_search_no_lp`          | +8 solvers: `fj_long_lin_default`, `fj_long_lin_random`, `fj_long_random`, `fj_short_lin_random`, `fj_short_random`, `fs_random_no_lp`, `fs_random_quick_restart_no_lp`                                                                                                                   | +1 solver: `violation_ls(3)`                                                                                                                                                                                                                                           |                                                                                   |
| **64**    |                                                                                | +11 solvers: `fj_long_default(2)`, `fj_long_lin_default(2)`, `fj_long_lin_random(2)`, `fj_long_random(2)`, `fj_short_default(2)`, `fj_short_lin_default(2)`, `fj_short_random(2)`, `fs_random(6)`, `fs_random_no_lp(6)`, `fs_random_quick_restart(6)`, `fs_random_quick_restart_no_lp(5)` | +1 solver: `violation_ls(7)`                                                                                                                                                                                                                                           |                                                                                   |

Важливі моменти:

- З одним воркером використовується лише default сабсолвер.
- З двома і більше воркерами CP-SAT починає використовувати неповні сабсолвери,
  тобто евристики, як-от LNS.
- З п’ятьма воркерами CP-SAT також має сабсолвер для першого розв’язку.
- З 32 воркерами використовуються всі 15 повних сабсолверів.
- Понад 32 воркери здебільшого збільшують кількість сабсолверів першого розв’язку.

**Full problem subsolvers** — це сабсолвери, які шукають у повному просторі,
наприклад за допомогою branch-and-bound. Доступні:

- `default_lp`: пошук на базі LCG із типовою лінеаризацією.
  - `max_lp`: те саме, але з максимальною лінеаризацією.
  - `no_lp`: те саме, але без лінеаризації.
- `lb_tree_search`: зосереджений на покращенні доведеної межі, а не на пошуку
  кращих розв’язків. Він спростовує дешеві вузли дерева, поступово покращуючи
  межу, але має мало шансів знайти кращі розв’язки.
- `objective_lb_search`: також фокусується на покращенні межі через спростування
  поточної нижньої межі.
  - `objective_lb_search_max_lp`: із максимальною лінеаризацією.
  - `objective_lb_search_no_lp`: без лінеаризації.
  - `objective_shaving_search_max_lp`: має бути схожим на `objective_lb_search_max_lp`.
  - `objective_shaving_search_no_lp`: має бути схожим на `objective_lb_search_no_lp`.
- `probing`: фіксує змінні та дивиться, що відбувається.
  - `probing_max_lp`: те саме, але з максимальною лінеаризацією.
- `pseudo_costs`: використовує псевдовартість для вибору гілок на основі
  історичних змін меж.
- `quick_restart`: робить частіші перезапуски. Перезапуски перебудовують дерево
  з нуля, але зберігають навчений набір клауз. Це допомагає виходити з поганих
  рішень і зменшує дерево пошуку.
  - `quick_restart_no_lp`: те саме, але без лінеаризації.
- `reduced_costs`: використовує reduced costs лінійної релаксації для розгалуження.
- `core`: стратегія зі світу SAT, що витягує unsat cores.
- `fixed`: користувацька стратегія.

Використані сабсолвери можна змінювати через `solver.parameters.subsolvers`,
`solver.parameters.extra_subsolvers` та `solver.parameters.ignore_subsolvers`.
Це може бути цікаво, якщо ви використовуєте CP-SAT, бо лінійна релаксація
марна (і BnB працює погано). Є ще більше опцій, див. документацію:
[сат параметри](https://github.com/google/or-tools/blob/49b6301e1e1e231d654d79b6032e79809868a70e/ortools/sat/sat_parameters.proto#L513).
Пам’ятайте: тюнінг solver-а складний, і часто можна зробити гірше, ніж краще.
Втім, я помічав, що зменшення кількості воркерів іноді покращує час виконання.
Це означає, що підбір правильних сабсолверів може бути корисним. Наприклад,
`max_lp` — марна трата ресурсів, якщо ви знаєте, що у моделі слабка лінійна
релаксація. У цьому контексті рекомендую дивитися на релаксовані розв’язки
складних задач, щоб краще зрозуміти, де solver може «застрягати» (для цього
можна використовувати LP-розв’язувач, наприклад Gurobi).

Ефективність стратегій можна оцінювати через блоки `Solutions` і `Objective bounds`
в логах. Приклад:

```
Solutions (7)             Num   Rank
                'no_lp':    3  [1,7]
        'quick_restart':    1  [3,3]
  'quick_restart_no_lp':    3  [2,5]

Objective bounds                     Num
                  'initial_domain':    1
             'objective_lb_search':    2
       'objective_lb_search_no_lp':    4
  'objective_shaving_search_no_lp':    1
```

Для розв’язків перше число — кількість розв’язків, друге — діапазон рангів. [1,7]
означає, що знайдені розв’язки мали ранги від 1 до 7. У цьому прикладі
стратегія `no_lp` знайшла як найкращий, так і найгірший розв’язок.

Для меж число означає, скільки разів стратегія внесла вклад у найкращу межу.
У цьому прикладі виглядає, що найуспішніші — `no_lp`. Важливо також, які
стратегії взагалі відсутні у списку.

У логу пошуку також видно, коли і який сабсолвер щось зробив. Це включає
неповні та first-solution сабсолвери:

```
#1       0.01s best:43    next:[6,42]     no_lp (fixed_bools=0/155)
#Bound   0.01s best:43    next:[7,42]     objective_shaving_search_no_lp (vars=73 csts=120)
#2       0.01s best:33    next:[7,32]     quick_restart_no_lp (fixed_bools=0/143)
#3       0.01s best:31    next:[7,30]     quick_restart (fixed_bools=0/123)
#4       0.01s best:17    next:[7,16]     quick_restart_no_lp (fixed_bools=2/143)
#5       0.01s best:16    next:[7,15]     quick_restart_no_lp (fixed_bools=22/147)
#Bound   0.01s best:16    next:[8,15]     objective_lb_search_no_lp
#6       0.01s best:15    next:[8,14]     no_lp (fixed_bools=41/164)
#7       0.01s best:14    next:[8,13]     no_lp (fixed_bools=42/164)
#Bound   0.01s best:14    next:[9,13]     objective_lb_search
#Bound   0.02s best:14    next:[10,13]    objective_lb_search_no_lp
#Bound   0.04s best:14    next:[11,13]    objective_lb_search_no_lp
#Bound   0.06s best:14    next:[12,13]    objective_lb_search
#Bound   0.25s best:14    next:[13,13]    objective_lb_search_no_lp
#Model   0.26s var:125/126 constraints:162/162
#Model   2.24s var:124/126 constraints:160/162
#Model   2.58s var:123/126 constraints:158/162
#Model   2.91s var:121/126 constraints:157/162
#Model   2.95s var:120/126 constraints:155/162
#Model   2.97s var:109/126 constraints:140/162
#Model   2.98s var:103/126 constraints:135/162
#Done    2.98s objective_lb_search_no_lp
#Done    2.98s quick_restart_no_lp
#Model   2.98s var:66/126 constraints:91/162
```

**Неповні сабсолвери** — це евристики, які не шукають у повному просторі.
Приклади: large neighborhood search (LNS) і feasibility pumps. Перший шукає
кращий розв’язок, змінюючи лише кілька змінних, другий намагається зробити
недопустимі/неповні розв’язки допустимими. Можна запускати лише неповні
сабсолвери через `solver.parameters.use_lns_only = True`, але це потрібно
поєднувати з лімітом часу, бо вони не знають, коли зупинитися.

**Сабсолвери першого розв’язку** — стратегії, що намагаються швидко знайти
перший розв’язок. Вони часто використовуються для «прогріву» solver-а.

<!-- Source on Parallelization in Gurobi and general opportunities -->

Якщо цікавить, як Gurobi паралелізує пошук, є чудове відео
[тут](https://www.youtube.com/watch?v=FJz1UxaMWRQ). Ed Rothberg також пояснює
загальні можливості й виклики паралелізації solver-ів, що корисно і для
розуміння CP-SAT.

<!-- Give a disclaimer -->

> :warning: Цей розділ може потребувати допомоги: є ймовірність, що я
> переплутав деякі стратегії або зробив хибні висновки.

#### Імпорт/експорт моделей для порівняння на різному залізі

Якщо ви хочете порівняти продуктивність різних рівнів паралелізації або
обладнання, можна експортувати модель. Тоді не потрібно відтворювати її або
ділитися кодом — можна просто завантажити модель на іншій машині й запустити
solver.

```python
from ortools.sat.python import cp_model
from ortools.sat import cp_model_pb2
from google.protobuf import text_format
from pathlib import Path

def _detect_binary_mode(filename: str) -> bool:
    if filename.endswith((".txt", ".pbtxt", ".pb.txt")):
        return False
    if filename.endswith((".pb", ".bin", ".proto.bin", ".dat")):
        return True
    raise ValueError(f"Unknown extension for file: {filename}")

# Зміни в ortools 9.15: було model.Proto().SerializeToString() / text_format.MessageToString()
# Тепер використовуємо model.export_to_file(), який сам визначає формат
def export_model(model: cp_model.CpModel, filename: str, binary: bool | None = None):
    binary = _detect_binary_mode(filename) if binary is None else binary
    # export_to_file використовує .txt для текстового формату, інакше бінарний
    # Тому обробляємо невідповідності для деяких розширень
    if binary and filename.endswith(".txt"):
        # Примусово бінарний формат для .txt через тимчасовий файл
        temp_file = filename + ".pb"
        model.export_to_file(temp_file)
        Path(filename).write_bytes(Path(temp_file).read_bytes())
        Path(temp_file).unlink()
    elif not binary and not filename.endswith(".txt"):
        # Примусово текстовий формат без .txt
        temp_file = filename + ".txt"
        model.export_to_file(temp_file)
        Path(filename).write_text(Path(temp_file).read_text())
        Path(temp_file).unlink()
    else:
        model.export_to_file(filename)

# Зміни в ortools 9.15: було model.Proto().ParseFromString() / text_format.Parse()
# Тепер використовуємо model.Proto().parse_text_format() для тексту або cp_model_pb2 для бінарного
def import_model(filename: str, binary: bool | None = None) -> cp_model.CpModel:
    binary = _detect_binary_mode(filename) if binary is None else binary
    model = cp_model.CpModel()
    if binary:
        # Парсимо бінарний формат через protobuf, потім конвертуємо в текст
        proto = cp_model_pb2.CpModelProto()
        proto.ParseFromString(Path(filename).read_bytes())
        model.Proto().parse_text_format(text_format.MessageToString(proto))
    else:
        model.Proto().parse_text_format(Path(filename).read_text())
    return model
```

Бінарний формат ефективніший і має використовуватися для великих моделей.
Текстовий формат зручніший для читання й порівняння.

### Підказки (Hints)

Якщо ви маєте інтуїцію, яким може бути розв’язок — можливо, з подібної моделі
або доброї евристики — ви можете передати ці підказки CP-SAT. Деякі воркери
спробують слідувати цим підказкам, що може значно покращити продуктивність, якщо
підказки хороші. Якщо підказки представляють допустимий розв’язок, solver може
використати їх, щоб обрізати гілки з гіршими межами.

```python
model.add_hint(x, 1)  # Підказка: x, ймовірно, 1
model.add_hint(y, 2)  # Підказка: y, ймовірно, 2
```

Більше прикладів:
[official example](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/solution_hinting_sample_sat.py).
Ми також побачимо використання hints для багатокритеріальної оптимізації в
розділі [Coding Patterns](#06-coding-patterns).

> [!TIP]
>
> Підказки можуть суттєво покращити продуктивність, особливо якщо solver не
> може швидко знайти хороший початковий розв’язок (видно в логах). Це часто
> називають **warm-starting**. Не потрібно давати підказки для всіх допоміжних
> змінних, але якщо ви використовуєте цілочисельні змінні для апроксимації
> неперервних, варто дати підказки і для них. CP-SAT може довго доводити
> допустимість підказки, і лише завершені розв’язки можна використовувати для
> обрізання пошуку. Якщо solver довго завершує підказку, він може даремно
> витратити час на гілки, які можна було б обрізати.

Щоб перевірити, чи підказки допустимі, можна тимчасово зафіксувати змінні на
підказаних значеннях і перевірити, чи модель не стає недопустимою:

```python
solver.parameters.fix_variables_to_their_hinted_value = True
status = solver.solve(model)
if status == cp_model.INFEASIBLE:
    print("Hints are conflicting or infeasible!")
```

Якщо підказки не використовуються, це може свідчити про логічну помилку в моделі
або баг у коді. Такий підхід надійно виявляє недопустимі підказки.

Також є `solver.parameters.debug_crash_on_bad_hint = True`, що падає solver
при неможливості завершити підказку. Але ця функція ненадійна: спрацьовує лише
в multi-worker режимі, залежить від гонки між воркерами, і контролюється
`hint_conflict_limit` (за замовчуванням 10). Підхід із
`fix_variables_to_their_hinted_value` простіший і детермінований.

> [!WARNING]
>
> У старих версіях CP-SAT підказки могли уповільнювати solver навіть якщо вони
> були правильні, але не оптимальні. У нових версіях ця проблема, здається,
> вирішена, але погані підказки все одно можуть вести solver у хибному напрямку.

Часто потрібно дослідити вплив фіксації змінних на певні значення. Щоб не
копіювати модель, можна використовувати hints і параметр
`fix_variables_to_their_hinted_value`.

```python
solver.parameters.fix_variables_to_their_hinted_value = True
```

Після цього підказки можна очистити через `model.clear_hints()` і тестувати інші
варіанти без дублювання моделі. Хоча складні вирази напряму додати не можна,
фіксація змінних дозволяє експериментувати зі складними обмеженнями без копій
моделі. Для тимчасових складних обмежень інколи все ж потрібне копіювання через
`model.CopyFrom` разом із копіюванням змінних.

Також можна використати функцію для автоматичного завершення hints для
допоміжних змінних, які часто складно задавати вручну. Запустіть функцію перед
розв’язанням. Встановіть ліміт часу залежно від складності. Якщо значення можна
вивести простою пропагацією, навіть великі моделі обробляються швидко.

```python
def complete_hint(
    model: cp_model.CpModel,
    time_limit: float = 0.5,
):
    """
    Completes the hint via a limited solve. Since CP-SAT only accepts complete hints,
    performing this step can improve solver performance.

    Args:
        model: The CpModel object to update.
        time_limit: Time limit for the solve (in seconds).

    Notes:
        This function performs a quick solve to deduce variable values.
        If successful, it replaces any existing hint with a complete one.
        If not successful, the model remains unchanged and a warning is issued.
    """
    logging.info("Completing hint with a time limit of %d seconds", time_limit)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.fix_variables_to_their_hinted_value = True
    status = solver.solve(model)
    logging.info(
        "Automatically completing hint with status: %s", solver.status_name(status)
    )
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        # Clear the existing hint to avoid model invalidation.
        model.clear_hints()
        # Set a new complete hint using the solver result.
        for i, _ in enumerate(model.proto.variables):
            v_ = model.get_int_var_from_proto_index(i)
            model.add_hint(v_, solver.value(v_))
        logging.info(
            "Hint successfully completed within time limit. Status: %s",
            solver.status_name(status),
        )
    else:
        logging.warning(
            "Unable to complete hint within time limit. Status: %s",
            solver.status_name(status),
        )

```

> [!WARNING]
>
> Під час presolve модель змінюється. Наприклад, симетрії можуть бути зламані
> шляхом заборони еквівалентних варіантів одного розв’язку. Хоча це може
> суттєво покращити продуктивність, це ускладнює збереження допустимості hints,
> наприклад, якщо підказка відповідає «обрізаній» варіації.
>
> На жаль, CP-SAT історично мав труднощі зі збереженням допустимості hints під
> час presolve. Хоча деякі проблеми вже виправлені, я досі бачу цю поведінку у
> свіжих релізах.
>
> Як обхідний шлях, можна змусити CP-SAT зберігати всі допустимі розв’язки під
> час presolve:
>
> ```python
> solver.parameters.keep_all_feasible_solutions_in_presolve = True
> ```
>
> Але цей параметр може погіршити продуктивність. Якщо помічаєте, що hints
> стають недопустимими після presolve, експериментально визначте, що краще:
> зберігати hints чи повноцінний presolve.

## Посилення моделі

Для просунутих користувачів, які працюють із CP-SAT інкрементально — тобто
модифікують і розв’язують модель багато разів — може бути цікавим параметр:

```python
solver.parameters.fill_tightened_domains_in_response = True
```

Коли ви прибираєте цільову функцію і розв’язуєте задачу на здійсненність,
solver повертає звужені домени змінних. Це може суттєво зменшити простір пошуку,
покращивши продуктивність, особливо якщо ви розв’язуєте модель багато разів із
різними цілями чи додатковими обмеженнями.

Однак, якщо цільова функція залишається, деякі допустимі розв’язки можуть бути
виключені. Вони можуть стати релевантними, якщо цілі чи обмеження зміняться
пізніше.

Увімкнення цього параметра не змінює модель; він лише повертає список звужених
доменів у response об’єкті, які ви можете використати в моделі.

```python
# Приклад після розв’язання
for i, v in enumerate(self.model.proto.variables):
    print(f"Tightened domain for variable {i} '{v.name}' is {solver.response_proto.tightened_variables[i].domain}")
```

### Припущення

Ще один спосіб дослідити вплив фіксації змінних — це припущення, що є типовою
функцією в SAT-розв’язувачах. На відміну від фіксації hints, припущення
обмежені булевими літералами в CP-SAT.

```python
b1 = model.new_bool_var("b1")
b2 = model.new_bool_var("b2")
b3 = model.new_bool_var("b3")

model.add_assumptions([b1, ~b2])  # припускаємо b1=True, b2=False
model.add_assumption(b3)  # припускаємо b3=True (один літерал)
# ... solve again and analyze ...
model.clear_assumptions()  # очищаємо припущення
```

> [!NOTE]
>
> Інкрементальні SAT-розв’язувачі можуть повторно використовувати вивчені
> клаузи між запуском за різних припущень. CP-SAT цього не підтримує. Його
> solver — безстанний і завжди починає з нуля.

Хоча припущення дозволяють досліджувати різні присвоєння булевих змінних без
перебудови моделі, CP-SAT має потужнішу можливість: витягнення unsat core з
недопустимих моделей. Це особливо корисно для налагодження. Умикаючи
обмеження умовно через `only_enforce_if(b)` і додаючи `b` як припущення, можна
ізолювати джерела недопустимості. Якщо модель недопустима, CP-SAT може
повернути мінімальний набір припущень (і відповідних обмежень), що спричиняють
конфлікт.

Розгляньмо приклад. Маємо три цілочисельні змінні $x$, $y$, $z$ і обмеження:

1. $x + y \leq 4$
2. $x + z \leq 2$
3. $z \geq 4$

За умови невід’ємності змінних модель явно недопустима через конфлікт між (2)
і (3). У великих моделях знайти джерело недопустимості складно. Але з
припущеннями і `sufficient_assumptions_for_infeasibility` CP-SAT може
автоматично показати, які обмеження винні.

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()

# Цілочисельні змінні
x = model.new_int_var(0, 100, 'x')
y = model.new_int_var(0, 100, 'y')
z = model.new_int_var(0, 100, 'z')

# Обмеження
indicator_1 = model.new_bool_var('Indicator 1: x + y <= 4')
model.add(x + y <= 4).only_enforce_if(indicator_1)
indicator_2 = model.new_bool_var('Indicator 2: x + z <= 2')
model.add(x + z <= 2).only_enforce_if(indicator_2)
indicator_3 = model.new_bool_var('Indicator 3: z >= 4')
model.add(z >= 4).only_enforce_if(indicator_3)

# Припущення
model.add_assumptions([indicator_1, indicator_2, indicator_3])

# Розв’язання
solver = cp_model.CpSolver()
status = solver.solve(model)

assert status == cp_model.INFEASIBLE

print("Minimal unsat core:")
for var_index in solver.sufficient_assumptions_for_infeasibility():
    print(f"{var_index}: '{model.proto.variables[var_index].name}'")
```

Результат:

```
Minimal unsat core:
4: 'Indicator 2: x + z <= 2'
5: 'Indicator 3: z >= 4'
```

На жаль, не всі обмеження в CP-SAT підтримують реїфікацію. Проте нижчорівневі
обмеження, де недопустимість найімовірніша, зазвичай підтримують. Для
високорівневих обмежень можуть існувати обхідні перетворення.

> :reference:
>
> Більше інформації у
> [документації CPMpy](https://cpmpy.readthedocs.io/en/latest/unsat_core_extraction.html),
> бібліотеки моделювання з підтримкою CP-SAT як бекенду.

### Попереднє спрощення (presolve)

CP-SAT має етап presolve, який спрощує модель перед розв’язанням. Це може
суттєво зменшити простір пошуку і покращити продуктивність. Однак presolve може
бути дорогим, особливо для великих моделей. Якщо ваша модель відносно проста
(тобто є небагато складних рішень), і ви бачите, що presolve займає багато
часу, а сам пошук швидкий — можна зменшити presolve.

Наприклад, presolve можна повністю вимкнути:

```python
solver.parameters.cp_model_presolve = False
```

Але це надто радикально, тож можна лише обмежити presolve, а не вимикати.

Щоб зменшити кількість ітерацій presolve, використовуйте:

```python
solver.parameters.max_presolve_iterations = 3
```

Також можна обмежити конкретні техніки presolve, наприклад probing, який
фіксує змінні та дивиться результат. Probing потужний, але затратний.

```python
solver.parameters.cp_model_probing_level = 1
solver.parameters.presolve_probing_deterministic_time_limit = 5
```

Є додаткові параметри для керування presolve. Перед змінами рекомендую
переглянути лог, щоб зрозуміти, що саме забирає час.

Пам’ятайте: зменшення presolve може погіршити здатність розв’язувати складні
моделі. Не жертвуйте продуктивністю на складних інстансах лише заради швидких
простих випадків.

### Додавання власного сабсолвера до портфеля

Як ми бачили, CP-SAT використовує портфель сабсолверів із різними налаштуваннями
(наприклад, різним рівнем лінеаризації). Ви можете задати свій сабсолвер із
певною конфігурацією. Важливо не змінювати параметри на верхньому рівні, бо це
вплине на всі сабсолвери, включно з LNS. Це може зруйнувати баланс портфеля,
увімкнувши дорогі техніки для LNS, що сповільнить їх до непридатності. Також
ви ризикуєте створити сабсолвер за замовчуванням, несумісний з моделлю
(наприклад, якщо він вимагає ціль), і тоді CP-SAT може виключити більшість або
усі сабсолвери, зробивши solver неефективним або неспроможним.

Наприклад, у задачах пакування деякі дорогі техніки пропагації можуть сильно
прискорити пошук, але при неправильному використанні — сильно сповільнити.
Тому можна додати один сабсолвер із цими техніками. Якщо вони не допоможуть,
сповільниться лише один воркер, а решта портфеля працюватиме нормально. Якщо ж
допоможуть — цей воркер зможе ділитися розв’язками та межами з іншими,
покращуючи загальну продуктивність.

Ось як додати власний сабсолвер:

```python
from ortools.sat import sat_parameters_pb2

packing_subsolver = sat_parameters_pb2.SatParameters()
packing_subsolver.name = "MyPackingSubsolver"
packing_subsolver.use_area_energetic_reasoning_in_no_overlap_2d = True
packing_subsolver.use_energetic_reasoning_in_no_overlap_2d = True
packing_subsolver.use_timetabling_in_no_overlap_2d = True
packing_subsolver.max_pairs_pairwise_reasoning_in_no_overlap_2d = 5_000

# Додаємо сабсолвер до портфеля
solver.parameters.subsolver_params.append(packing_subsolver)  # Визначення
solver.parameters.extra_subsolvers.append(
    packing_subsolver.name
)  # Активація
```

Після додавання перевірте в логах, чи сабсолвер активний. Якщо його немає,
ймовірно, параметри несумісні з моделлю, і його виключили.

```
8 full problem subsolvers: [MyPackingSubsolver, default_lp, max_lp, no_lp, probing, probing_max_lp, quick_restart, quick_restart_no_lp]
```

Якщо хочете дізнатися, як налаштовані існуючі сабсолвери, див. файл
[cp_model_search.cc](https://github.com/google/or-tools/blob/stable/ortools/sat/cp_model_search.cc)
в репозиторії OR-Tools.

> [!TIP]
>
> Ви також можете перевизначити параметри існуючого сабсолвера, використавши
> те саме ім’я. Зміняться лише параметри, які ви явно задаєте. Також можна
> додати кілька сабсолверів у портфель, але майте на увазі, що це може
> «виштовхнути» деякі попередньо визначені сабсолвери, якщо воркерів
> недостатньо.

### Стратегія рішень

Наприкінці цього розділу — параметр, який може бути цікавим для просунутих
користувачів, бо дає уявлення про алгоритм пошуку. Його можна використовувати
разом із `solver.parameters.enumerate_all_solutions = True` щоб задати порядок
перебору рішень. Це може впливати на продуктивність, але передбачити це складно,
тому не змінюйте параметри без вагомої причини.

Ми можемо вказати CP-SAT, як розгалужуватися, коли пропагація більше нічого не
дедукує. Для цього потрібен список змінних (порядок може бути важливим),
стратегія вибору змінної (зафіксовані змінні автоматично пропускаються) і
стратегія вибору значення.

Варіанти вибору змінної:

- `CHOOSE_FIRST`: перша незакріплена змінна у списку.
- `CHOOSE_LOWEST_MIN`: змінна, що потенційно може мати найменше значення.
- `CHOOSE_HIGHEST_MAX`: змінна, що потенційно може мати найбільше значення.
- `CHOOSE_MIN_DOMAIN_SIZE`: змінна з найменшим доменом.
- `CHOOSE_MAX_DOMAIN_SIZE`: змінна з найбільшим доменом.

Варіанти вибору значення/діапазону:

- `SELECT_MIN_VALUE`: пробуємо найменше значення.
- `SELECT_MAX_VALUE`: пробуємо найбільше значення.
- `SELECT_LOWER_HALF`: беремо нижню половину.
- `SELECT_UPPER_HALF`: беремо верхню половину.
- `SELECT_MEDIAN_VALUE`: пробуємо медіану.

```python
model.add_decision_strategy([x], cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)

# можна змусити CP-SAT слідувати цій стратегії точно
solver.parameters.search_branching = cp_model.FIXED_SEARCH
```

Наприклад, для [розфарбування графа](https://en.wikipedia.org/wiki/Graph_coloring)
(з цілими кольорами) ми можемо впорядкувати змінні за спаданням степеня
(через `CHOOSE_FIRST`) і завжди пробувати найменший колір (`SELECT_MIN_VALUE`).
Це дає неявну kernelization: якщо потрібно щонайменше $k$ кольорів, вершини з
менш ніж $k$ сусідами тривіальні. Розміщуючи їх наприкінці списку, CP-SAT
розглядатиме їх лише після розфарбування вершин з більшим степенем. Інша
стратегія — `CHOOSE_LOWEST_MIN`, щоб завжди брати вершину з найменшим доступним
кольором. Чи допоможе це — треба перевіряти: CP-SAT часто сам знаходить критичні
вершини через конфлікти.

> [!WARNING]
>
> Я трохи експериментував із ручними стратегіями. Навіть для розфарбування,
> де це здається логічним, це дало перевагу лише для поганої моделі. Після
> покращення моделі через розбиття симетрії результат став гіршим. Я також
> припускаю, що CP-SAT сам динамічно навчається найкращій стратегії (як це
> робить Gurobi) краще, ніж статичні ручні налаштування.

---
