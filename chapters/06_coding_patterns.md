<!-- EDIT THIS PART VIA 06_coding_patterns.md -->

# Частина 2: Просунуті теми

<a name="06-coding-patterns"></a>

## Патерни кодування для задач оптимізації

<!-- START_SKIP_FOR_README -->

![Обкладинка «Патерни»](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/logo_4.webp)

<!-- STOP_SKIP_FOR_README -->

У цьому розділі ми розглянемо різні патерни кодування, які допомагають
структурувати реалізації оптимізаційних задач із використанням CP-SAT. Ці
патерни особливо корисні, коли ви працюєте зі складними задачами, які потрібно
розв’язувати постійно й потенційно за змінних вимог. Хоча ми зосереджуємося на
Python API CP-SAT, багато патернів можна адаптувати до інших розв’язувачів і
мов.

У багатьох випадках достатньо просто описати модель і розв’язати її без
особливої структури. Але є ситуації, коли моделі складні й потребують частих
ітерацій — або через продуктивність, або через зміну вимог. Тоді критично мати
добру структуру, щоб легко змінювати й розширювати код, не ламаючи його, а також
щоб полегшити тестування й розуміння. Уявіть, що у вас складна модель і потрібно
адаптувати обмеження через нові вимоги. Якщо код не модульний, а тестова система
перевіряє лише всю модель, навіть невелика зміна змусить переписувати всі тести.
Після кількох ітерацій ви можете взагалі відмовитися від тестів — а це
небезпечний шлях.

Ще одна типова проблема в складних моделях — ризик забути додати тривіальні
обмеження, що з’єднують допоміжні змінні, через що частина моделі перестає
працювати. Якщо це стосується здійсненності, ви, можливо, помітите проблему
під час перевірки здійсненності розв’язку. Але якщо це впливає на цільову
функцію (наприклад, штрафи), ви можете не помітити, що розв’язок субоптимальний,
бо штрафи не застосовано. Крім того, реалізація складних обмежень часто важка,
і модульна структура дозволяє тестувати такі обмеження окремо. Розробка через
тести (TDD) — ефективний підхід для швидкої й надійної реалізації складних
обмежень.

Оптимізація як сфера дуже неоднорідна, і відсоток оптимізаторів із професійним
бекграундом у програмній інженерії, здається, напрочуд низький. Багато
оптимізаційної роботи роблять математики, фізики та інженери, які мають глибоку
експертизу у своїх галузях, але обмежений досвід у софт-інженерії. Вони
зазвичай дуже кваліфіковані й можуть створювати чудові моделі, але їхній код
часто непідтримуваний і не відповідає практикам ПЗ. Багато задач подібні між
собою, тож мінімальних пояснень чи структури часто вважають достатніми —
подібно до побудови графіків через копіювання шаблонів. Такий підхід може бути
не надто читабельним, але для багатьох у галузі це звично. Також для
математиків типово спочатку документувати модель, а потім реалізовувати її.
З погляду інженерії це нагадує водоспадну модель, яка не є гнучкою.

Схоже, що літератури про agile-розробку в оптимізації бракує, і цей розділ
покликаний заповнити прогалину, описуючи патерни, які я вважаю корисними у
своїй роботі. Я запитував старших колег, але вони не змогли запропонувати
ресурси або навіть не бачили потреби в них. Для багатьох кейсів простого
підходу справді достатньо. Але я виявив, що ці патерни роблять мій agile,
тест-орієнтований робочий процес значно легшим, швидшим і приємнішим. З огляду
на брак джерел, цей розділ значною мірою базується на моєму особистому досвіді.
Буду радий почути ваші історії та патерни, які ви вважаєте корисними.

Далі ми почнемо з базового патерну на основі функцій, а потім перейдемо до
інших концепцій і патернів, які я вважаю цінними. Ми працюватимемо на простих
прикладах, де переваги патернів можуть бути неочевидними, але сподіваюся, ви
побачите їхній потенціал у складніших задачах. Альтернатива — навести складні
приклади, що могло б відволікти від самих патернів.

> [!TIP]
>
> Наведені патерни зосереджуються на деталях, специфічних для обчислювальної
> оптимізації. Проте багато інженерів-оптимізаторів походять із математики чи
> фізики і можуть не мати професійного досвіду в Python чи софт-інженерії. Якщо
> це про вас, рекомендую ознайомитися особливо з
> [базовими структурами даних і їх _comprehensions_](https://docs.python.org/3/tutorial/datastructures.html)
> та елегантними циклами з
> [itertools](https://docs.python.org/3/library/itertools.html). Ці інструменти
> дозволяють елегантніше виражати математичні ідеї у Python і особливо корисні
> для задач оптимізації.
>
> Також є чудові інструменти для автоматичного форматування, перевірки та
> покращення коду, наприклад [ruff](https://docs.astral.sh/ruff/tutorial/).
> Регулярний запуск `ruff check --fix` та `ruff format` підвищує якість коду з
> мінімальними зусиллями. Оптимально інтегрувати це через
> [pre-commit hook](https://pre-commit.com/).
>
> Для старту з побудови оптимізаційних моделей загалом дуже рекомендую статтю
> [The Art Of Not Making It An Art](https://www.gurobi.com/resources/optimization-modeling-the-art-of-not-making-it-an-art/).
> Вона чудово підсумовує принципи успішного ведення оптимізаційного проєкту
> незалежно від конкретної мови чи розв’язувача.

### Проста функція

Для простих задач оптимізації практично обгорнути побудову та розв’язання
моделі в одну функцію. Цей метод підходить для простих випадків, але має меншу
гнучкість для складних сценаріїв. Параметри на кшталт ліміту часу або допуску
оптимальності можна задавати через keyword-аргументи з значеннями за
замовчуванням.

Нижче приклад функції на Python для задачі рюкзака. Нагадаємо, що в задачі
рюкзака ми вибираємо предмети (з вагою і цінністю), щоб максимізувати сумарну
цінність при обмеженні на вагу. Через простоту (лише одне обмеження) ця задача
ідеальна для вступних прикладів.

```python
from ortools.sat.python import cp_model
from typing import List


def solve_knapsack(
    weights: List[int],
    values: List[int],
    capacity: int,
    *,
    time_limit: int = 900,
    opt_tol: float = 0.01,
) -> List[int]:
    # ініціалізуємо модель
    model = cp_model.CpModel()
    n = len(weights)  # кількість предметів
    # Змінні рішень
    x = [model.new_bool_var(f"x_{i}") for i in range(n)]
    # Обмеження місткості
    model.add(sum(weights[i] * x[i] for i in range(n)) <= capacity)
    # Цільова функція
    model.maximize(sum(values[i] * x[i] for i in range(n)))
    # Розв’язання
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.relative_gap_limit = opt_tol
    status = solver.solve(model)
    # Витягуємо розв’язок
    return (
        [i for i in range(n) if solver.value(x[i])]
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        else []
    )
```

Можна додати гнучкість, дозволивши передавати будь-які параметри solver-а.

```python
def solve_knapsack(
    weights: List[int],
    values: List[int],
    capacity: int,
    *,
    time_limit: int = 900,
    opt_tol: float = 0.01,
    **kwargs,
) -> List[int]:
    # ініціалізуємо модель
    model = cp_model.CpModel()
    # ...
    # Розв’язання
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.relative_gap_limit = opt_tol
    for key, value in kwargs.items():
        setattr(solver.parameters, key, value)
    # ...
```

Додайте модульні тести у окремому файлі (наприклад, `test_knapsack.py`), щоб
переконатися, що модель працює очікувано.

> [!TIP]
>
> Пишіть тести до написання коду. Це підхід TDD, який допомагає структурувати
> код і забезпечити його правильність. Він також змушує продумати API функції
> до реалізації.

```python
# Переконайтеся, що у вас правильна структура проєкту і ви можете імпортувати функцію
from myknapsacksolver import solve_knapsack

def test_knapsack_empty():
    # Тест для тривіального випадку завжди корисний
    assert solve_knapsack([], [], 0) == []

def test_knapsack_nothing_fits():
    # Якщо нічого не вміщується, розв’язок має бути порожнім
    assert solve_knapsack([10, 20, 30], [1, 2, 3], 5) == []

def test_knapsack_one_item():
    # Якщо вміщується лише один предмет
    assert solve_knapsack([10, 20, 30], [1, 2, 3], 10) == [0]

def test_knapsack_all_items():
    # Якщо вміщуються всі предмети
    assert solve_knapsack([10, 20, 30], [1, 2, 3], 100) == [0, 1, 2]
```

Запустити всі тести в проєкті можна через `pytest .`. Гарний туторіал —
[Real Python](https://realpython.com/pytest-python-testing/).

### Логування побудови моделі

У великих задачах логування процесу побудови моделі може бути критичним для
пошуку й виправлення проблем. Часто проблема не в solver-і, а саме в моделі.

У прикладі нижче додаємо базове логування у функцію solver-а, щоб отримати
інсайти про побудову моделі. Логування легко вмикається/вимикається через
logging framework, що дозволяє використовувати його і в продакшені.

Якщо ви не знайомі з logging у Python — це чудова нагода. Я вважаю це
необхідною навичкою для продакшн-коду, і подібні фреймворки використовуються
майже всюди. Офіційна документація має
[гарний туторіал](https://docs.python.org/3/howto/logging.html). Дехто любить
інші фреймворки, але вбудований logging цілком достатній і кращий за `print`.

```python
import logging
from ortools.sat.python import cp_model
from typing import List

# Налаштовуємо logging, якщо ще не налаштовано
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)

_logger = logging.getLogger(__name__)


def solve_knapsack(
    weights: List[int],
    values: List[int],
    capacity: int,
    *,
    time_limit: int = 900,
    opt_tol: float = 0.01,
) -> List[int]:
    _logger.debug("Building the knapsack model")
    model = cp_model.CpModel()
    n = len(weights)
    _logger.debug("Number of items: %d", n)
    if n > 0:
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "Min/Mean/Max weight: %d/%.2f/%d",
                min(weights),
                sum(weights) / n,
                max(weights),
            )
            _logger.debug(
                "Min/Mean/Max value: %d/%.2f/%d", min(values), sum(values) / n, max(values)
            )
    x = [model.new_bool_var(f"x_{i}") for i in range(n)]
    model.add(sum(weights[i] * x[i] for i in range(n)) <= capacity)
    model.maximize(sum(values[i] * x[i] for i in range(n)))
    _logger.debug("Model created with %d items", n)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.relative_gap_limit = opt_tol
    _logger.debug(
        "Starting the solution process with time limit %d seconds", time_limit
    )
    status = solver.solve(model)
    selected_items = (
        [i for i in range(n) if solver.value(x[i])]
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        else []
    )
    _logger.debug("Selected items: %s", selected_items)
    return selected_items
```

У наступних прикладах ми не використовуватимемо логування, щоб зекономити місце,
але варто подумати про його додавання до коду.

> [!TIP]
>
> Класний хак із logging — можна легко під’єднатися до коду і робити аналіз
> не лише через тексти логів. Ви можете написати handler, який, наприклад,
> ловить тег "Selected items: %s" і отримує сам об’єкт (а не лише рядок). Це
> дуже корисно для збору статистики чи візуалізації процесу пошуку без зміни
> (продакшн) коду.

### Власні data-класи для інстансів, конфігурацій і розв’язків

Використання серіалізованих data-класів зі строгими схемами для інстансів,
конфігурацій і розв’язків значно підвищує читабельність і підтримуваність
коду. Це також полегшує документацію, тестування і забезпечує узгодженість
даних у великих проєктах.

Одна з популярних бібліотек для цього —
[Pydantic](https://docs.pydantic.dev/latest/). Вона проста і дає багато
функціональності «з коробки». Нижче — класи для інстансу, конфігурації і
розв’язку задачі рюкзака. Хоча duck typing у Python зручний для швидкої
внутрішньої розробки, він може створювати проблеми для API. Користувачі часто
неправильно використовують інтерфейс і звинувачують вас. Pydantic допомагає
зменшити ці проблеми, даючи чіткий інтерфейс і валідацію. До того ж можна
легко створити API через FastAPI, який побудований на Pydantic.

```python
# pip install pydantic
from pydantic import (
    BaseModel,
    PositiveInt,
    NonNegativeFloat,
    PositiveFloat,
    Field,
    model_validator,
)


class KnapsackInstance(BaseModel):
    # Інстанс задачі
    weights: list[PositiveInt] = Field(..., description="The weight of each item.")
    values: list[PositiveInt] = Field(..., description="The value of each item.")
    capacity: PositiveInt = Field(..., description="The capacity of the knapsack.")

    @model_validator(mode="after")
    def check_lengths(cls, v):
        if len(v.weights) != len(v.values):
            raise ValueError("Mismatch in number of weights and values.")
        return v


class KnapsackSolverConfig(BaseModel):
    # Конфігурація solver-а
    time_limit: PositiveFloat = Field(
        default=900.0, description="Time limit in seconds."
    )
    opt_tol: NonNegativeFloat = Field(
        default=0.01, description="Optimality tolerance (1% gap allowed)."
    )
    log_search_progress: bool = Field(
        default=False, description="Whether to log the search progress."
    )


class KnapsackSolution(BaseModel):
    # Розв’язок задачі
    selected_items: list[int] = Field(..., description="Indices of selected items.")
    objective: float = Field(..., description="Objective value of the solution.")
    upper_bound: float = Field(
        ..., description="Upper bound of the solution, i.e., a proven limit on how good a solution could be."
    )
```

> [!WARNING]
>
> Схема даних має бути повністю підготовлена для оптимізації без додаткової
> передобробки. Підготовка даних і оптимізація — обидві складні задачі, і їхнє
> змішування значно ускладнює код. Ідеально, щоб оптимізаційний код просто
> проходив по даних і додавав відповідні обмеження та цілі до моделі.

Початковий код потрібно адаптувати під ці data-класи.

```python
from ortools.sat.python import cp_model


def solve_knapsack(
    instance: KnapsackInstance, config: KnapsackSolverConfig
) -> KnapsackSolution:
    model = cp_model.CpModel()
    n = len(instance.weights)
    x = [model.new_bool_var(f"x_{i}") for i in range(n)]
    model.add(sum(instance.weights[i] * x[i] for i in range(n)) <= instance.capacity)
    model.maximize(sum(instance.values[i] * x[i] for i in range(n)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.time_limit
    solver.parameters.relative_gap_limit = config.opt_tol
    solver.parameters.log_search_progress = config.log_search_progress
    status = solver.solve(model)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return KnapsackSolution(
            selected_items=[i for i in range(n) if solver.value(x[i])],
            objective=solver.objective_value,
            upper_bound=solver.best_objective_bound,
        )
    return KnapsackSolution(selected_items=[], objective=0, upper_bound=0)
```

Можна використовувати можливості серіалізації Pydantic для швидкого створення
тест-кейсів на основі реальних даних. Хоча такі тести не гарантують коректність,
вони принаймні сигналізують про неочікувані зміни логіки після рефакторингу.

```python
from datetime import datetime
from hashlib import md5
from pathlib import Path


def add_test_case(instance: KnapsackInstance, config: KnapsackSolverConfig):
    """
    Швидко генеруємо тест-кейс на основі інстансу і конфігурації.
    """
    test_folder = Path(__file__).parent / "test_data"
    unique_id = (
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        + "_"
        + md5(
            (instance.model_dump_json() + config.model_dump_json()).encode()
        ).hexdigest()
    )
    subfolder = test_folder / "knapsack" / unique_id
    subfolder.mkdir(parents=True, exist_ok=True)
    with open(subfolder / "instance.json", "w") as f:
        f.write(instance.model_dump_json())
    with open(subfolder / "config.json", "w") as f:
        f.write(config.model_dump_json())
    solution = solve_knapsack(instance, config)
    with open(subfolder / "solution.json", "w") as f:
        f.write(solution.model_dump_json())


def test_saved_test_cases():
    test_folder = Path(__file__).parent / "test_data"
    for subfolder in test_folder.glob("knapsack/*"):
        with open(subfolder / "instance.json") as f:
            instance = KnapsackInstance.model_validate_json(f.read())
        with open(subfolder / "config.json") as f:
            config = KnapsackSolverConfig.model_validate_json(f.read())
        with open(subfolder / "solution.json") as f:
            solution = KnapsackSolution.model_validate_json(f.read())
        new_solution = solve_knapsack(instance, config)
        assert (
            new_solution.objective <= solution.upper_bound
        ), "New solution is better than the previous upper bound: One has to be wrong."
        assert (
            solution.objective <= new_solution.upper_bound
        ), "Old solution is better than the new upper bound: One has to be wrong."
        # Не тестуємо selected_items, бо solver може знайти інший розв’язок тієї ж якості
```

Тепер можна легко генерувати тест-кейси і перевіряти їх таким кодом. Ідеально,
якщо ви використовуєте реальні інстанси, наприклад автоматично зберігаючи 1%
інстансів з продакшену.

```python
# Інстанс задачі рюкзака
instance = KnapsackInstance(
    weights=[23, 31, 29, 44, 53, 38, 63, 85, 89, 82],
    values=[92, 57, 49, 68, 60, 43, 67, 84, 87, 72],
    capacity=165,
)
# Конфігурація solver-а
config = KnapsackSolverConfig(
    time_limit=10.0, opt_tol=0.01, log_search_progress=False
)
# Розв’язання
solution = solve_knapsack(instance, config)
# Додаємо тест-кейс у папку
add_test_case(instance, config)
```

Також легко підтримувати зворотну сумісність, якщо додавати значення за
замовчуванням для нових полів.

> [!TIP]
>
> Часто доводиться проектувати data-класи максимально загальними, щоб вони
> підходили для кількох solver-ів і залишалися сумісними на різних етапах
> оптимізації. Наприклад, граф можна подати як список ребер, матрицю
> суміжності або список суміжності — і вибір формату залежить від задачі. Але
> конвертація між форматами зазвичай проста, потребує лише кількох рядків і має
> незначний вплив порівняно з оптимізацією. Тому я рекомендую фокусуватися на
> функціональності для вашого поточного solver-а і не ускладнювати цю частину.

### Клас розв’язувача

У багатьох реальних сценаріях оптимізації потрібно ітеративно уточнювати модель
і розв’язок. Наприклад, нові обмеження можуть з’явитися після показу первинного
розв’язку користувачу або іншому алгоритму (наприклад, фізичній симуляції, яку
занадто складно оптимізувати напряму). У таких випадках важлива гнучкість, тому
корисно інкапсулювати модель і solver в одному класі. Це дозволяє динамічно
додавати обмеження і повторно розв’язувати задачу без повної перебудови моделі,
а також потенційно використовувати warm-start.

Нижче — `KnapsackSolver`, який інкапсулює побудову та розв’язання задачі. Ми
також розбиваємо побудову моделі на менші методи, що корисно для складних
моделей.

```python
class KnapsackSolver:
    def __init__(self, instance: KnapsackInstance, config: KnapsackSolverConfig):
        self.instance = instance
        self.config = config
        self.model = cp_model.CpModel()
        self.n = len(instance.weights)
        self.x = [self.model.new_bool_var(f"x_{i}") for i in range(self.n)]
        self._build_model()
        self.solver = cp_model.CpSolver()

    def _add_constraints(self):
        used_weight = sum(
            weight * x_i for weight, x_i in zip(self.instance.weights, self.x)
        )
        self.model.add(used_weight <= self.instance.capacity)

    def _add_objective(self):
        self.model.maximize(
            sum(value * x_i for value, x_i in zip(self.instance.values, self.x))
        )

    def _build_model(self):
        self._add_constraints()
        self._add_objective()

    def solve(self, time_limit: float | None = None) -> KnapsackSolution:
        self.solver.parameters.max_time_in_seconds = time_limit if time_limit else self.config.time_limit
        self.solver.parameters.relative_gap_limit = self.config.opt_tol
        self.solver.parameters.log_search_progress = self.config.log_search_progress
        status = self.solver.solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return KnapsackSolution(
                selected_items=[
                    i for i in range(self.n) if self.solver.value(self.x[i])
                ],
                objective=self.solver.objective_value,
                upper_bound=self.solver.best_objective_bound,
            )
        return KnapsackSolution(
            selected_items=[], objective=0, upper_bound=float("inf")
        )

    def prohibit_combination(self, item_a: int, item_b: int):
        """
        Забороняє комбінацію двох предметів у розв’язку.
        Це корисно, якщо після показу розв’язку з’ясувалося, що ці предмети
        не можна пакувати разом. Після цього просто викликаємо `solve` знову.
        """
        self.model.add(self.x[item_a] + self.x[item_b] <= 1)
```

На перший погляд це може здаватися громіздким — треба створити solver-об’єкт, а
потім викликати `solve`. Але така структура підходить для багатьох кейсів, і я
використовую її варіанти у більшості проєктів. Для простих випадків можна додати
обгортку-функцію.

```python
instance = KnapsackInstance(weights=[1, 2, 3], values=[4, 5, 6], capacity=3)
config = KnapsackSolverConfig(time_limit=10, opt_tol=0.01, log_search_progress=True)
solver = KnapsackSolver(instance, config)
solution = solver.solve()

print(solution)
# Припустимо, симуляція показала, що перші два предмети не можна пакувати разом.
solver.prohibit_combination(0, 1)

# Розв’язуємо знову, але лише 5 секунд.
solution = solver.solve(time_limit=5)
print(solution)
```

Хоча повторне використання класу здебільшого економить на повторній побудові
моделі, кожен `solve` все одно стартує новий пошук. Проте інкрементальне
уточнення моделі в межах одного екземпляра solver-а більш інтуїтивне для коду,
ніж повністю нова постановка на кожній ітерації. Ба більше, як ми побачимо
далі, це дозволяє покращити продуктивність через warm-start.

### Покращення продуктивності через warm-start

Оскільки solver-клас зберігає стан і пам’ятає попередні ітерації, можна легко
додавати оптимізації, які було б складно реалізувати у статичній функції. Одна
з них — warm-start, коли solver використовує попередній розв’язок як підказку
для наступного запуску. Це може суттєво пришвидшити розв’язання, бо solver може
використати попередній розв’язок як хороший старт для «ремонту», навіть якщо
він став недопустимим через нові обмеження. Це працює лише тоді, коли нові
обмеження не змінюють задачу кардинально.

Оскільки ремонт недопустимої підказки може бути дорогим, CP-SAT обережно до
цього ставиться. Ви можете наказати CP-SAT ремонтувати підказку через
`solver.parameters.repair_hint = True`, а також керувати лімітом конфліктів через
`solver.parameters.hint_conflict_limit`.

Приклад:

```python
class KnapsackSolver:
    # ...

    def _set_solution_as_hint(self):
        """Використати поточний розв’язок як підказку для наступного solve."""
        for i, v in enumerate(self.model.proto.variables):
            v_ = self.model.get_int_var_from_proto_index(i)
            assert v.name == v_.name, "Variable names should match"
            self.model.add_hint(v_, self.solver.value(v_))
        # Сказати CP-SAT ремонтувати підказку
        self.solver.parameters.repair_hint = True
        self.solver.parameters.hint_conflict_limit = 20

    def solve(self, time_limit: float | None = None) -> KnapsackSolution:
        self.solver.parameters.max_time_in_seconds = time_limit if time_limit else self.config.time_limit
        self.solver.parameters.relative_gap_limit = self.config.opt_tol
        self.solver.parameters.log_search_progress = self.config.log_search_progress
        status = self.solver.solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Є розв’язок — використовуємо його як підказку
            self._set_solution_as_hint()
            return KnapsackSolution(
                selected_items=[
                    i for i in range(self.n) if self.solver.value(self.x[i])
                ],
                objective=self.solver.objective_value,
                upper_bound=self.solver.best_objective_bound,
            )
        return KnapsackSolution(
            selected_items=[], objective=0, upper_bound=float("inf")
        )

    # ...
```

Щоб ще покращити підхід, можна додати евристику для ремонту підказки. Допустима
підказка набагато корисніша, ніж та, яку треба сильно ремонтувати. Наприклад,
якщо підказка недопустима через заборонену комбінацію предметів, можна просто
викинути найменш цінний предмет.

> [!WARNING]
>
> Часта помилка при ітеративній оптимізації — додавати попередню межу як
> обмеження. Це може дозволити CP-SAT «продовжити» з попередньої межі, але часто
> обмежує його здатність знаходити кращі розв’язки, бо додає сильне обмеження,
> не пов’язане зі здійсненністю. Це може заважати внутрішнім алгоритмам,
> наприклад зменшувати ефективність лінійної релаксації.
>
> Якщо межі сильно впливають на продуктивність, краще використовувати callback,
> який перевіряє, чи поточна ціль достатньо близька до попередньої межі, і
> зупиняє пошук. Це менш агресивно, хоча callbacks додають накладні витрати.

|                                                                                                                                                                                                                                                                                                                                                                         ![Impact of Lower Bound Constraint on Relaxation](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/impact_lb_constraint_tsp.png)                                                                                                                                                                                                                                                                                                                                                                          |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Це зображення показує негативний вплив додавання обмеження нижньої межі в моделі TSP. Лінійна релаксація явно погіршується: якщо початкова релаксація збігалася з оптимальним розв’язком на 44 з 50 ребер, то релаксація з обмеженням нижньої межі (рівною оптимуму) збігається лише на 38 ребрах і має більше дробових значень. Розгалуження на певних ребрах може перестати бути корисним; якщо не розгалужуватися по «правильному» ребру, значення цілі часто не змінюється через домінування нижньої межі. Крім втрати інформативності релаксації, такі обмеження відомі як джерело числових нестабільностей (хоча CP-SAT, ймовірно, не страждає від цього через цілочисельну арифметику). |

### Змінна ціль / багатокритеріальна оптимізація

У реальних задачах цілі часто нечіткі. Зазвичай є кілька цілей із різним
пріоритетом, і їх складно об’єднати. Розгляньмо задачу рюкзака як логістичну
ситуацію: ми хочемо перевезти максимальну цінність за одну поїздку. Основна
ціль — максимізувати цінність. Але після того, як показали розв’язок, нас
можуть попросити знайти альтернативу, що заповнює машину меншою мірою, навіть
якщо це означає падіння цінності до 5%.

|                              [![xkcd grapfruit](https://imgs.xkcd.com/comics/fuck_grapefruit.png)](https://xkcd.com/388/)                              |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| Який фрукт найкращий? Багато задач багатокритеріальні й не мають однозначної мети. Автор [xkcd](https://xkcd.com/388/) (CC BY-NC 2.5) |

Щоб впоратися, можна оптимізувати у два етапи. Спочатку максимізуємо цінність
за обмеженням ваги. Потім додаємо обмеження, що цінність має бути не менше 95%
від початкового розв’язку, і змінюємо ціль на мінімізацію ваги. Цей процес можна
повторювати, досліджуючи фронт Парето. Більш складні задачі можна розв’язувати
схожими підходами.

Проблема такого методу — уникнути створення кількох моделей і старту з нуля
кожного разу. Оскільки у нас уже є розв’язок близький до нового, а зміна цілі
не впливає на здійсненність, це чудова можливість використовувати поточний
розв’язок як підказку.

Нижче — приклад розширення solver-класу для змінної цілі. Ми зберігаємо поточну
ціль у `_objective` і додаємо методи для максимізації цінності або мінімізації
ваги. Також додаємо метод для фіксації поточної цілі, щоб уникнути деградації,
і автоматично ставимо підказку при успішному розв’язанні.

```python
class MultiObjectiveKnapsackSolver:
    def __init__(self, instance: KnapsackInstance, config: KnapsackSolverConfig):
        self.instance = instance
        self.config = config
        self.model = cp_model.CpModel()
        self.n = len(instance.weights)
        self.x = [self.model.new_bool_var(f"x_{i}") for i in range(self.n)]
        self._objective = 0
        self._build_model()
        self.solver = cp_model.CpSolver()

    def set_maximize_value_objective(self):
        """Максимізувати цінність."""
        self._objective = sum(
            value * x_i for value, x_i in zip(self.instance.values, self.x)
        )
        self.model.maximize(self._objective)

    def set_minimize_weight_objective(self):
        """Мінімізувати вагу."""
        self._objective = sum(
            weight * x_i for weight, x_i in zip(self.instance.weights, self.x)
        )
        self.model.minimize(self._objective)

    def _set_solution_as_hint(self):
        """Використати поточний розв’язок як підказку для наступного solve."""
        for i, v in enumerate(self.model.proto.variables):
            v_ = self.model.get_int_var_from_proto_index(i)
            assert v.name == v_.name, "Variable names should match"
            self.model.add_hint(v_, self.solver.value(v_))

    def fix_current_objective(self, ratio: float = 1.0):
        """Зафіксувати поточну ціль, щоб уникнути деградації."""
        if ratio == 1.0:
            self.model.add(self._objective == self.solver.objective_value)
        elif ratio > 1.0:
            self.model.add(self._objective <= ceil(self.solver.objective_value * ratio))
        else:
            self.model.add(
                self._objective >= floor(self.solver.objective_value * ratio)
            )

    def _add_constraints(self):
        """Додаємо обмеження ваги."""
        used_weight = sum(
            weight * x_i for weight, x_i in zip(self.instance.weights, self.x)
        )
        self.model.add(used_weight <= self.instance.capacity)

    def _build_model(self):
        """Будуємо модель з обмеженнями та ціллю."""
        self._add_constraints()
        self.set_maximize_value_objective()

    def solve(self, time_limit: float | None = None) -> KnapsackSolution:
        """Розв’язуємо задачу рюкзака та повертаємо розв’язок."""
        self.solver.parameters.max_time_in_seconds = time_limit if time_limit else self.config.time_limit
        self.solver.parameters.relative_gap_limit = self.config.opt_tol
        self.solver.parameters.log_search_progress = self.config.log_search_progress
        status = self.solver.solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            self._set_solution_as_hint()
            return KnapsackSolution(
                selected_items=[
                    i for i in range(self.n) if self.solver.value(self.x[i])
                ],
                objective=self.solver.objective_value,
                upper_bound=self.solver.best_objective_bound,
            )
            )
        return KnapsackSolution(
            selected_items=[], objective=0, upper_bound=float("inf")
        )
```

Можна використовувати `MultiObjectiveKnapsackSolver` так:

```python
config = KnapsackSolverConfig(time_limit=15, opt_tol=0.01, log_search_progress=True)
solver = MultiObjectiveKnapsackSolver(instance, config)
solution_1 = solver.solve()

# зберігаємо щонайменше 95% поточного значення цілі
solver.fix_current_objective(0.95)
# змінюємо ціль на мінімізацію ваги
solver.set_minimize_weight_objective()
solution_2 = solver.solve(time_limit=10)
```

Існують більш просунуті й точні методи обчислення
[фронту Парето](https://en.wikipedia.org/wiki/Pareto_front), але
[багатокритеріальна оптимізація](https://en.wikipedia.org/wiki/Multi-objective_optimization)
— окрема складна сфера досліджень. Якщо ваша задача вже складна з однією
ціллю, додаткові цілі лише підвищать складність.

Підхід лексикографічної оптимізації (з послабленням) або об’єднання кількох
цілей в одну, наприклад через ваги, часто є розумним компромісом. Також можна
використовувати евристики, щоб досліджувати простір розв’язків навколо
початкового розв’язку з CP-SAT.

Втім, багатокритеріальна оптимізація залишається складною темою, і навіть
експерти покладаються на суттєвий trial-and-error, бо компроміси часто
неминучі.

### Контейнери змінних

У складних моделях змінні відіграють ключову роль і можуть охоплювати всю
модель. Для простих моделей достатньо списку або словника, але у складних це
стає громіздким і схильним до помилок. Один неправильний індекс може
спричинити тонкі помилки, які складно відстежити.

Оскільки змінні — основа моделі, їх рефакторинг стає складнішим із ростом
моделі. Тому важливо рано налагодити надійну систему керування. Інкапсуляція
змінних у класі забезпечує правильний доступ до них. Це також дозволяє легко
додавати нові змінні або змінювати логіку без переписування всієї моделі.

Крім того, зрозумілі методи-запити допомагають підтримувати читабельність
обмежень. Читабельні обмеження без складних схем доступу гарантують, що вони
відповідають задуму.

Нижче ми вводимо `_ItemSelectionVars` як контейнер для змінних вибору. Цей
клас створює змінні і має допоміжні методи для взаємодії з ними, що підвищує
читабельність і підтримуваність.

```python
from typing import Generator, Tuple, List


class _ItemSelectionVars:
    def __init__(self, instance: KnapsackInstance, model: cp_model.CpModel, var_name: str = "x"):
        self.instance = instance
        self.x = [model.new_bool_var(f"{var_name}_{i}") for i in range(len(instance.weights))]

    def __getitem__(self, i: int) -> cp_model.IntVar:
        return self.x[i]

    def packs_item(self, i: int) -> cp_model.IntVar:
        return self.x[i]

    def extract_packed_items(self, solver: cp_model.CpSolver) -> List[int]:
        return [i for i, x_i in enumerate(self.x) if solver.value(x_i)]

    def used_weight(self) -> cp_model.LinearExprT:
        return sum(weight * x_i for weight, x_i in zip(self.instance.weights, self.x))

    def packed_value(self) -> cp_model.LinearExprT:
        return sum(value * x_i for value, x_i in zip(self.instance.values, self.x))

    def iter_items(
        self,
        weight_lb: float = 0.0,
        weight_ub: float = float("inf"),
        value_lb: float = 0.0,
        value_ub: float = float("inf"),
    ) -> Generator[Tuple[int, cp_model.IntVar], None, None]:
        """
        Приклад складнішого методу-запиту для фільтрації предметів.
        """
        for i, (weight, x_i) in enumerate(zip(self.instance.weights, self.x)):
            if (
                weight_lb <= weight <= weight_ub
                and value_lb <= self.instance.values[i] <= value_ub
            ):
                yield i, x_i

```

Цей клас можна використовувати в `KnapsackSolver`, який відповідає за високий
рівень логіки (що має робити модель), а деталі сховати в контейнері.

```python
class KnapsackSolver:
    def __init__(self, instance: KnapsackInstance, config: KnapsackSolverConfig):
        self.instance = instance
        self.config = config
        self.model = cp_model.CpModel()
        self._item_vars = _ItemSelectionVars(instance, self.model)
        self._build_model()
        self.solver = cp_model.CpSolver()

    def _add_constraints(self):
        self.model.add(self._item_vars.used_weight() <= self.instance.capacity)

    def _add_objective(self):
        self.model.maximize(self._item_vars.packed_value())

    def _build_model(self):
        self._add_constraints()
        self._add_objective()

    def solve(self) -> KnapsackSolution:
        self.solver.parameters.max_time_in_seconds = self.config.time_limit
        self.solver.parameters.relative_gap_limit = self.config.opt_tol
        self.solver.parameters.log_search_progress = self.config.log_search_progress
        status = self.solver.solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return KnapsackSolution(
                selected_items=self._item_vars.extract_packed_items(self.solver),
                objective=self.solver.objective_value,
                upper_bound=self.solver.best_objective_bound,
            )
        return KnapsackSolution(
            selected_items=[], objective=0, upper_bound=float("inf")
        )

    def prohibit_combination(self, item_a: int, item_b: int):
        self.model.add_at_most_one(self._item_vars.packs_item(item_a),
        self._item_vars.packs_item(item_b))
```

Наприклад, `self.model.add(self._item_vars.used_weight() <= self.instance.capacity)`
тепер прямо виражає зміст обмеження, що підвищує читабельність і зменшує
ймовірність помилок. У контейнері можна також приховати оптимізації, не змінюючи
високорівневий код solver-а. Наприклад, контейнер може автоматично замінювати
змінні предметів, які не можуть вміститися в рюкзак, на константи.

Можна повторно використовувати тип контейнера, якщо з’явиться два рюкзаки. Код
нижче показує, як розширити solver на два рюкзаки без втрати читабельності.

```python
class KnapsackSolver:
    def __init__(self, # ...
    ):
        #...
        self._knapsack_a = _ItemSelectionVars(instance, self.model, var_name="x1")
        self._knapsack_b = _ItemSelectionVars(instance, self.model, var_name="x2")
        #...

    def _add_constraints(self):
        self.model.add(self._knapsack_a.used_weight() <= self.instance.capacity_1)
        self.model.add(self._knapsack_b.used_weight() <= self.instance.capacity_2)
        self.model.add(self._knapsack_a.used_weight() + self._knapsack_b.used_weight() <= self.instance.capacity_total)
        # Забороняємо пакувати предмет у два рюкзаки
        for i in range(len(instance.weights)):
            self.model.add_at_most_one(self._knapsack_a.packs_item(i), self._knapsack_b.packs_item(i))

    def _add_objective(self):
        self.model.maximize(self._knapsack_a.packed_value() + self._knapsack_b.packed_value())
```

> [!WARNING]
>
> Не створюйте контейнерний клас для простих моделей, якщо він лише обгортає
> список або словник без додаткової логіки. У таких випадках простий список або
> словник читається легше й коротше. Те саме стосується окремих змінних, яким
> не потрібен контейнер.

### Ліниве створення змінних

У моделях з великою кількістю допоміжних змінних часто реально використовується
лише невелика підмножина. Спроба створювати лише потрібні змінні заздалегідь
може ускладнити код і при подальших розширеннях ще більше ускладнюється. Тут
допомагає ліниве створення змінних: вони створюються лише при доступі до них.
Це гарантує, що створюються лише потрібні змінні, економлячи пам’ять та час.
Якщо зрештою використовується більшість змінних, це може бути дорожчим, але коли
потрібні лише кілька — економія суттєва.

Для ілюстрації введемо `_CombiVariables`. Він управляє допоміжними змінними для
пар предметів, щоб задавати бонус за пакування разом. Теоретично кількість пар
квадратична, але на практиці релевантні лише деякі. Ліниве створення економить
ресурси.

```python
class _CombiVariables:
    def __init__(
        self,
        instance: KnapsackInstance,
        model: cp_model.CpModel,
        item_vars: _ItemSelectionVars,
    ):
        self.instance = instance
        self.model = model
        self.item_vars = item_vars
        self.bonus_vars = {}

    def __getitem__(self, item_pair: Tuple[int, int]) -> cp_model.IntVar:
        i, j = sorted(item_pair)
        if (i, j) not in self.bonus_vars:
            var = self.model.NewBoolVar(f"bonus_{i}_{j}")
            self.model.add(
                self.item_vars.packs_item(i) + self.item_vars.packs_item(j) >= 2 * var
            )
            self.bonus_vars[(i, j)] = var
        return self.bonus_vars[(i, j)]
```

У `KnapsackSolver` можна поводитися так, ніби всі змінні існують, не турбуючись
про оптимізацію. Зверніть увагу: ціль ми будуємо у `solve`, бо бонуси змінюють
ціль. Також, інкапсулювавши змінні у `_ItemSelectionVars`, ми можемо легко
передавати їх іншим компонентам.

```python
class KnapsackSolver:
    def __init__(self, instance: KnapsackInstance, config: KnapsackSolverConfig):
        self.instance = instance
        self.config = config
        self.model = cp_model.CpModel()
        self._item_vars = _ItemSelectionVars(instance, self.model)
        self._bonus_vars = _CombiVariables(instance, self.model, self._item_vars)
        self._objective_terms = [self._item_vars.packed_value()]
        self.solver = cp_model.CpSolver()

    def solve(self) -> KnapsackSolution:
        self.model.maximize(sum(self._objective_terms))
        self.solver.parameters.max_time_in_seconds = self.config.time_limit
        self.solver.parameters.relative_gap_limit = self.config.opt_tol
        self.solver.parameters.log_search_progress = self.config.log_search_progress
        status = self.solver.solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return KnapsackSolution(
                selected_items=self._item_vars.extract_packed_items(self.solver),
                objective=self.solver.objective_value,
                upper_bound=self.solver.best_objective_bound,
            )
        return KnapsackSolution(
            selected_items=[], objective=0, upper_bound=float("inf")
        )

    def add_bonus(self, item_a: int, item_b: int, bonus: int):
        bonus_var = self._bonus_vars[(item_a, item_b)]
        self._objective_terms.append(bonus * bonus_var)
```

> [!TIP]
>
> Якщо ми точно знаємо, що `add_bonus` викликається для пари лише один раз,
> можна не зберігати bonus_vars, а створювати змінну і додавати в ціль одразу.
> Немає потреби зберігати handle, якщо він не потрібен пізніше.

### Підмоделі

Ми можемо покращити підхід, інкапсулюючи цілі секції моделі в окремі підмоделі.
Це корисно для складних моделей, де компоненти слабо пов’язані. Розбиття на
підмоделі підвищує модульність і підтримуваність. Підмоделі комунікують із
головною моделлю через спільні змінні, приховуючи деталі, як-от допоміжні
змінні. Якщо вимоги зміняться, можна переписати одну підмодель без впливу на
інші. Також часто логіка повторюється у різних оптимізаційних задачах, тому
бібліотека підмоделей дозволяє швидко збирати нові моделі з компонентів.

Наприклад, кусково-лінійні функції можна оформити як підмодель, як у класі
`PiecewiseLinearConstraint` у
[piecewise_linear_function.py](https://github.com/d-krupke/cpsat-primer/blob/main/utils/piecewise_functions/piecewise_linear_function.py).
Кожна підмодель керує однією функцією, взаємодіючи з моделлю через `x` та `y`.
Інкапсуляція робить логіку повторно використовуваною і тестованою окремо.

```python
from ortools.sat.python import cp_model

requirements_1 = (3, 5, 2)
requirements_2 = (2, 1, 3)

model = cp_model.CpModel()
buy_1 = model.new_int_var(0, 1_500, "buy_1")
buy_2 = model.new_int_var(0, 1_500, "buy_2")
buy_3 = model.new_int_var(0, 1_500, "buy_3")

produce_1 = model.new_int_var(0, 300, "produce_1")
produce_2 = model.new_int_var(0, 300, "produce_2")

model.add(produce_1 * requirements_1[0] + produce_2 * requirements_2[0] <= buy_1)
model.add(produce_1 * requirements_1[1] + produce_2 * requirements_2[1] <= buy_2)
model.add(produce_1 * requirements_1[2] + produce_2 * requirements_2[2] <= buy_3)

# PiecewiseLinearFunction і PiecewiseLinearConstraint — у utils
from piecewise_functions import PiecewiseLinearFunction, PiecewiseLinearConstraint

# Функції витрат
costs_1 = [(0, 0), (1000, 400), (1500, 1300)]
costs_2 = [(0, 0), (300, 300), (700, 500), (1200, 600), (1500, 1100)]
costs_3 = [(0, 0), (200, 400), (500, 700), (1000, 900), (1500, 1500)]

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
gain_1 = [(0, 0), (100, 800), (200, 1600), (300, 2000)]
gain_2 = [(0, 0), (80, 1000), (150, 1300), (200, 1400), (300, 1500)]

f_gain_1 = PiecewiseLinearFunction(
    xs=[x for x, y in gain_1], ys=[y for x, y in gain_1]
)
f_gain_2 = PiecewiseLinearFunction(
    xs=[x for x, y in gain_2], ys=[y for x, y in gain_2]
)

# y >= f(x) для витрат
x_costs_1 = PiecewiseLinearConstraint(model, buy_1, f_costs_1, upper_bound=False)
x_costs_2 = PiecewiseLinearConstraint(model, buy_2, f_costs_2, upper_bound=False)
x_costs_3 = PiecewiseLinearConstraint(model, buy_3, f_costs_3, upper_bound=False)

# y <= f(x) для доходу
x_gain_1 = PiecewiseLinearConstraint(model, produce_1, f_gain_1, upper_bound=True)
x_gain_2 = PiecewiseLinearConstraint(model, produce_2, f_gain_2, upper_bound=True)

# Максимізуємо дохід мінус витрати
model.maximize(x_gain_1.y + x_gain_2.y - (x_costs_1.y + x_costs_2.y + x_costs_3.y))
```

Тестування складних оптимізаційних моделей часто складне, бо результати можуть
змінюватися навіть через дрібні зміни. Навіть якщо тест знаходить помилку,
виявити джерело складно. Винесення елементів у підмоделі дозволяє тестувати їх
окремо, забезпечуючи коректність перед інтеграцією.

Підмоделі зазвичай значно простіші за основну задачу, тож оптимізуються швидко,
а отже тести працюють швидко.

```python
from ortools.sat.python import cp_model

def test_piecewise_linear_upper_bound_constraint():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 20, "x")
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 5])

    # Використовуємо підмодель
    c = PiecewiseLinearConstraint(model, x, f, upper_bound=True)
    model.maximize(c.y)

    # Перевіряємо поведінку
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL
    assert solver.value(c.y) == 10
    assert solver.value(x) == 10
```

Альтернативно можна тестувати здійсненність/недопустимість, особливо якщо
підмодель не є оптимізаційною задачею сама по собі.

```python
from ortools.sat.python import cp_model

def test_piecewise_linear_upper_bound_constraint_via_fixation():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 20, "x")
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 5])
    c = PiecewiseLinearConstraint(model, x, f, upper_bound=True)

    # Фіксуємо змінні на конкретних значеннях
    model.add(x == 10)
    model.add(c.y == 10)

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.OPTIMAL, "Модель має бути допустимою"

def test_piecewise_linear_upper_bound_constraint_via_fixation_infeasible():
    model = cp_model.CpModel()
    x = model.new_int_var(0, 20, "x")
    f = PiecewiseLinearFunction(xs=[0, 10, 20], ys=[0, 10, 5])
    c = PiecewiseLinearConstraint(model, x, f, upper_bound=True)

    # Фіксуємо значення, що порушують обмеження
    model.add(x == 10)
    model.add(c.y == 11)

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    assert status == cp_model.INFEASIBLE, "Модель має бути недопустимою"
```

### Вбудовування CP-SAT у застосунок через multiprocessing

Якщо ви хочете вбудувати CP-SAT у застосунок для потенційно довгих
оптимізаційних задач, можна використати callbacks, щоб показувати прогрес і
дозволяти раннє припинення. Проте застосунок може реагувати лише в момент
callback, а вони не завжди викликаються часто. Це може спричинити затримки і
погано підходить для GUI чи API.

Альтернатива — запускати solver в окремому процесі й спілкуватися з ним через
pipe. Це дозволяє перервати solver у будь-який момент і дає миттєву реакцію.
Python multiprocessing пропонує досить прості інструменти для цього.
[Цей приклад](https://github.com/d-krupke/cpsat-primer/blob/main/examples/embedding_cpsat/)
показує такий підхід. Щоб масштабувати його, зазвичай потрібна черга задач, де
solver запускається воркерами. Multiprocessing все одно корисний, бо дозволяє
воркеру залишатися чутливим до сигналів зупинки, поки solver працює.

| ![Interactive Solver with Streamlit using multiprocessing](https://github.com/d-krupke/cpsat-primer/blob/main/images/streamlit_solver.gif) |
| :----------------------------------------------------------------------------------------------------------------------------------------: |
|                                _Використовуючи multiprocessing, можна створити чутливий інтерфейс для solver-а._                                 |

[@oulianov](https://github.com/oulianov) розгорнув це
[тут](https://cpsat-embeddings-demo.streamlit.app/), щоб можна було спробувати
у браузері.

---
