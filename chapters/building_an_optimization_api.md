<a name="building_an_optimization_api"></a>

# Побудова оптимізаційного API (ЧЕРНЕТКА)

У цьому розділі ми створимо базовий сервіс оптимізації, який виконує обчислення
на кластері, а не на стороні клієнта. Цей сервіс можна доповнити простим
фронтендом, що дозволить нетехнічним користувачам отримати доступ до можливостей
оптимізації. Інкапсулювавши код оптимізації у зручному API, ви спрощуєте
інтеграцію в більші системи та відокремлюєте розробку алгоритмів від розгортання.

Хоч цей розділ і не охоплює всі аспекти побудови продакшн-готового API, він
розгляне багато важливих моментів. Цих базових знань буде достатньо, щоб
ефективно співпрацювати з інженерами інтеграції й завершити деталі реалізації,
або навіть зробити це самостійно.

Щоб проілюструвати ці принципи, ми розробимо просту оптимізаційну модель для
задачі комівояжера (TSP). Користувачі зможуть надсилати екземпляр TSP, а API
повертатиме розв’язок. Основна складність, порівняно з багатьма іншими API,
полягає в тому, що TSP — NP-складна задача, і розв’язувачу CP-SAT можуть
знадобитися кілька хвилин навіть для помірних за розміром екземплярів. Крім
того, ми не можемо запускати розв’язувач на вебсервері; натомість треба
розподіляти обчислення по кластеру. Якщо одночасно надходить багато запитів,
окремі запити можуть чекати, перш ніж обчислення стартують.

Отже, це не буде просте API «надіслав запит — отримав відповідь». Натомість ми
реалізуємо чергу задач, повертатимемо ID задачі, і користувачі зможуть
перевіряти статус обчислення та отримувати результат після готовності. Щоб
покращити досвід користувача, ми дозволимо вказувати webhook URL, на який
надішлемо повідомлення після завершення обчислення.

Ви зможете легко адаптувати це API до власних оптимізаційних задач і швидко
надавати свої алгоритми колегам.

## Визначення ключових ендпойнтів

Перш ніж почати кодити, потрібно визначити ендпойнти, які надаватиме API, щоб
розуміти, що саме реалізовувати. Цим можна безпосередньо поділитися з колегами,
які робитимуть фронтенд чи інші частини системи, щоб вони знали, чого очікувати,
і могли починати свою частину. Зазвичай деталі payload-ів легко змінити пізніше,
але змінювати загальну логіку API значно складніше.

Базові операції, які ми підтримуватимемо:

1. **POST /jobs**: ендпойнт приймає JSON-пейлоад із екземпляром TSP. API створює
   нову задачу, зберігає екземпляр і повертає ID задачі. Пейлоад також дозволяє
   вказати webhook URL, який буде викликано після завершення обчислень.
2. **GET /jobs/{task_id}**: повертає статус задачі з заданим ID.
3. **GET /jobs/{task_id}/solution**: повертає розв’язок задачі з заданим ID, коли
   він доступний.
4. **DELETE /jobs/{task_id}**: скасовує задачу з заданим ID.
5. **GET /jobs**: повертає список усіх задач із їхніми статусами та метаданими.

Визначивши ці ендпойнти, ми гарантуємо, що API буде надійним і зможе виконувати
ключові функції керування та розв’язання екземплярів TSP. Така структура
полегшує взаємодію користувачів — від подання задач до отримання розв’язків і
моніторингу статусу запитів.

Після успішного запуску сервісу оптимізації TSP можна очікувати запитів на
розширення оптимізаційних можливостей до інших задач. Тому варто додати
префікс `/tsp_solver/v1` до всіх ендпойнтів, щоб полегшити майбутні розширення
API додатковими розв’язувачами, наприклад `knapsack_solver/v1` або
`/tsp_solver/v2_experimental`.

Можна запитати, чому не зробити окремий проєкт для кожного розв’язувача, а потім
«склеїти» їх на верхньому рівні. Причина в тому, що ми хочемо ділити одну й ту
саму інфраструктуру між усіма розв’язувачами, особливо чергу задач і кластер
воркерів. Це не лише простіше в підтримці, а й дешевше, бо ресурси спільні. Тому
логічно тримати їх в одному проєкті. Водночас має сенс відокремити самі
алгоритми від API-коду й лише імпортувати їх у API-проєкт. Ми цього не робитимемо
в цьому розділі, але я особисто надаю перевагу максимально можливому
відокремленню алгоритмів, бо вони й так достатньо складні.

## Архітектура

Окресливши вимоги, розгляньмо архітектуру системи. Вона включатиме такі
компоненти:

1. **FastAPI для реалізації ендпойнтів**: FastAPI — сучасний високопродуктивний
   вебфреймворк для побудови API на Python. Ми використаємо FastAPI для
   визначення ендпойнтів і обробки HTTP-запитів завдяки його простоті, швидкості
   та автоматичній інтерактивній документації.

2. **Redis як база даних і інтерфейс комунікації з воркерами**: Redis — це
   in-memory сховище структур даних, яке може працювати як база даних, кеш і
   брокер повідомлень. Ми використовуватимемо Redis через його швидкість і
   ефективність у зберіганні та передачі задач і розв’язків, що забезпечує
   швидкий доступ і автоматичне видалення даних, коли вони більше не потрібні.

3. **Воркери, керовані RQ (Redis Queue)**: RQ — проста Python-бібліотека для
   постановки задач у чергу та обробки їх у фоні воркерами. Це дозволяє API
   обробляти задачі асинхронно, виносячи обчислювально дорогі процеси у фон,
   і таким чином підвищувати швидкодію API.

Щоб легко керувати цими компонентами, ми використаємо Docker і Docker Compose
для контейнеризації API, Redis і воркерів. Це дозволить швидко підняти сервіс
локально або в хмарі.

> [!WARNING]
>
> У цьому розділі ми ігноруємо аспекти безпеки. Цей сервіс має використовуватися
> лише всередині власної мережі й не має бути доступним з інтернету. Якщо ви
> хочете його публікувати, потрібно додати автентифікацію, rate limiting та інші
> заходи безпеки.

### Структура проєкту

Оскільки у цьому проєкті лише один розв’язувач, ми не будемо відокремлювати його
від API і не інкапсулюватимемо API, а використаємо просту пласку структуру.
Повний проєкт можна знайти в
[./examples/optimization_api](https://github.com/d-krupke/cpsat-primer/blob/main/examples/optimization_api).

```text
├── app
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   ├── main.py
│   ├── models.py
│   ├── solver.py
│   └── tasks.py
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

Швидко пройдемося по складових проєкту:

1. **Requirements**: ми визначаємо необхідні Python-пакети у файлі
   `requirements.txt`, щоб середовище можна було легко налаштувати та
   відтворити. Цей файл містить усі залежності проєкту. `requirements.txt`
   дещо застарілий підхід, але це найпростіший спосіб встановити залежності в
   контейнері.

2. **Docker-оточення**:

   - `Dockerfile`: визначає Docker-образ і налаштування середовища для API. Це
     гарантує, що застосунок працює однаково на різних машинах.
   - `docker-compose.yml`: налаштовує сервіси проєкту, включно з API, Redis і
     воркерами. Docker Compose спрощує керування кількома контейнерами,
     забезпечуючи коректну збірку та порядок запуску. Команда
     `docker-compose up -d --build` підніме всю систему.

3. **Реалізація розв’язувача**:

   - `./app/solver.py`: модуль містить реалізацію TSP-розв’язувача на CP-SAT, а
     також визначає очікувані вхідні та вихідні дані.

4. **Моделі запитів і відповідей**:

   - `./app/models.py`: модуль визначає додаткові моделі даних для запитів і
     відповідей API.

5. **База даних**:

   - `./app/db.py`: модуль реалізує проксі-клас для взаємодії з Redis, абстрагуючи
     операції БД для збереження та отримання запитів, статусів і розв’язків.

6. **Конфігурація**:

   - `./app/config.py`: модуль містить конфігураційні функції для налаштування
     підключення до БД та черги задач. Централізована конфігурація означає, що
     інші частини застосунку не повинні керувати деталями з’єднання, що робить
     код більш модульним і зручним у підтримці.

7. **Задачі**:

   - `./app/tasks.py`: модуль визначає задачі, що передаються воркерам; наразі
     це лише задача оптимізації. Вебсерверу потрібне лише посилання на функції
     задач, щоб ставити їх у чергу, але він не запускає цей код. Для воркерів
     цей файл є точкою входу.

8. **API**:
   - `./app/main.py`: модуль реалізує FastAPI-застосунок з маршрутами для
     подання задач, перевірки статусів, отримання розв’язків і скасування задач.
     Це точка входу для вебсерверу.

### Запуск застосунку

Для запуску застосунку ми використовуємо Docker і Docker Compose, щоб зібрати і
запустити контейнери. Це гарантує коректне налаштування API та його залежностей.
Після запуску контейнерів можна взаємодіяти з API через HTTP-запити.

## Docker-оточення

Ми використовуємо Docker, щоб забезпечити узгоджене середовище розробки та
продакшну. Docker дозволяє запакувати застосунок разом з усіма залежностями в
стандартизований контейнер. Docker Compose використовується для керування
мультиконтейнерними застосунками. Оскільки вебсервер і воркери по суті
використовують один і той самий код, але з різними точками входу, ми можемо
використати один Docker-образ для обох. Різні точки входу задаються в
`docker-compose.yml`.

### Dockerfile

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy the current directory contents into the container at /app
COPY ./app /app
```

### docker-compose.yml

Щоб підняти композицію контейнерів з API, Redis і воркерами, використовуємо
наступний файл `docker-compose.yml`:

```yaml
services:
  optimization_api_fastapi: # The web server
    build: .
    container_name: optimization_api_fastapi
    ports: # exposing the API on port 80. Change this if you want to use a different port.
      - "80:80"
    depends_on: # Ensuring that the web server starts after the database.
      - optimization_api_redis
    command: python3 -m uvicorn main:app --host 0.0.0.0 --port 80 --reload

  optimization_api_redis: # The database. We use the official Redis image.
    image: redis:latest
    container_name: optimization_api_redis

  optimization_api_worker: # The worker
    build: .
    command: # Running this command will make our container a worker instead of a web server.
      rq worker --with-scheduler --url redis://optimization_api_redis:6379/1
    depends_on: # Ensuring that the worker starts after the database, as it needs to connect to it.
      - optimization_api_redis
    deploy:
      replicas: 2 # Adding two workers for parallel processing
```

Файл `docker-compose.yml` налаштовує три сервіси:

- `optimization_api_fastapi`: збирає FastAPI-застосунок, відкриває порт 80 і
  гарантує запуск після готовності Redis.
- `optimization_api_redis`: підіймає Redis з офіційного образу. Потрібно лише
  пам’ятати назву контейнера для підключення.
- `optimization_api_worker`: запускає воркер, який обробляє задачі з черги.
  Кількість воркерів можна масштабувати, збільшуючи кількість реплік.
  Теоретично їх можна запускати на різних машинах для горизонтального
  масштабування.

### Розв’язувач

У цьому розділі ми розглянемо реалізацію оптимізаційного алгоритму, який будемо
деплоїти як API. Зокрема, зосередимося на простій реалізації задачі комівояжера
(TSP) з використанням обмеження `add_circuit` із CP-SAT в OR-Tools.

Розв’язувач є ядром застосунку і відповідає за пошук оптимального розв’язку для
екземпляра TSP, наданого користувачем. Алгоритм реалізовано безпосередньо в
API-проєкті для простоти. Однак для складніших оптимізаційних алгоритмів
рекомендовано відокремлювати алгоритм в окремий модуль або проєкт. Такий поділ
полегшує ізольоване тестування й бенчмаркінг алгоритму та покращує процес
розробки, особливо коли в команді різні підгрупи підтримують API та алгоритм
оптимізації.

```python
# ./app/solver.py
from typing import Callable
from ortools.sat.python import cp_model
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# A precise definition of the input and output data for the TSP solver.
# ---------------------------------------------------------------------


class DirectedEdge(BaseModel):
    source: int = Field(..., ge=0, description="The source node of the edge.")
    target: int = Field(..., ge=0, description="The target node of the edge.")
    cost: int = Field(..., ge=0, description="The cost of traversing the edge.")


class TspInstance(BaseModel):
    num_nodes: int = Field(
        ..., gt=0, description="The number of nodes in the TSP instance."
    )
    edges: list[DirectedEdge] = Field(
        ..., description="The directed edges of the TSP instance."
    )


class OptimizationParameters(BaseModel):
    timeout: int = Field(
        default=60,
        gt=0,
        description="The maximum time in seconds to run the optimization.",
    )


class TspSolution(BaseModel):
    node_order: list[int] | None = Field(
        ..., description="The order of the nodes in the solution."
    )
    cost: float = Field(..., description="The cost of the solution.")
    lower_bound: float = Field(..., description="The lower bound of the solution.")
    is_infeasible: bool = Field(
        default=False, description="Whether the instance is infeasible."
    )


# ---------------------------------------------------------------------
# The TSP solver implementation using the CP-SAT solver from OR-Tools.
# ---------------------------------------------------------------------


class TspSolver:
    def __init__(
        self, tsp_instance: TspInstance, optimization_parameters: OptimizationParameters
    ):
        self.tsp_instance = tsp_instance
        self.optimization_parameters = optimization_parameters
        self.model = cp_model.CpModel()
        self.edge_vars = {
            (edge.source, edge.target): self.model.new_bool_var(
                f"x_{edge.source}_{edge.target}"
            )
            for edge in tsp_instance.edges
        }
        self.model.minimize(
            sum(
                edge.cost * self.edge_vars[(edge.source, edge.target)]
                for edge in tsp_instance.edges
            )
        )
        self.model.add_circuit(
            [(source, target, var) for (source, target), var in self.edge_vars.items()]
        )

    def solve(self, log_callback: Callable[[str], None] | None = None):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.optimization_parameters.timeout
        if log_callback:
            solver.parameters.log_search_progress = True
            solver.log_callback = log_callback
        status = solver.Solve(self.model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return TspSolution(
                node_order=[
                    source
                    for (source, target), var in self.edge_vars.items()
                    if solver.value(var)
                ],
                cost=solver.objective_value,
                lower_bound=solver.best_objective_bound,
            )
        if status == cp_model.INFEASIBLE:
            return TspSolution(
                node_order=None,
                cost=float("inf"),
                lower_bound=float("inf"),
                is_infeasible=True,
            )
        return TspSolution(
            node_order=None,
            cost=float("inf"),
            lower_bound=solver.best_objective_bound,
        )
```

> [!TIP]
>
> CP-SAT сам використовує Protobuf для входів, виходів і конфігурації. Чітко
> визначені моделі даних допомагають уникати проблем типу «garbage in, garbage out»
> і полегшують інтеграцію з іншими системами. Це також спрощує тестування й
> налагодження, адже можна просто серіалізувати конкретний сценарій. Для
> конфігурації наявність значень за замовчуванням дуже корисна, бо дозволяє
> розширювати конфігурацію без порушення зворотної сумісності. Це суттєва
> перевага, оскільки вимоги зазвичай не відомі заздалегідь. Pydantic чудово
> справляється з цим завданням і може використовуватися також у веб-API.
> Protobuf, хоча й не специфічний для Python і тому універсальніший, складніший
> у використанні й не має такої гнучкості, як Pydantic.

## Моделі запитів і відповідей

У цьому розділі ми визначимо моделі запитів і відповідей для API. Ці моделі
полегшують комунікацію між клієнтом і сервером, гарантуючи, що дані
обмінюються у структурованому та валідованому вигляді.

Моделі визначено у файлі `models.py`, і вони містять потрібні структури даних
для надсилання запиту на задачу TSP та відстеження статусу задачі.

```python
# ./app/models.py
"""
This file contains the implementation of additional data models for the optimization API.
"""

from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field
from uuid import UUID, uuid4
from solver import OptimizationParameters, TspInstance
```

Модель `TspJobRequest` інкапсулює інформацію, потрібну для надсилання задачі TSP
до API. Вона містить екземпляр TSP, параметри оптимізації та необов’язковий
webhook URL для сповіщень після завершення задачі.

```python
class TspJobRequest(BaseModel):
    """
    A request model for a TSP job.
    """

    tsp_instance: TspInstance = Field(..., description="The TSP instance to solve.")
    optimization_parameters: OptimizationParameters = Field(
        default_factory=OptimizationParameters,
        description="The optimization parameters.",
    )
    webhook_url: HttpUrl | None = Field(
        default=None, description="The URL to call once the computation is complete."
    )
```

Приклад запиту може виглядати так:

```json
{
    "tsp_instance": {
        "num_nodes": 4,
        "edges": [
            {"source": 0, "target": 1, "cost": 1},
            {"source": 1, "target": 2, "cost": 2},
            {"source": 2, "target": 3, "cost": 3},
            {"source": 3, "target": 0, "cost": 4},
        ],
    },
    "optimization_parameters": {"timeout": 5},
    "webhook_url": null,
},
```

Модель `TspJobStatus` використовується для відстеження статусу задачі TSP. Вона
містить поля для моніторингу різних етапів життєвого циклу задачі — від
надсилання до завершення.

```python
class TspJobStatus(BaseModel):
    """
    A response model for the status of a TSP job.
    """

    task_id: UUID = Field(default_factory=uuid4, description="The ID of the task.")
    status: str = Field(default="Submitted", description="The status of the task.")
    submitted_at: datetime = Field(
        default_factory=datetime.now, description="The time the task was submitted."
    )
    started_at: datetime | None = Field(
        default=None, description="The time the task was started."
    )
    completed_at: datetime | None = Field(
        default=None, description="The time the task was completed."
    )
    error: str | None = Field(
        default=None, description="The error message if the task failed."
    )
```

Ці моделі гарантують, що дані між клієнтом і сервером є чітко визначеними та
валідованими.

## База даних

У цьому розділі ми реалізуємо проксі для бази даних, щоб зберігати задачі та
розв’язки. Для простоти використовуємо Redis, який слугує і базою даних, і
чергою задач. Такий підхід мінімізує потребу в додаткових базах даних і
використовує можливості Redis як key-value сховища та автоматичного
протермінування даних.

Клас `TspJobDbConnection` інкапсулює взаємодію з базою Redis. Він надає методи
для реєстрації нових задач, оновлення статусів, отримання запитів, статусів і
розв’язків, перегляду списку задач та їх видалення.

```python
# ./app/db.py
"""
This file contains a proxy class to interact with the database.
We are using Redis as the database for this example, but the implementation
can be easily adapted to other databases, as the proxy class abstracts the
database operations.
"""

import json
from models import TspJobStatus, TspJobRequest
from solver import TspSolution
from uuid import UUID
import redis
from typing import Optional, List
import logging
```

Клас ініціалізується Redis-клієнтом і часом протермінування для збережених
даних. Метод `_get_data` є допоміжним і отримує та парсить JSON-дані з Redis за
ключем.

```python
class TspJobDbConnection:
    def __init__(self, redis_client: redis.Redis, expire_time: int = 24 * 60 * 60):
        """Initialize the Redis connection and expiration time."""
        self._redis = redis_client
        self._expire_time = expire_time
        logging.basicConfig(level=logging.INFO)

    def _get_data(self, key: str) -> Optional[dict]:
        """Get data from Redis by key and parse JSON."""
        try:
            data = self._redis.get(key)
            if data is not None:
                return json.loads(data)
        except redis.RedisError as e:
            logging.error(f"Redis error: {e}")
        return None
```

Методи `get_request`, `get_status` і `get_solution` відповідно отримують запит,
статус і розв’язок задачі TSP за її ідентифікатором.

```python
def get_request(self, task_id: UUID) -> Optional[TspJobRequest]:
    """Retrieve a TSP job request by task ID."""
    data = self._get_data(f"request:{task_id}")
    return TspJobRequest(**data) if data else None


def get_status(self, task_id: UUID) -> Optional[TspJobStatus]:
    """Retrieve a TSP job status by task ID."""
    data = self._get_data(f"status:{task_id}")
    return TspJobStatus(**data) if data else None


def get_solution(self, task_id: UUID) -> Optional[TspSolution]:
    """Retrieve a TSP solution by task ID."""
    data = self._get_data(f"solution:{task_id}")
    return TspSolution(**data) if data else None
```

Метод `set_solution` зберігає розв’язок TSP у Redis з часом протермінування.
Метод `register_job` реєструє новий запит TSP і статус у Redis.

```python
def set_solution(self, task_id: UUID, solution: TspSolution) -> None:
    """Set a TSP solution in Redis with an expiration time."""
    try:
        self._redis.set(
            f"solution:{task_id}", solution.model_dump_json(), ex=self._expire_time
        )
    except redis.RedisError as e:
        logging.error("Redis error: %s", e)


def register_job(self, request: TspJobRequest) -> TspJobStatus:
    """Register a new TSP job request and status in Redis."""
    job_status = TspJobStatus()
    try:
        pipeline = self._redis.pipeline()
        pipeline.set(
            f"status:{job_status.task_id}",
            job_status.model_dump_json(),
            ex=self._expire_time,
        )
        pipeline.set(
            f"request:{job_status.task_id}",
            request.model_dump_json(),
            ex=self._expire_time,
        )
        pipeline.execute()
    except redis.RedisError as e:
        logging.error("Redis error: %s", e)

    return job_status
```

Метод `update_job_status` оновлює статус наявної задачі TSP. Метод `list_jobs`
повертає список статусів усіх задач.

```python
def update_job_status(self, job_status: TspJobStatus) -> None:
    """Update the status of an existing TSP job."""
    try:
        self._redis.set(
            f"status:{job_status.task_id}",
            job_status.model_dump_json(),
            ex=self._expire_time,
        )
    except redis.RedisError as e:
        logging.error("Redis error: %s", e)


def list_jobs(self) -> List[TspJobStatus]:
    """List all TSP job statuses."""
    try:
        status_keys = self._redis.keys("status:*")
        data = self._redis.mget(status_keys)
        return [TspJobStatus(**json.loads(status)) for status in data if status]
    except redis.RedisError as e:
        logging.error("Redis error: %s", e)

        return []
```

Метод `delete_job` видаляє запит, статус і розв’язок задачі TSP з Redis.

```python
def delete_job(self, task_id: UUID) -> None:
    """Delete a TSP job request, status, and solution from Redis."""
    try:
        pipeline = self._redis.pipeline()
        pipeline.delete(f"status:{task_id}")
        pipeline.delete(f"request:{task_id}")
        pipeline.delete(f"solution:{task_id}")
        pipeline.execute()
    except redis.RedisError as e:
        logging.error("Redis error: %s", e)
```

## Конфігурація

Базі даних і черзі задач потрібне підключення перед використанням. Ми надаємо
функції `get_db_connection` і `get_task_queue` у файлі `config.py` з трьох
основних причин:

- щоб гарантувати коректне налаштування бази даних і черги задач із правильними
  параметрами підключення; якщо зміниться хост Redis, достатньо оновити його в
  одному місці;
- щоб інтегрувати ці функції в систему ін’єкції залежностей FastAPI, забезпечивши
  доступність бази й черги в ендпойнтах без встановлення підключення в кожному
  з них; це також полегшує тестування з іншою базою даних;
- щоб і застосунок FastAPI, і воркери використовували однакові конфігураційні
  функції, попри різні точки входу.

```python
# ./app/config.py
"""
This file contains the configuration for the optimization API.
For this simple project, it only sets up the database connection and the task queue.
The other parts of the API should not be aware of the specific connection details.
"""

from db import TspJobDbConnection
import redis
from rq import Queue


def get_db_connection() -> TspJobDbConnection:
    """Provides a TspJobDbConnection instance."""
    redis_client = redis.Redis(
        host="optimization_api_redis", port=6379, decode_responses=True, db=0
    )
    return TspJobDbConnection(redis_client=redis_client)


def get_task_queue() -> Queue:
    """Provides a Redis Queue instance."""
    redis_client = redis.Redis(host="optimization_api_redis", port=6379, db=1)
    return Queue(connection=redis_client)
```

## Задачі

Маючи базу даних, ми можемо створити задачі, що запускатимуть оптимізацію.
Оптимізація працюватиме в окремому процесі та використовуватиме базу даних для
комунікації з вебсервером. Для простоти ми передаватимемо в задачу лише
посилання на роботу. Задача отримає потрібні дані з бази та оновить базу
результатами. Додатково, завдяки блоку `if __name__ == "__main__":`, ці задачі
можна запускати через зовнішню чергу як системні команди.

Файл `tasks.py` містить функції й логіку для запуску оптимізації в окремому
процесі воркера.

```python
# ./app/tasks.py
"""
This file is responsible for running the optimization job in a separate worker.
"""

from config import get_db_connection
from models import TspJobRequest, TspJobStatus
from solver import TspSolver
from datetime import datetime
from uuid import UUID
from db import TspJobDbConnection
import httpx
import logging
```

Функція `send_webhook` надсилає POST-запит на вказаний webhook URL зі статусом
задачі. Це дозволяє асинхронні сповіщення після завершення обчислень.

```python
def send_webhook(job_request: TspJobRequest, job_status: TspJobStatus) -> None:
    if job_request.webhook_url:
        try:
            # Send a POST request to the webhook URL
            response = httpx.post(
                url=f"{job_request.webhook_url}", json=job_status.model_dump_json()
            )
            response.raise_for_status()  # Raise an error for bad responses
        except httpx.HTTPStatusError as e:
            logging.error(
                f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            logging.error(f"An error occurred: {e}")
```

Функція `run_optimization_job` отримує запит із бази даних, запускає
оптимізаційний алгоритм і зберігає розв’язок назад у базі. Вона також оновлює
статус задачі та надсилає webhook-сповіщення після завершення.

```python
def run_optimization_job(
    job_id: UUID, db_connection: TspJobDbConnection | None = None
) -> None:
    """
    Will fetch the job request from the database, run the optimization algorithm,
    and store the solution back in the database. Finally, it will send a webhook
    to the URL specified in the job. This function may be run on a separate worker,
    which is why we do not pass or return data directly, but rather use the database.
    """
    if db_connection is None:
        db_connection = get_db_connection()
    job_status = db_connection.get_status(job_id)
    job_request = db_connection.get_request(job_id)
    if job_status is None or job_request is None:
        return  # job got deleted
    job_status.status = "Running"
    job_status.started_at = datetime.now()
    db_connection.update_job_status(job_status)
    solver = TspSolver(job_request.tsp_instance, job_request.optimization_parameters)
    solution = solver.solve(log_callback=print)
    db_connection.set_solution(job_id, solution)
    job_status.status = "Completed"
    job_status.completed_at = datetime.now()
    db_connection.update_job_status(job_status)
    send_webhook(job_request, job_status)
```

## API

У цьому фінальному розділі ми зберемо реальний API на FastAPI. Цей API
надаватиме ендпойнти для надсилання запитів на TSP, перевірки статусів,
отримання розв’язків, скасування задач і перегляду списку всіх задач. FastAPI —
ефективний і простий у використанні фреймворк для побудови веб-API на Python.

Файл `main.py` містить налаштування застосунку FastAPI та маршрути API. Для
простоти всі маршрути зібрані в одному файлі, але в більших проєктах їх варто
розділяти на модулі.

```python
# ./app/main.py
"""
This file contains the main FastAPI application.
For a larger project, we would move the routes to separate files, but for this example, we keep everything in one file.
"""

from uuid import UUID
from fastapi import FastAPI, APIRouter, HTTPException, Depends

from models import TspJobRequest, TspJobStatus
from solver import TspSolution
from config import get_db_connection, get_task_queue
from tasks import run_optimization_job
```

Застосунок FastAPI ініціалізується з назвою та описом. API-роутер створюється
для групування маршрутів, пов’язаних із TSP-розв’язувачем.

```python
app = FastAPI(
    title="My Optimization API",
    description="This is an example on how to deploy an optimization algorithm based on CP-SAT as an API.",
)

tsp_solver_v1_router = APIRouter(tags=["TSP_solver_v1"])
```

Ендпойнт `post_job` дозволяє користувачам надсилати нову задачу TSP. Задача
реєструється в базі даних, а оптимізаційна робота ставиться в чергу для
асинхронної обробки.

```python
@tsp_solver_v1_router.post("/jobs", response_model=TspJobStatus)
def post_job(
    job_request: TspJobRequest,
    db_connection=Depends(get_db_connection),
    task_queue=Depends(get_task_queue),
):
    """
    Submit a new job to solve a TSP instance.
    """
    job_status = db_connection.register_job(job_request)
    # enqueue the optimization job in the task queue.
    # Will return immediately, the job will be run in a separate worker.
    task_queue.enqueue(
        run_optimization_job,
        job_status.task_id,
        # adding a 60 second buffer to the job timeout
        job_timeout=job_request.optimization_parameters.timeout + 60,
    )
    return job_status
```

Ендпойнт `get_job` повертає статус конкретної задачі за її ID.

```python
@tsp_solver_v1_router.get("/jobs/{task_id}", response_model=TspJobStatus)
def get_job(task_id: UUID, db_connection=Depends(get_db_connection)):
    """
    Return the status of a job.
    """
    status = db_connection.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status
```

Ендпойнт `get_solution` повертає розв’язок конкретної задачі, якщо він доступний.

```python
@tsp_solver_v1_router.get("/jobs/{task_id}/solution", response_model=TspSolution)
def get_solution(task_id: UUID, db_connection=Depends(get_db_connection)):
    """
    Return the solution of a job, if available.
    """
    solution = db_connection.get_solution(task_id)
    if solution is None:
        raise HTTPException(status_code=404, detail="Solution not found")
    return solution
```

Ендпойнт `cancel_job` видаляє або скасовує задачу. Якщо вона вже виконується,
це не зупиняє її миттєво.

```python
@tsp_solver_v1_router.delete("/jobs/{task_id}")
def cancel_job(task_id: UUID, db_connection=Depends(get_db_connection)):
    """
    Deletes/cancels a job. This will *not* immediately stop the job if it is running.
    """
    db_connection.delete_job(task_id)
```

Ендпойнт `list_jobs` повертає список усіх задач та їхніх статусів.

```python
@tsp_solver_v1_router.get("/jobs", response_model=list[TspJobStatus])
def list_jobs(db_connection=Depends(get_db_connection)):
    """
    List all jobs.
    """
    return db_connection.list_jobs()
```

Насамкінець ми додаємо API-роутер до застосунку FastAPI під префіксом
`/tsp_solver/v1`.

```python
app.include_router(tsp_solver_v1_router, prefix="/tsp_solver/v1")
```

### Запуск застосунку

Після виконання `docker-compose up -d --build` ви можете відкрити API за адресою
`http://localhost:80/docs`. Це відкриє Swagger UI, де можна протестувати API.
Ви можете надіслати задачу, перевірити статус і отримати розв’язок. Також можна
скасувати задачу або переглянути список усіх задач.

| ![Swagger UI](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/swagger_optimization_api.png) |
| :-------------------------------------------------------------------------------------------------------------: |
|               FastAPI має вбудований Swagger UI, який дозволяє взаємодіяти з API.               |

| ![Swagger UI - Job Submission](https://raw.githubusercontent.com/d-krupke/cpsat-primer/main/images/swagger_try_it_out.png) |
| :------------------------------------------------------------------------------------------------------------------------: |
|                           Натиснувши "try it out", ви можете напряму надіслати задачу до API.                           |

<details>
<summary>
Ось інстанс для спроби: (натисніть, щоб розгорнути)
</summary>

```json
{
  "optimization_parameters": {
    "timeout": 5
  },
  "tsp_instance": {
    "num_nodes": 15,
    "edges": [
      { "source": 0, "target": 1, "cost": 82 },
      { "source": 1, "target": 0, "cost": 82 },
      { "source": 0, "target": 2, "cost": 35 },
      { "source": 2, "target": 0, "cost": 35 },
      { "source": 0, "target": 3, "cost": 58 },
      { "source": 3, "target": 0, "cost": 58 },
      { "source": 0, "target": 4, "cost": 9 },
      { "source": 4, "target": 0, "cost": 9 },
      { "source": 0, "target": 5, "cost": 13 },
      { "source": 5, "target": 0, "cost": 13 },
      { "source": 0, "target": 6, "cost": 91 },
      { "source": 6, "target": 0, "cost": 91 },
      { "source": 0, "target": 7, "cost": 72 },
      { "source": 7, "target": 0, "cost": 72 },
      { "source": 0, "target": 8, "cost": 16 },
      { "source": 8, "target": 0, "cost": 16 },
      { "source": 0, "target": 9, "cost": 50 },
      { "source": 9, "target": 0, "cost": 50 },
      { "source": 0, "target": 10, "cost": 80 },
      { "source": 10, "target": 0, "cost": 80 },
      { "source": 0, "target": 11, "cost": 92 },
      { "source": 11, "target": 0, "cost": 92 },
      { "source": 0, "target": 12, "cost": 28 },
      { "source": 12, "target": 0, "cost": 28 },
      { "source": 0, "target": 13, "cost": 17 },
      { "source": 13, "target": 0, "cost": 17 },
      { "source": 0, "target": 14, "cost": 97 },
      { "source": 14, "target": 0, "cost": 97 },
      { "source": 1, "target": 2, "cost": 14 },
      { "source": 2, "target": 1, "cost": 14 },
      { "source": 1, "target": 3, "cost": 32 },
      { "source": 3, "target": 1, "cost": 32 },
      { "source": 1, "target": 4, "cost": 41 },
      { "source": 4, "target": 1, "cost": 41 },
      { "source": 1, "target": 5, "cost": 52 },
      { "source": 5, "target": 1, "cost": 52 },
      { "source": 1, "target": 6, "cost": 58 },
      { "source": 6, "target": 1, "cost": 58 },
      { "source": 1, "target": 7, "cost": 20 },
      { "source": 7, "target": 1, "cost": 20 },
      { "source": 1, "target": 8, "cost": 1 },
      { "source": 8, "target": 1, "cost": 1 },
      { "source": 1, "target": 9, "cost": 54 },
      { "source": 9, "target": 1, "cost": 54 },
      { "source": 1, "target": 10, "cost": 75 },
      { "source": 10, "target": 1, "cost": 75 },
      { "source": 1, "target": 11, "cost": 15 },
      { "source": 11, "target": 1, "cost": 15 },
      { "source": 1, "target": 12, "cost": 45 },
      { "source": 12, "target": 1, "cost": 45 },
      { "source": 1, "target": 13, "cost": 94 },
      { "source": 13, "target": 1, "cost": 94 },
      { "source": 1, "target": 14, "cost": 41 },
      { "source": 14, "target": 1, "cost": 41 },
      { "source": 2, "target": 3, "cost": 82 },
      { "source": 3, "target": 2, "cost": 82 },
      { "source": 2, "target": 4, "cost": 44 },
      { "source": 4, "target": 2, "cost": 44 },
      { "source": 2, "target": 5, "cost": 83 },
      { "source": 5, "target": 2, "cost": 83 },
      { "source": 2, "target": 6, "cost": 91 },
      { "source": 6, "target": 2, "cost": 91 },
      { "source": 2, "target": 7, "cost": 78 },
      { "source": 7, "target": 2, "cost": 78 },
      { "source": 2, "target": 8, "cost": 51 },
      { "source": 8, "target": 2, "cost": 51 },
      { "source": 2, "target": 9, "cost": 6 },
      { "source": 9, "target": 2, "cost": 6 },
      { "source": 2, "target": 10, "cost": 81 },
      { "source": 10, "target": 2, "cost": 81 },
      { "source": 2, "target": 11, "cost": 77 },
      { "source": 11, "target": 2, "cost": 77 },
      { "source": 2, "target": 12, "cost": 93 },
      { "source": 12, "target": 2, "cost": 93 },
      { "source": 2, "target": 13, "cost": 97 },
      { "source": 13, "target": 2, "cost": 97 },
      { "source": 2, "target": 14, "cost": 33 },
      { "source": 14, "target": 2, "cost": 33 },
      { "source": 3, "target": 4, "cost": 66 },
      { "source": 4, "target": 3, "cost": 66 },
      { "source": 3, "target": 5, "cost": 47 },
      { "source": 5, "target": 3, "cost": 47 },
      { "source": 3, "target": 6, "cost": 54 },
      { "source": 6, "target": 3, "cost": 54 },
      { "source": 3, "target": 7, "cost": 39 },
      { "source": 7, "target": 3, "cost": 39 },
      { "source": 3, "target": 8, "cost": 98 },
      { "source": 8, "target": 3, "cost": 98 },
      { "source": 3, "target": 9, "cost": 90 },
      { "source": 9, "target": 3, "cost": 90 },
      { "source": 3, "target": 10, "cost": 5 },
      { "source": 10, "target": 3, "cost": 5 },
      { "source": 3, "target": 11, "cost": 27 },
      { "source": 11, "target": 3, "cost": 27 },
      { "source": 3, "target": 12, "cost": 61 },
      { "source": 12, "target": 3, "cost": 61 },
      { "source": 3, "target": 13, "cost": 95 },
      { "source": 13, "target": 3, "cost": 95 },
      { "source": 3, "target": 14, "cost": 19 },
      { "source": 14, "target": 3, "cost": 19 },
      { "source": 4, "target": 5, "cost": 34 },
      { "source": 5, "target": 4, "cost": 34 },
      { "source": 4, "target": 6, "cost": 10 },
      { "source": 6, "target": 4, "cost": 10 },
      { "source": 4, "target": 7, "cost": 20 },
      { "source": 7, "target": 4, "cost": 20 },
      { "source": 4, "target": 8, "cost": 44 },
      { "source": 8, "target": 4, "cost": 44 },
      { "source": 4, "target": 9, "cost": 33 },
      { "source": 9, "target": 4, "cost": 33 },
      { "source": 4, "target": 10, "cost": 29 },
      { "source": 10, "target": 4, "cost": 29 },
      { "source": 4, "target": 11, "cost": 36 },
      { "source": 11, "target": 4, "cost": 36 },
      { "source": 4, "target": 12, "cost": 62 },
      { "source": 12, "target": 4, "cost": 62 },
      { "source": 4, "target": 13, "cost": 77 },
      { "source": 13, "target": 4, "cost": 77 },
      { "source": 4, "target": 14, "cost": 63 },
      { "source": 14, "target": 4, "cost": 63 },
      { "source": 5, "target": 6, "cost": 73 },
      { "source": 6, "target": 5, "cost": 73 },
      { "source": 5, "target": 7, "cost": 6 },
      { "source": 7, "target": 5, "cost": 6 },
      { "source": 5, "target": 8, "cost": 91 },
      { "source": 8, "target": 5, "cost": 91 },
      { "source": 5, "target": 9, "cost": 5 },
      { "source": 9, "target": 5, "cost": 5 },
      { "source": 5, "target": 10, "cost": 61 },
      { "source": 10, "target": 5, "cost": 61 },
      { "source": 5, "target": 11, "cost": 11 },
      { "source": 11, "target": 5, "cost": 11 },
      { "source": 5, "target": 12, "cost": 91 },
      { "source": 12, "target": 5, "cost": 91 },
      { "source": 5, "target": 13, "cost": 7 },
      { "source": 13, "target": 5, "cost": 7 },
      { "source": 5, "target": 14, "cost": 88 },
      { "source": 14, "target": 5, "cost": 88 },
      { "source": 6, "target": 7, "cost": 52 },
      { "source": 7, "target": 6, "cost": 52 },
      { "source": 6, "target": 8, "cost": 86 },
      { "source": 8, "target": 6, "cost": 86 },
      { "source": 6, "target": 9, "cost": 48 },
      { "source": 9, "target": 6, "cost": 48 },
      { "source": 6, "target": 10, "cost": 13 },
      { "source": 10, "target": 6, "cost": 13 },
      { "source": 6, "target": 11, "cost": 31 },
      { "source": 11, "target": 6, "cost": 31 },
      { "source": 6, "target": 12, "cost": 91 },
      { "source": 12, "target": 6, "cost": 91 },
      { "source": 6, "target": 13, "cost": 62 },
      { "source": 13, "target": 6, "cost": 62 },
      { "source": 6, "target": 14, "cost": 30 },
      { "source": 14, "target": 6, "cost": 30 },
      { "source": 7, "target": 8, "cost": 79 },
      { "source": 8, "target": 7, "cost": 79 },
      { "source": 7, "target": 9, "cost": 94 },
      { "source": 9, "target": 7, "cost": 94 },
      { "source": 7, "target": 10, "cost": 58 },
      { "source": 10, "target": 7, "cost": 58 },
      { "source": 7, "target": 11, "cost": 12 },
      { "source": 11, "target": 7, "cost": 12 },
      { "source": 7, "target": 12, "cost": 81 },
      { "source": 12, "target": 7, "cost": 81 },
      { "source": 7, "target": 13, "cost": 2 },
      { "source": 13, "target": 7, "cost": 2 },
      { "source": 7, "target": 14, "cost": 89 },
      { "source": 14, "target": 7, "cost": 89 },
      { "source": 8, "target": 9, "cost": 15 },
      { "source": 9, "target": 8, "cost": 15 },
      { "source": 8, "target": 10, "cost": 94 },
      { "source": 10, "target": 8, "cost": 94 },
      { "source": 8, "target": 11, "cost": 23 },
      { "source": 11, "target": 8, "cost": 23 },
      { "source": 8, "target": 12, "cost": 50 },
      { "source": 12, "target": 8, "cost": 50 },
      { "source": 8, "target": 13, "cost": 79 },
      { "source": 13, "target": 8, "cost": 79 },
      { "source": 8, "target": 14, "cost": 65 },
      { "source": 14, "target": 8, "cost": 65 },
      { "source": 9, "target": 10, "cost": 68 },
      { "source": 10, "target": 9, "cost": 68 },
      { "source": 9, "target": 11, "cost": 81 },
      { "source": 11, "target": 9, "cost": 81 },
      { "source": 9, "target": 12, "cost": 34 },
      { "source": 12, "target": 9, "cost": 34 },
      { "source": 9, "target": 13, "cost": 21 },
      { "source": 13, "target": 9, "cost": 21 },
      { "source": 9, "target": 14, "cost": 16 },
      { "source": 14, "target": 9, "cost": 16 },
      { "source": 10, "target": 11, "cost": 10 },
      { "source": 11, "target": 10, "cost": 10 },
      { "source": 10, "target": 12, "cost": 12 },
      { "source": 12, "target": 10, "cost": 12 },
      { "source": 10, "target": 13, "cost": 60 },
      { "source": 13, "target": 10, "cost": 60 },
      { "source": 10, "target": 14, "cost": 61 },
      { "source": 14, "target": 10, "cost": 61 },
      { "source": 11, "target": 12, "cost": 36 },
      { "source": 12, "target": 11, "cost": 36 },
      { "source": 11, "target": 13, "cost": 78 },
      { "source": 13, "target": 11, "cost": 78 },
      { "source": 11, "target": 14, "cost": 79 },
      { "source": 14, "target": 11, "cost": 79 },
      { "source": 12, "target": 13, "cost": 54 },
      { "source": 13, "target": 12, "cost": 54 },
      { "source": 12, "target": 14, "cost": 33 },
      { "source": 14, "target": 12, "cost": 33 },
      { "source": 13, "target": 14, "cost": 29 },
      { "source": 14, "target": 13, "cost": 29 }
    ]
  }
}
```

</details>
