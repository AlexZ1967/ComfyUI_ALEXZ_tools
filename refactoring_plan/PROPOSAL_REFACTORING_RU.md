# ComfyUI_ALEXZ_tools — Руководство по рефакторингу

**Статус**: Предложенная инициатива рефакторинга  
**Область**: Архитектура, организация кода, улучшение качества  
**Приоритет**: Высокий (решает проблемы интеграции Vue, поддерживаемость, тестируемость)

---

## 1. Резюме

Проект ComfyUI_ALEXZ_tools созрел с 12 стабильными нодами и сложной системой обнаружения модулей. Однако кодовая база демонстрирует архитектурные паттерны, которые усложняют отладку:

- **2410-строчный файл JavaScript UI** с разбросанным глобальным состоянием и конкурирующими механизмами синхронизации
- **2231-строчный Python backend** с монолитными определениями функций и блокирующими операциями  
- **Критическая ошибка интеграции Vue.js**, возникающая из-за несоответствия ожиданий фреймворка и DOM
- **Минимальное покрытие тестами** (только 4 smoke теста)
- **Код разбросан по модулям** без четкого разделения ответственности

**Общая оценка ДО рефакторинга**: 6.5/10 (штраф за поддерживаемость)  
**Ожидаемая оценка ПОСЛЕ рефакторинга**: 8.5/10 (правильная архитектура + тестовое покрытие)

---

## 2. Анализ корневой причины: ошибка интеграции Vue.js

### Формулировка проблемы
Виджет Module Nodes становится невидимым при втором переходе (Module Nodes → NodeMap → Module Nodes → NodeMap).

### Корневая причина
```
1. renderPicker(container) добавляет root в container
2. Vue.js обнаруживает изменение DOM
3. Vue переренисовывает и перестраивает container
4. Root элемент перемещается во внутреннее wrapper DIV Vue
5. Проверки CSS display не срабатывают, т.к. родитель изменился
6. Логика восстановления не может перехватить смещение из-за race condition
```

### Почему сложно исправить сейчас
- **Смешанные ответственности**: манипуляция DOM, синхронизация состояния, видимость CSS в одном файле
- **Разбросанное состояние**: 8+ window-свойств не координируются правильно
- **Конкурирующие механизмы**: 3 разных подхода к синхронизации (Tab Relay, Container Ownership, recovery intervals)
- **Границы фреймворка неясны**: нет абстракционного слоя между Vue и raw DOM

---

## 3. Архитектурные улучшения

### 3.1 Модуляризация Frontend (JavaScript)

**Текущая структура** (монолитная):
```
module_node_picker.js (2410 строк)
├── Утилиты DOM
├── Fetchers API
├── Управление состоянием
├── Синхронизация Tab relay
├── Синхронизация Container ownership
├── Диагностика
└── Логика рендера
```

**Предложенная структура** (модульная):
```
web/
├── module_node_picker.js (300 строк)
│   └── Точка входа, регистрация расширения
├── picker-ui/
│   ├── renderer.js (400 строк)
│   │   └── renderPicker(), DOM construction
│   ├── styles.js (150 строк)
│   │   └── Инжекция CSS, обработка темы
│   └── event-handlers.js (200 строк)
│       └── Обработчики Click, expand, actions
├── state/
│   ├── state-machine.js (250 строк)
│   │   └── Состояние tab, выбранный модуль, видимость
│   ├── store.js (150 строк)
│   │   └── Централизованное состояние с уведомлениями
│   └── persistence.js (100 строк)
│       └── Операции localStorage
├── api/
│   ├── catalog-api.js (80 строк)
│   │   └── fetchNodeCatalog, fetchModuleInfo
│   ├── actions-api.js (100 строк)
│   │   └── startModuleUpdate, startRefresh
│   └── error-handler.js (80 строк)
│       └── Стандартная обработка ошибок
├── sync/
│   ├── vue-integration.js (150 строк)
│   │   └── Жизненный цикл Vue, обнаружение переренисовки
│   └── minimal-relay.js (120 строк)
│       └── Синхронизация переключения tab
├── diagnostics/
│   ├── logger.js (100 строк)
│   │   └── Условная система логирования
│   └── debug-panel.js (80 строк)
│       └── UI диагностики только для разработки
└── test-utils/
    └── mocks.js (100 строк)
        └── Mock API, DOM stubs
```

**Преимущества**:
- Каждый модуль имеет одну ответственность
- Легче тестировать в изоляции
- Более четкий поток зависимостей
- Интеграция Vue изолирована в `vue-integration.js`

### 3.2 Исправление интеграции Vue

**Новый подход**: осведомленность о жизненном цикле Vue

```javascript
// vue-integration.js
class VueLifecycleManager {
  constructor(container) {
    this.container = container;
    this.savedContainer = container.cloneNode(false);
    this.isReconstructing = false;
    this.observer = new MutationObserver(this.onVueReconstruct.bind(this));
  }

  onVueReconstruct(mutations) {
    // Обнаружить перестройку container Vue
    if (this.container.textContent.includes('vue-')) {
      this.isReconstructing = true;
      // Временно приостановить операции
    }
  }

  safeAppend(element) {
    this.savedContainer = this.container;
    this.container.appendChild(element);
    
    // Установить контрольную точку восстановления
    this.checkpointRoot = element;
    requestAnimationFrame(() => this.verifyPlacement());
  }

  verifyPlacement() {
    // Проверка после переренисовки Vue
    if (!this.checkpointRoot.parentElement === this.savedContainer) {
      this.recover();
    }
  }

  recover() {
    if (this.checkpointRoot && this.savedContainer) {
      this.savedContainer.appendChild(this.checkpointRoot);
    }
  }
}
```

**Шаги реализации**:
1. Обернуть container в VueLifecycleManager
2. Использовать `safeAppend()` вместо прямого appendChild
3. Мониторить мутации Vue на container
4. Auto-восстановление на следующем animation frame

### 3.3 Централизованное управление состоянием

**Текущее** (разбросанное по `window[...]`):
```javascript
window.__alexz_module_picker_sidebar_sync__ = {...}
window.__alexz_module_nodes_container_sync_state__ = {...}
window.__alexz_module_picker_debug__ = {...}
// + 5 еще window-свойств
```

**Предложенное State Store**:
```javascript
// state/store.js
class ModuleNodeStore {
  constructor() {
    this.state = {
      selectedGroup: 'core',
      selectedModule: 'ComfyUI_ALEXZ_tools',
      catalog: null,
      moduleInfo: {},
      refreshStatus: {},
      updateStatus: {},
      visibility: 'hidden',
      debug: false,
    };
    this.listeners = new Map();
  }

  subscribe(key, listener) {
    if (!this.listeners.has(key)) {
      this.listeners.set(key, []);
    }
    this.listeners.get(key).push(listener);
  }

  setState(partial) {
    Object.assign(this.state, partial);
    this.notifyListeners(Object.keys(partial));
  }

  notifyListeners(changedKeys) {
    changedKeys.forEach(key => {
      this.listeners.get(key)?.forEach(fn => fn(this.state[key]));
    });
  }
}

export const store = new ModuleNodeStore();
```

**Использование**:
```javascript
import { store } from './state/store.js';

// Подписка на изменения
store.subscribe('selectedModule', (module) => {
  renderModuleNodes(module);
});

// Обновление состояния
store.setState({ selectedModule: 'ComfyUI_XXXYZ' });
```

### 3.4 Извлечение системы диагностики

**Текущее**: код отладки перемешан с production кодом  
**Предложенное**: отдельный модуль условного логирования

```javascript
// diagnostics/logger.js
export class DiagnosticsLogger {
  constructor() {
    this.enabled = localStorage.getItem('alexz_diagnostics_enabled') === 'true';
    this.logs = [];
  }

  enable() {
    this.enabled = true;
    localStorage.setItem('alexz_diagnostics_enabled', 'true');
  }

  disable() {
    this.enabled = false;
    localStorage.removeItem('alexz_diagnostics_enabled');
  }

  log(category, message, data = {}) {
    if (!this.enabled) return;
    
    const entry = {
      timestamp: new Date().toISOString(),
      category,
      message,
      data,
    };
    
    this.logs.push(entry);
    console.debug(`[${category}]`, message, data);
    
    // Хранить последние 100 записей
    if (this.logs.length > 100) {
      this.logs.shift();
    }
  }

  render() {
    if (!this.enabled) return null;
    
    // Вернуть элемент UI диагностики
  }
}

export const diags = new DiagnosticsLogger();
```

**Использование**:
```javascript
import { diags } from './diagnostics/logger.js';

// Условное логирование
diags.log('sync', 'Tab switched', { from: 'Module Nodes' });

// Нет влияния на производительность при отключении
```

---

## 4. Рефакторинг Backend (Python)

### 4.1 Разделение монолитного файла API

**Текущее**: один файл из 2231 строки, смешивающий ответственности  
**Предложенное**: многоуровневая архитектура

```
utils/
├── module_node_browser_api.py (200 строк)
│   └── Обработчики маршрутов, запуск расширения
├── catalog/
│   ├── __init__.py
│   ├── node_collector.py (200 строк)
│   │   └── _collect_nodes(), _build_node_snapshots()
│   ├── module_classifier.py (200 строк)
│   │   └── _classify_by_relative_module()
│   └── catalog_builder.py (150 строк)
│       └── _build_group_catalog(), _build_group_modules()
├── git/
│   ├── __init__.py
│   ├── git_state.py (150 строк)
│   │   └── _module_git_state(), _comfyui_git_status()
│   ├── git_sync.py (150 строк)
│   │   └── _sync_module_upstream(), _pull_custom_module()
│   └── git_utils.py (100 строк)
│       └── Обертки git команд
├── module_info/
│   ├── __init__.py
│   ├── info_builder.py (200 строк)
│   │   └── _resolve_module_info() [рефакторено]
│   ├── change_tracking.py (150 строк)
│   │   └── _apply_node_change_info(), обнаружение изменений
│   └── cache.py (100 строк)
│       └── Слой кеширования с TTL
└── job_queue/
    ├── __init__.py
    ├── job_queue.py (200 строк)
    │   └── Асинхронная очередь задач для refresh/update
    ├── refresh_job.py (150 строк)
    │   └── _refresh_comfyui(), _refresh_modules()
    └── update_job.py (150 строк)
        └── _update_comfyui(), _install_requirements()
```

**Размеры файлов после рефакторинга**: максимум 250 строк на файл

### 4.2 Конвертация Threading в Async/Await

**Текущий паттерн** (блокирующие вызовы, ручное управление потоками):
```python
def _refresh_modules():
    """Блокирующая операция с ручным управлением потоками."""
    with _REFRESH_LOCK:
        # ...долгие блокирующие subprocess вызовы...
        result = subprocess.run(["git", "fetch"], cwd=path, timeout=2)
```

**Предложенный паттерн** (async-first):
```python
# job_queue/refresh_job.py
async def refresh_modules_async(modules):
    """Неблокирующее обновление модулей с параллельными git операциями."""
    tasks = [
        refresh_single_module_async(mod)
        for mod in modules
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def refresh_single_module_async(module_name):
    """Обновление одного модуля с timeout."""
    try:
        return await asyncio.wait_for(
            git.fetch_async(module_name),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        return {"error": "git fetch timeout"}
```

**Преимущества**:
- Неблокирующий UI во время операций
- Параллельные операции (N модулей примерно за время 1)
- Чище код без ручных locks
- Лучше использование ресурсов

### 4.3 Отдельные модели данных

**Текущее**: raw dicts везде  
**Предложенное**: type-safe dataclasses

```python
# catalog/models.py
from dataclasses import dataclass, asdict
from typing import Optional, List

@dataclass
class NodeInfo:
    """Метаданные ноды и классификация."""
    node_name: str
    display_name: str
    module: str
    group: str
    category: str
    annotation: str

@dataclass
class ModuleGitState:
    """Состояние git-репозитория модуля."""
    module_name: str
    installed_commit: str
    installed_commit_short: str
    git_has_upstream: bool
    git_ahead: Optional[int]
    git_behind: Optional[int]
    update_available: Optional[bool]

@dataclass
class ModuleInfo:
    """Полная информация модуля."""
    module: str
    group: str
    title: str
    author: str
    description: str
    repository: str
    owner_url: str
    source: str
    git_state: Optional[ModuleGitState]
    new_nodes: List[str]
    updated_nodes: List[str]
```

**Преимущества**:
- Type safety с IDE автодополнением
- Автоматическое валидирование
- Меньше ошибок в dict-ключах
- Self-documenting код

---

## 5. Разделение ответственности

### 5.1 Извлечение логики ноды

**Текущий паттерн**: 12 нод с дублирующейся структурой  
**Проблема**: каждый файл ноды имеет похожий boilerplate

**Предложенный базовый класс**:
```python
# nodes/base_node.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class ComfyUINode(ABC):
    """Базовый класс для ALEXZ_tools нод с общими паттернами."""
    
    CATEGORY: str = "ALEXZ"
    OUTPUT_TOOLTIPS: List[str] = None
    
    def __init_subclass__(cls, **kwargs):
        """Auto-attach UI metadata."""
        super().__init_subclass__(**kwargs)
        cls.OUTPUT_TOOLTIPS = cls.OUTPUT_TOOLTIPS or []
    
    @abstractmethod
    def execute(self, **inputs) -> tuple:
        """Реализовать логику ноды."""
        pass
    
    def validate_inputs(self, **inputs) -> None:
        """Override для добавления валидации входов."""
        pass
    
    def log_execution(self, **inputs) -> None:
        """Логировать выполнение ноды для отладки."""
        logger.debug(f"{self.__class__.__name__} executed", extra=inputs)
```

**Использование**:
```python
# nodes/video_inpaint.py
class VideoInpaintWatermark(ComfyUINode):
    CATEGORY = "ALEXZ/video"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "mask": ("MASK",),
            }
        }
    
    def execute(self, video_frames, mask):
        self.log_execution(video_frames=video_frames.shape)
        # Бизнес-логика здесь
        return (result,)
```

### 5.2 CLI Tools модуль

**Текущее**: всё работает в контексте ComfyUI  
**Предложенное**: standalone CLI tools для тестирования/отладки

```
tools/
├── cli.py (150 строк)
│   └── CLI интерфейс используя argparse
├── node_tester.py (100 строк)
│   └── Тестировать отдельные ноды вне ComfyUI
└── catalog_checker.py (100 строк)
    └── Проверить регистрацию нод и группировку
```

**Пример CLI**:
```bash
# Тестировать конкретную ноду
python tools/cli.py test-node ImageAlignOverlayToBackground

# Проверить целостность каталога
python tools/cli.py check-catalog

# Проверить git состояние
python tools/cli.py git-status ComfyUI_ALEXZ_tools
```

---

## 6. Стратегия тестирования

### 6.1 Расширение тестового покрытия

**Текущее**: 4 smoke теста, 0 unit тестов  
**Цель**: >70% покрытие

```
tests/
├── unit/
│   ├── test_catalog.py (200 строк)
│   │   ├── test_node_collection()
│   │   ├── test_module_classification()
│   │   └── test_catalog_building()
│   ├── test_git_state.py (150 строк)
│   │   ├── test_parse_git_log()
│   │   ├── test_detect_upstream()
│   │   └── test_ahead_behind_count()
│   ├── test_module_info.py (150 строк)
│   │   ├── test_resolve_module_info()
│   │   └── test_change_tracking()
│   └── test_change_detection.py (100 строк)
│       └── test_node_change_markers()
├── integration/
│   ├── test_api_endpoints.py (200 строк)
│   │   ├── test_node_catalog_route()
│   │   ├── test_module_info_route()
│   │   └── test_refresh_route()
│   └── test_frontend_backend.py (150 строк)
│       └── Тестировать API контракты
├── e2e/
│   ├── test_picker_widget.py (200 строк)
│   │   ├── test_initial_render()
│   │   ├── test_module_switch()
│   │   └── test_node_insert()
│   └── test_update_flow.py (150 строк)
│       └── Тестировать workflow обновления модулей
└── fixtures/
    ├── conftest.py (100 строк)
    ├── mock_comfy.py (100 строк)
    └── sample_nodes.py (80 строк)
```

### 6.2 Frontend тестирование

```
tests/
├── unit/
│   ├── state.test.js (150 строк)
│   │   ├── test('Store setState notifies listeners')
│   │   └── test('Store persists to localStorage')
│   ├── renderer.test.js (200 строк)
│   │   └── test('renderPicker creates proper DOM structure')
│   └── api.test.js (100 строк)
│       └── test('API wrappers handle errors')
├── integration/
│   ├── picker-widget.test.js (200 строк)
│   │   └── test('Full picker workflow with mocked API')
│   └── vue-integration.test.js (150 строк)
│       └── test('Detects and recovers from Vue re-renders')
└── mocks/
    ├── api-mock.js (80 строк)
    └── comfy-mock.js (100 строк)
```

### 6.3 CI/CD интеграция

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.9"
      - run: pip install -e . pytest pytest-cov pytest-asyncio
      - run: pytest tests/unit tests/integration --cov --cov-report=term --cov-report=xml
      - uses: codecov/codecov-action@v3

  js-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm install
      - run: npm test -- --coverage
```

---

## 7. План реализации

### Фаза 1: Основание (Недели 1-2)
1. ✅ Извлечение модуля store состояния (`state/store.js`)
2. ✅ Извлечение логгера диагностики (`diagnostics/logger.js`)
3. ✅ Создание dataclasses каталога (Python)
4. Установка test fixtures и mocks

### Фаза 2: Frontend (Недели 3-4)
1. Разделение `module_node_picker.js` на логические модули
2. Реализация менеджера жизненного цикла Vue
3. Добавление паттерна подписки на состояние к обработчикам событий
4. Unit тесты для слоев состояния и API

### Фаза 3: Backend (Недели 5-6)
1. Создание подмодуля catalog с логикой сбора нод
2. Создание подмодуля git с операциями состояния/синхронизации
3. Создание job_queue/ с конвертацией async/await
4. Integration тесты для API маршрутов

### Фаза 4: Полировка & Тестирование (Недели 7-8)
1. E2E тесты для полных workflows
2. CLI tools для standalone использования
3. Обновление документации
4. Финальный рефакторинг на основе gaps покрытия

---

## 8. Набор критических улучшений

| Область | До | После | Влияние |
|--------|-------|--------|----------|
| **Frontend размер** | 2410 строк (1 файл) | ~300 строк/модуль (9 файлов) | 75% ↓ длина функции |
| **Связанность состояния** | 8+ window-свойств | 1 централизованное store | 100% ↓ coupling |
| **Интеграция Vue** | 3 конкурирующих механизма | 1 менеджер жизненного цикла | 100% ↓ race conditions |
| **Backend размер** | 2231 строк (1 файл) | ~250 строк/модуль (9 файлов) | 88% ↓ макс длина функции |
| **Threading** | Ручные locks, блокировка | async/await | 100% ↓ UI blocking |
| **Тестовое покрытие** | 4 smoke теста | 50+ unit/integration тестов | 600% ↑ покрытие |
| **Модели данных** | Raw dicts | Type-safe dataclasses | 100% ↓ dict key ошибки |

---

## 9. Стратегия миграции

### Подход без breaking changes
- Сохранить API маршруты `module_node_browser_api.py` идентичными
- Сохранить public интерфейс `module_node_picker.js`
- Рефакторить внутренности позади стабильных API
- Постепенная миграция старых путей кода

### Параллельная реализация
1. Строить новые модули рядом с существующим кодом
2. Использовать feature flags для переключения между старым/новым
3. Валидировать новую реализацию с тестами
4. Постепенно снимать старый код

### Пример: миграция State Store
```javascript
// Шаг 1: Представить новый store
import { store } from './state/store.js';

// Шаг 2: Зеркалировать операции
window.__alexz_module_picker_sidebar_sync__ = {
  get groupId() { return store.state.selectedGroup; },
  set groupId(val) { store.setState({ selectedGroup: val }); },
};

// Шаг 3: Обновить код для использования store
store.subscribe('selectedGroup', updateUI);

// Шаг 4: Убрать window proxy после обновления всех callers
```

---

## 10. Метрики успеха

После рефакторинга:
- ✅ Второе переключение tab работает надежно (0 race conditions обнаружено)
- ✅ Каждый модуль < 300 строк (кроме generated кода)
- ✅ Тестовое покрытие > 70%
- ✅ Время сборки не изменилось или быстрее
- ✅ Нет breaking API changes для ComfyUI интеграции
- ✅ Новые разработчики понимают код < чем за 4 часа

---

## 11. Открытые вопросы & Решения

**Базовый класс ноды**: должны ли все пользовательские ноды наследоваться от `ComfyUINode`?
- **За**: Консистентность, общая валидация
- **Против**: Добавляет indirection, concern ComfyUI совместимости
- **Рекомендация**: Реализовать как опциональный mixin

**Async Backend**: можем ли мы использовать async/await без нарушения существующих endpoints?
- **За**: Чище код, лучше concurrency
- **Против**: aiohttp уже обрабатывает async маршруты
- **Рекомендация**: Использовать async обработчики маршрутов, обернуть блокирующие вызовы

**E2E Тестирование**: как тестировать без полного ComfyUI окружения?
- **За**: Быстрее CI, чище изоляция тестов
- **Против**: Упущены ошибки реальной интеграции
- **Рекомендация**: Mock ComfyUI API, использовать Playwright для клиент-тестов

---

## 12. Ссылки

- [Vue.js Жизненный цикл компонента](https://v3.vuejs.org/guide/lifecycle-hooks.html)
- [aiohttp Best Practices](https://docs.aiohttp.org/)
- [Python Type Hints с Dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Jest Фреймворк тестирования](https://jestjs.io/)
- [pytest Best Practices](https://docs.pytest.org/)
