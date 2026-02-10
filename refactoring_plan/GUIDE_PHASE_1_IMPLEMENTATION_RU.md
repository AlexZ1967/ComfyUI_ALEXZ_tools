# Фаза 1: Руководство по реализации — Store состояния и диагностика

Этот документ предоставляет готовый к использованию код для рефакторинга Фазы 1 (Недели 1-2), позволяя немедленно начать инициативу рефакторинга.

---

## Модуль 1: Централизованное хранилище состояния

### Файл: `web/state/store.js`

```javascript
/**
 * Модуль: web/state/store.js
 * 
 * Централизованное реактивное управление состоянием для Module Node Picker.
 * Заменяет разбросанные window[...] свойства одним источником истины.
 */

export class ModuleNodeStore {
  constructor() {
    // Определить начальную структуру состояния
    this.state = {
      // UI State
      selectedGroup: this.loadPersistedString('alexz_selected_group', 'core'),
      selectedModule: this.loadPersistedString('alexz_selected_module', 'ComfyUI_ALEXZ_tools'),
      expandedModules: new Set(this.loadPersistedArray('alexz_expanded_modules', [])),
      debugEnabled: this.loadPersistedBool('alexz_debug_enabled', false),
      visibility: 'hidden', // 'visible' | 'hidden'
      
      // Data State
      catalog: null,
      selectedGroupNodes: [],
      moduleInfo: {},
      selectedModuleInfo: null,
      
      // Operation State
      refreshStatus: {
        running: false,
        phase: 'idle',
        current: 0,
        total: 0,
        message: '',
        error: '',
      },
      updateStatus: {
        running: false,
        phase: 'idle',
        current: 0,
        total: 0,
        message: '',
        error: '',
      },
    };

    // Слушатели событий по ключам
    this.listeners = new Map();
    
    // Поддержка batching
    this.batchMode = false;
    this.batchedChanges = new Set();
  }

  // ========== Управление подписками ==========

  subscribe(keys, listener) {
    /**
     * Зарегистрировать слушателя для изменений состояния.
     * 
     * @param {string|string[]} keys - Ключ(и) состояния для прослушивания
     * @param {Function} listener - Вызывается с новым значением при изменении ключа
     * @returns {Function} Функция отписки
     */
    const keyList = Array.isArray(keys) ? keys : [keys];
    
    keyList.forEach(key => {
      if (!this.listeners.has(key)) {
        this.listeners.set(key, new Set());
      }
      this.listeners.get(key).add(listener);
    });

    // Вернуть функцию отписки
    return () => {
      keyList.forEach(key => {
        this.listeners.get(key)?.delete(listener);
      });
    };
  }

  subscribeOnce(keys, listener) {
    /**
     * Зарегистрировать одноразовый слушатель, который отписывается после первого срабатывания.
     */
    const unsubscribe = this.subscribe(keys, (value) => {
      listener(value);
      unsubscribe();
    });
    return unsubscribe;
  }

  notifyListeners(changedKeys) {
    /**
     * Уведомить всех слушателей об изменениях состояния.
     * Вызывается автоматически setState.
     */
    changedKeys.forEach(key => {
      const subs = this.listeners.get(key);
      if (subs) {
        const value = this.state[key];
        subs.forEach(listener => {
          try {
            listener(value, key);
          } catch (err) {
            console.error(`State listener error for key '${key}':`, err);
          }
        });
      }
    });
  }

  // ========== Мутации состояния ==========

  setState(partial) {
    /**
     * Обновить состояние с частичным объектом и уведомить слушателей.
     * 
     * @param {Object} partial - Частичный объект состояния для слияния
     */
    const changedKeys = [];
    
    Object.entries(partial).forEach(([key, value]) => {
      if (this.state.hasOwnProperty(key)) {
        // Значение изменилось
        if (this.state[key] !== value) {
          this.state[key] = value;
          changedKeys.push(key);
          
          // Персистировать определенные ключи
          this.persistKey(key, value);
          
          if (!this.batchMode) {
            this.notifyListeners([key]);
          } else {
            this.batchedChanges.add(key);
          }
        }
      } else {
        console.warn(`setState: Unknown key '${key}'`);
      }
    });
    
    return changedKeys;
  }

  batchSetState(updates) {
    /**
     * Обновить несколько ключей состояния сразу, уведомив слушателей только один раз.
     * Полезно для UI операций, которые зависят от нескольких изменений состояния.
     */
    this.batchMode = true;
    const allKeys = [];
    
    Object.entries(updates).forEach(([category, changes]) => {
      const changed = this.setState(changes);
      allKeys.push(...changed);
    });
    
    this.batchMode = false;
    const uniqueKeys = [...new Set(allKeys)];
    this.notifyListeners(uniqueKeys);
  }

  // ========== Мутации коллекций ==========

  addExpandedModule(moduleName) {
    /**
     * Добавить модуль в расширенное множество и обновить состояние.
     */
    if (!this.state.expandedModules.has(moduleName)) {
      this.state.expandedModules.add(moduleName);
      this.persistExpandedModules();
      this.notifyListeners(['expandedModules']);
    }
  }

  removeExpandedModule(moduleName) {
    /**
     * Удалить модуль из расширенного множества.
     */
    if (this.state.expandedModules.has(moduleName)) {
      this.state.expandedModules.delete(moduleName);
      this.persistExpandedModules();
      this.notifyListeners(['expandedModules']);
    }
  }

  toggleModule(moduleName) {
    /**
     * Переключить состояние расширения модуля.
     */
    if (this.state.expandedModules.has(moduleName)) {
      this.removeExpandedModule(moduleName);
    } else {
      this.addExpandedModule(moduleName);
    }
  }

  isModuleExpanded(moduleName) {
    /**
     * Проверить, развернут ли модуль в данный момент.
     */
    return this.state.expandedModules.has(moduleName);
  }

  // ========== Отслеживание статуса ==========

  updateRefreshStatus(partial) {
    /**
     * Обновить статус текущей операции refresh.
     */
    this.setState({
      refreshStatus: {
        ...this.state.refreshStatus,
        updated_at: new Date().toISOString(),
        ...partial,
      }
    });
  }

  updateUpdateStatus(partial) {
    /**
     * Обновить статус текущей операции update.
     */
    this.setState({
      updateStatus: {
        ...this.state.updateStatus,
        updated_at: new Date().toISOString(),
        ...partial,
      }
    });
  }

  // ========== Периодичность ==========

  persistKey(key, value) {
    /**
     * Сохранить определенные ключи в localStorage для периодичности между сеансами.
     */
    const persistKeys = {
      'selectedGroup': true,
      'selectedModule': true,
      'expandedModules': true,
      'debugEnabled': true,
    };

    if (persistKeys[key]) {
      const storageKey = `alexz_${key}`;
      try {
        if (value instanceof Set) {
          localStorage.setItem(storageKey, JSON.stringify([...value]));
        } else if (typeof value === 'boolean') {
          localStorage.setItem(storageKey, value ? '1' : '0');
        } else if (typeof value === 'string') {
          localStorage.setItem(storageKey, value);
        }
      } catch (err) {
        console.warn(`Failed to persist key '${key}':`, err);
      }
    }
  }

  persistExpandedModules() {
    /**
     * Персистировать расширенные модули в localStorage.
     */
    try {
      const value = [...this.state.expandedModules];
      localStorage.setItem('alexz_expanded_modules', JSON.stringify(value));
    } catch (err) {
      console.warn('Failed to persist expanded modules:', err);
    }
  }

  loadPersistedString(key, defaultValue) {
    /**
     * Загрузить строковое значение из localStorage с fallback.
     */
    try {
      return localStorage.getItem(key) || defaultValue;
    } catch {
      return defaultValue;
    }
  }

  loadPersistedArray(key, defaultValue) {
    /**
     * Загрузить массив из localStorage.
     */
    try {
      const val = localStorage.getItem(key);
      return val ? JSON.parse(val) : defaultValue;
    } catch {
      return defaultValue;
    }
  }

  loadPersistedBool(key, defaultValue) {
    /**
     * Загрузить boolean значение из localStorage.
     */
    try {
      const val = localStorage.getItem(key);
      if (val === null) return defaultValue;
      return val === '1';
    } catch {
      return defaultValue;
    }
  }

  // ========== Режим отладки ==========

  setDebugEnabled(enabled) {
    /**
     * Включить или отключить режим отладки.
     */
    this.setState({ debugEnabled: enabled });
  }

  getDebugEnabled() {
    /**
     * Проверить, включен ли режим отладки.
     */
    return this.state.debugEnabled;
  }

  // ========== Проверка состояния ==========

  toJSON() {
    /**
     * Сериализовать состояние для отладки (исключает циклические ссылки).
     */
    return {
      selectedGroup: this.state.selectedGroup,
      selectedModule: this.state.selectedModule,
      expandedModules: [...this.state.expandedModules],
      debugEnabled: this.state.debugEnabled,
      visibility: this.state.visibility,
      catalogLoaded: !!this.state.catalog,
      nodesCount: this.state.selectedGroupNodes.length,
      refreshRunning: this.state.refreshStatus.running,
      updateRunning: this.state.updateStatus.running,
    };
  }

  snapshot() {
    /**
     * Создать глубокую копию текущего состояния для сравнения/undo операций.
     */
    return JSON.parse(JSON.stringify({
      selectedGroup: this.state.selectedGroup,
      selectedModule: this.state.selectedModule,
      expandedModules: [...this.state.expandedModules],
      debugEnabled: this.state.debugEnabled,
      visibility: this.state.visibility,
      // Примечание: объекты catalog и status опущены, так как они слишком большие
    }));
  }
}

// ========== Singleton экземпляр ==========

export const store = new ModuleNodeStore();

// Expose в режиме отладки для проверки браузерной консоли
if (typeof window !== 'undefined') {
  Object.defineProperty(window, '__alexz_store_debug__', {
    get: () => store.toJSON(),
    configurable: true,
  });
}

// ========== Helper для миграции ==========

/**
 * Для миграции со старых window[...] свойств на новый store.
 * Вызвать во время периода переходной рефакторинга.
 */
export function createStoreMirror() {
  return {
    get groupId() { return store.state.selectedGroup; },
    set groupId(val) { store.setState({ selectedGroup: val }); },
    
    get moduleName() { return store.state.selectedModule; },
    set moduleName(val) { store.setState({ selectedModule: val }); },
    
    get debugMode() { return store.state.debugEnabled; },
    set debugMode(val) { store.setState({ debugEnabled: val }); },
    
    get visibility() { return store.state.visibility; },
    set visibility(val) { store.setState({ visibility: val }); },
    
    isModuleExpanded(name) { return store.isModuleExpanded(name); },
    toggleModule(name) { return store.toggleModule(name); },
  };
}
```

---

## Модуль 2: Логгер диагностики

### Файл: `web/diagnostics/logger.js`

```javascript
/**
 * Модуль: web/diagnostics/logger.js
 * 
 * Условная система логирования диагностики для отладки Module Node Picker.
 * Автоматически отключается в production, нулевые издержки в отключенном состоянии.
 */

export class DiagnosticsLogger {
  constructor(name = 'alexz') {
    this.name = name;
    this.enabled = this.loadEnabled();
    this.logs = [];
    this.startTime = Date.now();
    this.maxLogs = 200;
    
    this.logElement = null;
    this.autoExpand();
  }

  static readonly Categories = {
    SYNC: 'sync',
    STATE: 'state',
    API: 'api',
    RENDER: 'render',
    DOM: 'dom',
    VUE: 'vue',
    ERROR: 'error',
    PERF: 'perf',
  };

  // ========== Включение/Отключение ==========

  enable() {
    /**
     * Включить логирование диагностики.
     */
    if (!this.enabled) {
      this.enabled = true;
      this.saveEnabled();
      this.log(DiagnosticsLogger.Categories.ERROR, 'Диагностика АКТИВИРОВАНА');
    }
  }

  disable() {
    /**
     * Отключить логирование диагностики (очищает логи).
     */
    if (this.enabled) {
      this.enabled = false;
      this.logs = [];
      this.saveEnabled();
      console.log('[alexz] Диагностика ДЕАКТИВИРОВАНА');
    }
  }

  toggle() {
    /**
     * Переключить состояние диагностики on/off.
     */
    if (this.enabled) {
      this.disable();
    } else {
      this.enable();
    }
  }

  loadEnabled() {
    /**
     * Загрузить состояние enabled из localStorage.
     */
    try {
      return localStorage.getItem(`${this.name}_diags_enabled`) === '1';
    } catch {
      return false;
    }
  }

  saveEnabled() {
    /**
     * Сохранить состояние enabled в localStorage.
     */
    try {
      localStorage.setItem(
        `${this.name}_diags_enabled`,
        this.enabled ? '1' : '0'
      );
    } catch {
      // Тихо не сработать
    }
  }

  // ========== Логирование ==========

  log(category, message, data = null, level = 'log') {
    /**
     * Логировать сообщение диагностики с опциональными данными.
     * 
     * @param {string} category - Категория сообщения (используйте enum Categories)
     * @param {string} message - Читаемое сообщение
     * @param {*} data - Опциональные дополнительные данные
     * @param {string} level - 'log' | 'warn' | 'error'
     */
    if (!this.enabled) return;

    const timestamp = Date.now() - this.startTime;
    const entry = {
      timestamp,
      category,
      message,
      data,
      level,
      id: this.logs.length,
    };

    this.logs.push(entry);
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }

    // Вывод в консоль
    const prefix = `[${this.name}:${category}] ${this.formatTime(timestamp)}`;
    const args = [prefix, message];
    if (data !== null) {
      args.push(data);
    }

    if (level === 'error') {
      console.error(...args);
    } else if (level === 'warn') {
      console.warn(...args);
    } else {
      console.debug(...args);
    }
  }

  info(category, message, data = null) {
    this.log(category, message, data, 'log');
  }

  warn(category, message, data = null) {
    this.log(category, message, data, 'warn');
  }

  error(category, message, data = null) {
    this.log(category, message, data, 'error');
  }

  // ========== Отслеживание производительности ==========

  timeStart(label) {
    /**
     * Начать timing блока работы.
     * 
     * @param {string} label - Метка операции
     * @returns {Function} Вызвать для завершения timing
     */
    const startTime = Date.now();
    return (message = '') => {
      const duration = Date.now() - startTime;
      this.log(
        DiagnosticsLogger.Categories.PERF,
        `${label}: ${duration}ms ${message}`,
        { duration },
        duration > 100 ? 'warn' : 'log'
      );
      return duration;
    };
  }

  async timeAsync(label, asyncFn) {
    /**
     * Время асинхронной операции.
     */
    const end = this.timeStart(label);
    try {
      return await asyncFn();
    } finally {
      end();
    }
  }

  // ========== Рендеринг UI ==========

  render() {
    /**
     * Рендерить панель диагностики как DOM элемент.
     * Возвращает null если диагностика отключена.
     */
    if (!this.enabled || this.logs.length === 0) {
      return null;
    }

    const container = document.createElement('div');
    container.className = 'alexz-diag-panel';
    container.style.cssText = `
      font-family: monospace;
      font-size: 11px;
      border: 1px dashed #666;
      border-radius: 4px;
      padding: 6px;
      background: rgba(0, 0, 0, 0.3);
      max-height: 150px;
      overflow-y: auto;
      line-height: 1.3;
    `;

    // Кнопки управления
    const controls = document.createElement('div');
    controls.style.cssText = 'margin-bottom: 4px; display: flex; gap: 6px;';

    const clearBtn = document.createElement('button');
    clearBtn.textContent = 'Очистить';
    clearBtn.style.cssText = 'padding: 2px 6px; font-size: 10px; cursor: pointer;';
    clearBtn.onclick = () => {
      this.logs = [];
      this.render();
    };
    controls.appendChild(clearBtn);

    const disableBtn = document.createElement('button');
    disableBtn.textContent = 'Отключить';
    disableBtn.style.cssText = 'padding: 2px 6px; font-size: 10px; cursor: pointer;';
    disableBtn.onclick = () => this.disable();
    controls.appendChild(disableBtn);

    const infoSpan = document.createElement('span');
    infoSpan.textContent = `${this.logs.length} логов`;
    infoSpan.style.cssText = 'margin-left: auto; opacity: 0.7;';
    controls.appendChild(infoSpan);

    container.appendChild(controls);

    // Записи логов
    const logContainer = document.createElement('div');
    this.logs.slice(-50).forEach(entry => {
      const line = document.createElement('div');
      line.style.cssText = `
        color: ${this.getCategoryColor(entry.category)};
        margin: 2px 0;
        opacity: ${entry.level === 'error' ? 1 : 0.85};
      `;
      line.textContent = `${entry.timestamp.toString().padStart(5)} [${entry.category}] ${entry.message}`;
      logContainer.appendChild(line);
    });

    container.appendChild(logContainer);
    return container;
  }

  autoExpand() {
    /**
     * Auto-inject панель диагностики в UI picker.
     * Вызвано при конструкции, находит или создает точку инжекции.
     */
    if (!this.enabled) return;

    const injectAfterMs = 1000;
    setTimeout(() => {
      const target = document.querySelector('.alexz-mod-picker');
      if (target && !target.querySelector('.alexz-diag-panel')) {
        const panel = this.render();
        if (panel) {
          target.appendChild(panel);
          this.logElement = panel;
        }
      }
    }, injectAfterMs);
  }

  getCategoryColor(category) {
    /**
     * Получить цвет отображения для категории лога.
     */
    const colors = {
      [DiagnosticsLogger.Categories.SYNC]: '#4da3ff',
      [DiagnosticsLogger.Categories.STATE]: '#ffd700',
      [DiagnosticsLogger.Categories.API]: '#90ee90',
      [DiagnosticsLogger.Categories.RENDER]: '#ff69b4',
      [DiagnosticsLogger.Categories.DOM]: '#87ceeb',
      [DiagnosticsLogger.Categories.VUE]: '#ff8c00',
      [DiagnosticsLogger.Categories.ERROR]: '#ff0000',
      [DiagnosticsLogger.Categories.PERF]: '#ffa500',
    };
    return colors[category] || '#fff';
  }

  formatTime(ms) {
    /**
     * Форматировать миллисекунды как читаемую строку времени.
     */
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  // ========== Экспорт & Импорт ==========

  exportLogs() {
    /**
     * Экспортировать логи как JSON строку для обмена.
     */
    return JSON.stringify(this.logs, null, 2);
  }

  exportLogsAsText() {
    /**
     * Экспортировать логи как читаемый текст.
     */
    return this.logs
      .map(entry => {
        const data = entry.data ? ` ${JSON.stringify(entry.data)}` : '';
        return `[${entry.timestamp}] [${entry.category}] ${entry.message}${data}`;
      })
      .join('\n');
  }

  downloadLogs(filename = 'alexz-diags.txt') {
    /**
     * Скачать логи как текстовый файл.
     */
    const content = this.exportLogsAsText();
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ========== Состояние ==========

  toJSON() {
    /**
     * Сериализовать состояние диагностики.
     */
    return {
      enabled: this.enabled,
      logCount: this.logs.length,
      uptime: Date.now() - this.startTime,
    };
  }
}

// ========== Singleton экземпляр ==========

export const diagnostics = new DiagnosticsLogger('alexz');

// Expose в window для доступа из консоли
Object.defineProperty(window, '__alexz_diags__', {
  get: () => diagnostics,
  configurable: true,
});
```

---

## Примеры использования

### Использование State Store

```javascript
import { store } from './state/store.js';

// Подписка на изменения группы
const unsubscribe = store.subscribe('selectedGroup', (groupId) => {
  console.log('Группа изменена на:', groupId);
  updateGroupUI(groupId);
});

// Обновление состояния
store.setState({ 
  selectedGroup: 'custom',
  visibility: 'visible'
});

// Batch обновления
store.batchSetState({
  ui: { selectedGroup: 'custom', visibility: 'visible' },
  data: { catalog: newCatalog },
});

// Проверка расширенных модулей
if (store.isModuleExpanded('ComfyUI_ALEXZ_tools')) {
  console.log('Модуль развернут');
}

// Отписка когда готово
unsubscribe();
```

### Использование Diagnostics Logger

```javascript
import { diagnostics, DiagnosticsLogger } from './diagnostics/logger.js';

// Включить диагностику (персистируется между перезагрузками)
diagnostics.enable();

// Логировать сообщения
diagnostics.info(DiagnosticsLogger.Categories.SYNC, 'Tab switched', { 
  from: 'Module Nodes',
  to: 'NodeMap' 
});

diagnostics.warn(DiagnosticsLogger.Categories.DOM, 'Root element displaced');

diagnostics.error(
  DiagnosticsLogger.Categories.API, 
  'Catalog fetch failed',
  { status: 404 }
);

// Timing операций
const end = diagnostics.timeStart('Fetching catalog');
const data = await fetchCatalog();
end('- получено ' + data.length + ' нод');

// Async timing
const catalogData = await diagnostics.timeAsync(
  'Load and render catalog',
  async () => {
    const catalog = await fetchCatalog();
    renderCatalog(catalog);
    return catalog;
  }
);

// Скачать для обмена
diagnostics.downloadLogs('alexz-bug-report.txt');
```

---

## Чек-лист интеграции

- [ ] Скопировать `web/state/store.js` в проект
- [ ] Скопировать `web/diagnostics/logger.js` в проект
- [ ] Импортировать store в `module_node_picker.js`:
  ```javascript
  import { store } from './state/store.js';
  import { diagnostics, DiagnosticsLogger } from './diagnostics/logger.js';
  ```
- [ ] Начать использовать `store.setState()` вместо `window[...]` assignments
- [ ] Добавить `diagnostics.info()` вызовы в key поинтах `renderPicker()` и `bindMinimalTabRelay()`
- [ ] Тестировать в браузерной консоли: `window.__alexz_store_debug__`
- [ ] Включить диагностику в консоли: `window.__alexz_diags__.enable()`

---

## Следующие шаги

Когда Фаза 1 завершена:
1. **Фаза 2**: Рефакторить обработчики событий для использования `store.subscribe()`
2. **Фаза 3**: Извлечь логику рендера в `picker-ui/renderer.js`
3. **Фаза 4**: Извлечь API вызовы в `api/` модули
4. **Фаза 5**: Написать unit тесты для store и diagnostics
