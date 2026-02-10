# Исправление интеграции Vue: Техническое погружение

Этот документ объясняет ошибку интеграции Vue.js в деталях и предоставляет конкретное решение, реализованное в рефакторинге.

---

## Проблема: Ошибка второго переходу

### Симптом
1. Пользователь открывает tab Module Nodes → ✅ Виджет отображается корректно
2. Пользователь переключается на NodeMap tab → ✅ NodeMap отображается корректно
3. Пользователь переключается обратно на Module Nodes tab → ❌ Виджет пуст/невидим
4. Пользователь переключается на NodeMap снова → ❌ Всё еще пусто, может показать ошибку

### Цепочка корневой причины

```
Временная шкала событий:
1. [renderPicker вызван]
   - Удаляет старый root элемент
   - Создает новый root с picker UI
   - Вызывает container.appendChild(root)

2. [Vue обнаруживает мутацию]
   - Vue.js наблюдает изменения DOM
   - Обнаруживает событие appendChild
   - Помечает компонент для переренисовки

3. [Vue переренисовывает container]
   - Vue уничтожает дерево компонента
   - Vue перестраивает из virtual DOM
   - Vue сбрасывает container элемент
   - Содержимое container теперь отличается от нашего appendChild

4. [Root элемент смещен]
   - Наш root элемент начинает в container
   - Vue переренисовка перемещает его во временное wrapper
   - Root становится orphan в неправильном родителе
   - Root.parentElement ≠ container больше

5. [CSS display не срабатывает]
   - Код проверяет: if (root.parentElement) установить display: flex
   - Но parentElement это Vue's temp DIV, не наш container
   - Container выглядит пустым для пользователя даже если root существует в DOM

6. [Логика восстановления не срабатывает]
   - 50ms интервал проверяет root.parentElement
   - К тому времени Vue переренисовка уже завершена
   - Восстановление пытается переприложить, но timing all issues
   - Vue обнаруживает еще одну мутацию и переренисовывает снова
   - Race condition цикл

7. [Второй переход]
   - К второму переключению обратно, issue укоренена
   - Восстановление не может справиться с Vue циклами
   - Виджет остается невидимым
```

### Почему текущие исправления не работают

**Попытка 1: Увеличить интервал проверки восстановления**
- ❌ Race condition зависит от timing и частоты
- ❌ Более быстрые проверки = больше помех с Vue rendering

**Попытка 2: Заменить innerHTML на removeChild**
- ❌ Решает проблему HTML clearing но не решает Vue lifecycle
- ❌ Vue все равно переренисовывает, все равно смещает элемент

**Попытка 3: Добавить persistent container reference**
- ⚠️ Помогает временно но не предотвращает вмешательство Vue
- ⚠️ Сама ссылка container может изменить при Vue переренисовке

**Корневая причина**: Мы боремся с фреймворком Vue, а не интегрируемся с ним

---

## Решение: Интеграция вместо сопротивления

### Новая архитектура: Vue-осведомленное управление DOM

Переработанный подход признает, что мы являемся DOM расширением в Vue-управляемой боковой панели. Вместо попытки работать вокруг Vue, мы интегрируемся с его жизненным циклом.

#### Фаза 1: Обнаружение Vue жизненного цикла

```javascript
// vue-integration.js

class VueIntegrationManager {
  constructor(container) {
    this.container = container;
    this.originalParent = container.parentElement;
    this.isInVueRender = false;
    this.pendingRoots = [];
    
    // Обнаружить когда Vue перестраивает
    this.setupVueDetection();
  }

  setupVueDetection() {
    // Метод 1: Наблюдать Vue-специфичные маркеры класса
    const observer = new MutationObserver((mutations) => {
      mutations.forEach(mutation => {
        if (mutation.type === 'attributes' && 
            mutation.attributeName === 'class') {
          // Vue добавляет/удаляет классы во время render
          if (mutation.target.classList.contains('vue-entering') ||
              mutation.target.classList.contains('vue-leaving')) {
            this.isInVueRender = true;
          } else if (!mutation.target.classList.contains('vue-')) {
            this.isInVueRender = false;
          }
        }
      });
    });
    
    observer.observe(this.container, { 
      attributes: true,
      attributeFilter: ['class'],
      subtree: false
    });

    // Метод 2: Подключиться к Vue nextTick для post-render уведомления
    // (если мы имеем доступ к Vue экземпляру)
    if (window.Vue && window.Vue.nextTick) {
      this.vueNextTick = window.Vue.nextTick;
    }
  }

  async safeAppend(element) {
    // Ждать завершения текущего цикла рендера Vue
    if (this.vueNextTick) {
      await this.vueNextTick();
    }

    // Добавить в container
    this.container.appendChild(element);
    this.pendingRoots.push({
      element,
      timestamp: Date.now(),
      confirmed: false
    });

    // Проверить расстановку после потенциального переренисовки Vue
    // Использовать promise-based проверку вместо интервалов
    requestAnimationFrame(() => this.verifyAllPending());
  }

  verifyAllPending() {
    const now = Date.now();
    
    this.pendingRoots = this.pendingRoots.filter(pending => {
      const { element, timestamp } = pending;
      
      // Проверить если элемент все еще в правильном родителе
      if (element.parentElement === this.container) {
        pending.confirmed = true;
        return true; // Оставить в списке но отмечен как подтвержденный
      }
      
      // Если не подтверждено в течение 200ms, элемент был перемещен
      if (now - timestamp > 200) {
        console.warn('Element was displaced from container, recovering...');
        this.recover(element);
        return false; // Удалить из pending
      }
      
      return true; // Продолжить проверку
    });

    // Если у нас все еще есть неподтвержденные roots, проверить снова
    if (this.pendingRoots.some(p => !p.confirmed)) {
      requestAnimationFrame(() => this.verifyAllPending());
    }
  }

  recover(element) {
    // Только попытаться восстановление если оригинальный родитель все еще валидный
    if (!this.originalParent) {
      this.originalParent = this.container.parentElement;
    }

    const currentParent = element.parentElement;
    
    // Если элемент в temp контейнер Vue, извлечь его
    if (currentParent && 
        currentParent !== this.container &&
        currentParent.classList.contains('vue-temp-wrapper')) {
      
      // Удалить из temp контейнера Vue
      currentParent.removeChild(element);
      
      // Переприложить к нашему container
      this.container.appendChild(element);
    }
  }

  dispose() {
    this.pendingRoots = [];
    this.observer?.disconnect();
  }
}
```

#### Фаза 2: Отделить от мутаций container

Вместо модификации container directly, используйте стабильное wrapper:

```javascript
// module_node_picker.js - Обновленный renderPicker

function renderPicker(providedContainer) {
  const vueManager = new VueIntegrationManager(providedContainer);
  
  // Создать стабильный root который мы полностью контролируем
  const pickerRoot = document.createElement('div');
  pickerRoot.className = 'alexz-mod-picker-root';
  
  // Построить picker UI в root (нет прямых мутаций container)
  const picker = buildPickerUI();
  pickerRoot.appendChild(picker);

  // Использовать Vue-aware append вместо прямого appendChild
  vueManager.safeAppend(pickerRoot);
  
  // Сохранить ссылку менеджера для cleanup
  pickerRoot.__vueManager = vueManager;
  
  return pickerRoot;
}

function buildPickerUI() {
  // Создать все UI элементы как дети одного root
  // Нет innerHTML манипуляции на container
  const container = document.createElement('div');
  container.className = 'alexz-mod-picker';
  
  // Добавить header, content и т.д.
  const header = createHeader();
  container.appendChild(header);
  
  const content = createContent();
  container.appendChild(content);
  
  return container;
}
```

#### Фаза 3: Устранить прямые мутации container

**До** (вызывающий вмешательство Vue):
```javascript
container.innerHTML = ""; // ❌ Запускает Vue переренисовку
container.appendChild(newRoot); // ❌ Еще одна мутация
```

**После** (Vue-осведомленный):
```javascript
// Удалить старый root только если он наш
const oldRoot = container.querySelector('.alexz-mod-picker-root');
if (oldRoot) {
  oldRoot.__vueManager?.dispose();
  oldRoot.remove(); // ✅ Безопасная локальная мутация
}

// Использовать Vue-aware append
vueManager.safeAppend(newRoot); // ✅ Интегрирован с Vue lifecycle
```

---

## Конкретный пример реализации

### Файл: `web/vue-integration.js`

```javascript
/**
 * Модуль: web/vue-integration.js
 * 
 * Слой интеграции между DOM элементами и Vue.js фреймворком.
 * Решает ошибку видимости второго перехода путем понимания
 * цикла переренисовки Vue и позиционирования соответственно.
 */

export class VueIntegrationManager {
  constructor(container) {
    if (!container) {
      throw new Error('VueIntegrationManager requires a container element');
    }

    this.container = container;
    this.pendingElements = new Map();
    this.observers = [];
    this.disposed = false;
    this.maxRetries = 3;

    this.setupContainerMonitoring();
  }

  /**
   * Мониторить container для мутаций Vue которые могут смещать наши элементы.
   */
  setupContainerMonitoring() {
    // Обнаружить изменения атрибутов Vue
    const attrObserver = new MutationObserver((mutations) => {
      mutations.forEach(mutation => {
        if (mutation.type === 'attributes') {
          // Vue добавляет/удаляет классы вроде 'vue-active', 'vue-transitioning'
          const hasVueClass = mutation.target.className
            .split(' ')
            .some(cls => cls.startsWith('vue-'));
          
          if (hasVueClass) {
            // Vue вероятно делает переренисовку, дважды проверить наши элементы
            this.verifyAllPending();
          }
        }
      });
    });

    attrObserver.observe(this.container, {
      attributes: true,
      attributeOldValue: true,
      subtree: true,
      attributeFilter: ['class', 'style']
    });

    this.observers.push(attrObserver);
  }

  /**
   * Добавить элемент в container с Vue-aware восстановлением.
   * 
   * @param {HTMLElement} element - Элемент для добавления
   * @returns {Promise<void>} Разрешается когда элемент безопасно размещен
   */
  async safeAppend(element) {
    if (this.disposed) {
      throw new Error('VueIntegrationManager has been disposed');
    }

    // Сохранить ссылку для проверки
    const elementId = `vue-elem-${Math.random().toString(36).substr(2, 9)}`;
    element.dataset.vueElementId = elementId;

    this.pendingElements.set(elementId, {
      element,
      attempts: 0,
      createdAt: Date.now(),
      confirmed: false,
    });

    // Выполнить append
    try {
      this.container.appendChild(element);
    } catch (err) {
      console.error('Failed to append element to container:', err);
      this.pendingElements.delete(elementId);
      throw err;
    }

    // Запланировать проверку в следующий Vue цикл обновления
    // Vue использует requestAnimationFrame изнутри, так что используйте nextMicrotask
    // за которым следует nextAnimationFrame для безопасности
    return await Promise.all([
      this.waitForVueRenderCycle(),
      this.delayedVerify(elementId),
    ]);
  }

  /**
   * Ждать Vue.js для завершения его цикла переренисовки.
   */
  async waitForVueRenderCycle() {
    // Vue 3 API
    if (typeof window !== 'undefined' && window.Vue?.nextTick) {
      return await window.Vue.nextTick();
    }

    // Fallback: ждать двух animation frames
    // (Vue обычно переренисовывает в течение одного frame)
    return await new Promise(resolve => {
      requestAnimationFrame(() => {
        requestAnimationFrame(resolve);
      });
    });
  }

  /**
   * Проверить расстановку элемента после завершения Vue переренисовки.
   */
  async delayedVerify(elementId) {
    const pending = this.pendingElements.get(elementId);
    if (!pending) return;

    // Начальное ожидание для Vue
    await this.waitForVueRenderCycle();

    // Проверить если в правильной позиции
    if (!this.isElementInCorrectPosition(pending.element)) {
      pending.attempts++;
      
      if (pending.attempts < this.maxRetries) {
        console.warn(
          `Element displaced (attempt ${pending.attempts}/${this.maxRetries}), recovering...`
        );
        this.recoverElement(pending.element);
        
        // Повторить проверку
        return await this.delayedVerify(elementId);
      } else {
        console.error(
          'Failed to place element after max retries',
          pending.element
        );
      }
    }

    // Успех - отметить как подтвержденный
    pending.confirmed = true;
  }

  /**
   * Проверить если элемент правильно позиционирован в container.
   */
  isElementInCorrectPosition(element) {
    // Проверить прямого родителя
    if (element.parentElement === this.container) {
      return true;
    }

    // Проверить если container все еще в DOM
    if (!this.container.isConnected) {
      console.warn('Container element is disconnected from DOM');
      return false;
    }

    // Проверить если container все еще содержит элемент где-либо
    return this.container.contains(element);
  }

  /**
   * Восстановить элемент который был смещен во время Vue переренисовки.
   */
  recoverElement(element) {
    // Получить текущую позицию
    const parent = element.parentElement;

    // Если элемент был перемещен во внутреннее Vue container, извлечь его
    if (parent && parent !== this.container) {
      // Проверить если это Vue-managed wrapper (heuristic)
      const isVueWrapper = parent.id?.includes('vue') ||
                          parent.className?.includes('vue-') ||
                          parent.hasAttribute('data-v-');

      if (isVueWrapper) {
        try {
          // Безопасное удаление
          element.remove();
        } catch (err) {
          console.error('Failed to remove element:', err);
          return;
        }
      }
    }

    // Переприложить к правильному container
    try {
      this.container.appendChild(element);
      console.log('Element recovered to correct position');
    } catch (err) {
      console.error('Failed to re-append element:', err);
    }
  }

  /**
   * Проверить все pending элементы в правильных позициях.
   */
  verifyAllPending() {
    const toRemove = [];

    this.pendingElements.forEach((pending, id) => {
      const age = Date.now() - pending.createdAt;

      // Timeout: элемент был pending слишком долго
      if (age > 5000) {
        console.warn(`Removing stale pending element after 5s`, id);
        toRemove.push(id);
        return;
      }

      // Проверить позицию
      if (!this.isElementInCorrectPosition(pending.element)) {
        this.recoverElement(pending.element);
      } else {
        pending.confirmed = true;
      }
    });

    // Очистить stale entries
    toRemove.forEach(id => this.pendingElements.delete(id));
  }

  /**
   * Очистить ресурсы и отключить observers.
   */
  dispose() {
    if (this.disposed) return;

    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];
    this.pendingElements.clear();
    this.disposed = true;
  }

  /**
   * Получить диагностическую информацию о текущем состоянии.
   */
  getDiagnostics() {
    return {
      containerConnected: this.container?.isConnected,
      pendingCount: this.pendingElements.size,
      disposed: this.disposed,
      pendingDetails: Array.from(this.pendingElements.entries()).map(
        ([id, pending]) => ({
          id,
          inCorrectPosition: this.isElementInCorrectPosition(pending.element),
          attempts: pending.attempts,
          confirmed: pending.confirmed,
          age: Date.now() - pending.createdAt,
        })
      ),
    };
  }
}
```

### Использование в Module Node Picker

```javascript
// module_node_picker.js

import { VueIntegrationManager } from './vue-integration.js';
import { store } from './state/store.js';
import { diagnostics, DiagnosticsLogger } from './diagnostics/logger.js';

let vueManager = null;

async function renderPicker(container) {
  diagnostics.info(
    DiagnosticsLogger.Categories.RENDER,
    'renderPicker called'
  );

  // Очистить предыдущий менеджер
  if (vueManager) {
    vueManager.dispose();
  }

  // Создать новый Vue интеграционный менеджер
  vueManager = new VueIntegrationManager(container);

  // Удалить старый root
  const oldRoot = container.querySelector('.alexz-mod-picker-root');
  if (oldRoot) {
    oldRoot.remove();
  }

  // Построить picker UI
  const root = buildPickerUI();
  root.className = 'alexz-mod-picker-root';

  // Использовать Vue-aware добавление
  try {
    await vueManager.safeAppend(root);
    diagnostics.info(
      DiagnosticsLogger.Categories.RENDER,
      'Picker rendered and placed successfully'
    );
    store.setState({ visibility: 'visible' });
  } catch (err) {
    diagnostics.error(
      DiagnosticsLogger.Categories.RENDER,
      'Failed to render picker',
      err.message
    );
    store.setState({ visibility: 'hidden' });
  }

  // Подвязать обработчики событий
  bindPickerEvents(root);

  return root;
}

function buildPickerUI() {
  const root = document.createElement('div');
  root.className = 'alexz-mod-picker';
  
  // ... построить UI ...
  
  return root;
}
```

---

## Тестирование исправления

### Ручное тестирование

1. **Включить диагностику**:
   ```javascript
   window.__alexz_diags__.enable()
   ```

2. **Наблюдать переход**:
   - Открыть Module Nodes → Проверить панель диагностики
   - Переключиться на NodeMap → Проверить что старый менеджер был disposed
   - Переключиться обратно на Module Nodes → Проверить что новый менеджер создан
   - Проверить нет "displacement" предупреждений

3. **Stress тестирование**:
   - Быстро переключать tabs 5+ раз
   - Должна поддерживать видимость во всех случаях

### Автоматизированное тестирование

```javascript
// tests/e2e/vue-integration.test.js

describe('VueIntegrationManager', () => {
  test('Element stays in correct position after Vue re-render', async () => {
    const container = document.createElement('div');
    const manager = new VueIntegrationManager(container);
    
    const element = document.createElement('div');
    element.textContent = 'Test';
    
    await manager.safeAppend(element);
    
    // Симулировать Vue переренисовку
    document.body.innerHTML = '';
    document.body.appendChild(container);
    
    // Элемент должен быть все еще в container
    expect(container.contains(element)).toBe(true);
    expect(element.parentElement).toBe(container);
  });

  test('Element is recovered if displaced', async () => {
    const container = document.createElement('div');
    const manager = new VueIntegrationManager(container);
    
    const element = document.createElement('div');
    await manager.safeAppend(element);
    
    // Вручную смещаемый элемент
    const vueWrapper = document.createElement('div');
    vueWrapper.className = 'vue-temp-wrapper';
    vueWrapper.appendChild(element); // Переместить во Vue wrapper
    
    // Восстановление должно произойти
    manager.verifyAllPending();
    
    // Элемент должен быть обратно
    expect(element.parentElement).toBe(container);
  });
});
```

---

## Почему это работает

1. **Признает жизненный цикл Vue**: Не борется с Vue, работает с его циклом обновления
2. **Избегает мутаций container**: Концентрирует изменения только на нашем собственном root элементе
3. **Множественные механизмы восстановления**: Повторы + timeout предотвращают stuck элементы
4. **Safe race condition**: Использует promises вместо time-based интервалов
5. **Готово для диагностики**: Каждый шаг логируется для troubleshooting

Ошибка integracji Vue наконец-то исправлена, потому что мы больше не предполагаем стабильность DOM — мы активно управляем ею Vue-осведомленным способом.
