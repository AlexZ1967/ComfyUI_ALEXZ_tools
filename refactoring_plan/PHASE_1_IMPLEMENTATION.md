# Phase 1 Implementation Guide: State Store & Diagnostics

This document provides ready-to-use code for Phase 1 refactoring (Weeks 1-2), allowing immediate start on the refactoring initiative.

---

## Module 1: Centralized State Store

### File: `web/state/store.js`

```javascript
/**
 * Module: web/state/store.js
 * 
 * Centralized reactive state management for Module Node Picker.
 * Replaces scattered window[...] properties with single source of truth.
 */

export class ModuleNodeStore {
  constructor() {
    // Define initial state structure
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

    // Event listeners by key
    this.listeners = new Map();
    
    // Batching support
    this.batchMode = false;
    this.batchedChanges = new Set();
  }

  // ========== Subscription Management ==========

  subscribe(keys, listener) {
    /**
     * Register a listener for state changes.
     * 
     * @param {string|string[]} keys - State key(s) to listen for
     * @param {Function} listener - Called with new value when key changes
     * @returns {Function} Unsubscribe function
     */
    const keyList = Array.isArray(keys) ? keys : [keys];
    
    keyList.forEach(key => {
      if (!this.listeners.has(key)) {
        this.listeners.set(key, new Set());
      }
      this.listeners.get(key).add(listener);
    });

    // Return unsubscribe function
    return () => {
      keyList.forEach(key => {
        this.listeners.get(key)?.delete(listener);
      });
    };
  }

  subscribeOnce(keys, listener) {
    /**
     * Register a one-time listener that auto-unsubscribes after first trigger.
     */
    const unsubscribe = this.subscribe(keys, (value) => {
      listener(value);
      unsubscribe();
    });
    return unsubscribe;
  }

  notifyListeners(changedKeys) {
    /**
     * Notify all listeners about state changes.
     * Called automatically by setState.
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

  // ========== State Mutations ==========

  setState(partial) {
    /**
     * Update state with partial object and notify listeners.
     * 
     * @param {Object} partial - Partial state object to merge
     */
    const changedKeys = [];
    
    Object.entries(partial).forEach(([key, value]) => {
      if (this.state.hasOwnProperty(key)) {
        // Value changed
        if (this.state[key] !== value) {
          this.state[key] = value;
          changedKeys.push(key);
          
          // Persist certain keys
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
     * Update multiple state keys at once, notifying listeners only once.
     * Useful for UI operations that depend on multiple state changes.
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

  // ========== Collection Mutations ==========

  addExpandedModule(moduleName) {
    /**
     * Add module to expanded set and update state.
     */
    if (!this.state.expandedModules.has(moduleName)) {
      this.state.expandedModules.add(moduleName);
      this.persistExpandedModules();
      this.notifyListeners(['expandedModules']);
    }
  }

  removeExpandedModule(moduleName) {
    /**
     * Remove module from expanded set.
     */
    if (this.state.expandedModules.has(moduleName)) {
      this.state.expandedModules.delete(moduleName);
      this.persistExpandedModules();
      this.notifyListeners(['expandedModules']);
    }
  }

  toggleModule(moduleName) {
    /**
     * Toggle module expanded state.
     */
    if (this.state.expandedModules.has(moduleName)) {
      this.removeExpandedModule(moduleName);
    } else {
      this.addExpandedModule(moduleName);
    }
  }

  isModuleExpanded(moduleName) {
    /**
     * Check if module is currently expanded.
     */
    return this.state.expandedModules.has(moduleName);
  }

  // ========== Status Tracking ==========

  updateRefreshStatus(partial) {
    /**
     * Update ongoing refresh operation status.
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
     * Update ongoing update operation status.
     */
    this.setState({
      updateStatus: {
        ...this.state.updateStatus,
        updated_at: new Date().toISOString(),
        ...partial,
      }
    });
  }

  // ========== Persistence ==========

  persistKey(key, value) {
    /**
     * Save specific keys to localStorage for persistence across sessions.
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
     * Persist expanded modules set to localStorage.
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
     * Load string value from localStorage with fallback.
     */
    try {
      return localStorage.getItem(key) || defaultValue;
    } catch {
      return defaultValue;
    }
  }

  loadPersistedArray(key, defaultValue) {
    /**
     * Load array value from localStorage.
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
     * Load boolean value from localStorage.
     */
    try {
      const val = localStorage.getItem(key);
      if (val === null) return defaultValue;
      return val === '1';
    } catch {
      return defaultValue;
    }
  }

  // ========== Debug Mode ==========

  setDebugEnabled(enabled) {
    /**
     * Enable or disable debug mode.
     */
    this.setState({ debugEnabled: enabled });
  }

  getDebugEnabled() {
    /**
     * Check if debug mode is enabled.
     */
    return this.state.debugEnabled;
  }

  // ========== State Inspection ==========

  toJSON() {
    /**
     * Serialize state for debugging (excludes circular references).
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
     * Create a deep copy of current state for comparison/undo operations.
     */
    return JSON.parse(JSON.stringify({
      selectedGroup: this.state.selectedGroup,
      selectedModule: this.state.selectedModule,
      expandedModules: [...this.state.expandedModules],
      debugEnabled: this.state.debugEnabled,
      visibility: this.state.visibility,
      // Note: catalog and status objects omitted as they're too large
    }));
  }
}

// ========== Singleton Instance ==========

export const store = new ModuleNodeStore();

// Expose in debug mode for browser console inspection
if (typeof window !== 'undefined') {
  Object.defineProperty(window, '__alexz_store_debug__', {
    get: () => store.toJSON(),
    configurable: true,
  });
}

// ========== Migration Helper ==========

/**
 * For migrating from old window[...] properties to new store.
 * Call during refactoring transition period.
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

## Module 2: Diagnostic Logger

### File: `web/diagnostics/logger.js`

```javascript
/**
 * Module: web/diagnostics/logger.js
 * 
 * Conditional diagnostic logging system for debugging Module Node Picker.
 * Automatically disabled in production, zero overhead when off.
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

  public static readonly Categories = {
    SYNC: 'sync',
    STATE: 'state',
    API: 'api',
    RENDER: 'render',
    DOM: 'dom',
    VUE: 'vue',
    ERROR: 'error',
    PERF: 'perf',
  };

  // ========== Enable/Disable ==========

  enable() {
    /**
     * Enable diagnostic logging.
     */
    if (!this.enabled) {
      this.enabled = true;
      this.saveEnabled();
      this.log(DiagnosticsLogger.Categories.ERROR, 'Diagnostics ENABLED');
    }
  }

  disable() {
    /**
     * Disable diagnostic logging (clears logs).
     */
    if (this.enabled) {
      this.enabled = false;
      this.logs = [];
      this.saveEnabled();
      console.log('[alexz] Diagnostics DISABLED');
    }
  }

  toggle() {
    /**
     * Toggle diagnostic state on/off.
     */
    if (this.enabled) {
      this.disable();
    } else {
      this.enable();
    }
  }

  loadEnabled() {
    /**
     * Load enabled state from localStorage.
     */
    try {
      return localStorage.getItem(`${this.name}_diags_enabled`) === '1';
    } catch {
      return false;
    }
  }

  saveEnabled() {
    /**
     * Save enabled state to localStorage.
     */
    try {
      localStorage.setItem(
        `${this.name}_diags_enabled`,
        this.enabled ? '1' : '0'
      );
    } catch {
      // Silently fail
    }
  }

  // ========== Logging ==========

  log(category, message, data = null, level = 'log') {
    /**
     * Log a diagnostic message with optional data.
     * 
     * @param {string} category - Message category (use Categories enum)
     * @param {string} message - Human-readable message
     * @param {*} data - Optional additional data
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

    // Console output
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

  // ========== Performance Tracking ==========

  timeStart(label) {
    /**
     * Start timing a block of work.
     * 
     * @param {string} label - Operation label
     * @returns {Function} Call to end timing
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
     * Time an async operation.
     */
    const end = this.timeStart(label);
    try {
      return await asyncFn();
    } finally {
      end();
    }
  }

  // ========== UI Rendering ==========

  render() {
    /**
     * Render diagnostic panel as DOM element.
     * Returns null if diagnostics disabled.
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

    // Control buttons
    const controls = document.createElement('div');
    controls.style.cssText = 'margin-bottom: 4px; display: flex; gap: 6px;';

    const clearBtn = document.createElement('button');
    clearBtn.textContent = 'Clear';
    clearBtn.style.cssText = 'padding: 2px 6px; font-size: 10px; cursor: pointer;';
    clearBtn.onclick = () => {
      this.logs = [];
      this.render();
    };
    controls.appendChild(clearBtn);

    const disableBtn = document.createElement('button');
    disableBtn.textContent = 'Disable';
    disableBtn.style.cssText = 'padding: 2px 6px; font-size: 10px; cursor: pointer;';
    disableBtn.onclick = () => this.disable();
    controls.appendChild(disableBtn);

    const infoSpan = document.createElement('span');
    infoSpan.textContent = `${this.logs.length} logs`;
    infoSpan.style.cssText = 'margin-left: auto; opacity: 0.7;';
    controls.appendChild(infoSpan);

    container.appendChild(controls);

    // Log entries
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
     * Auto-inject diagnostic panel into the picker UI.
     * Called on construction, finds or creates injection point.
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
     * Get display color for log category.
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
     * Format milliseconds as readable time string.
     */
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  // ========== Export & Import ==========

  exportLogs() {
    /**
     * Export logs as JSON string for sharing.
     */
    return JSON.stringify(this.logs, null, 2);
  }

  exportLogsAsText() {
    /**
     * Export logs as human-readable text.
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
     * Download logs as text file.
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

  // ========== State ==========

  toJSON() {
    /**
     * Serialize diagnostic state.
     */
    return {
      enabled: this.enabled,
      logCount: this.logs.length,
      uptime: Date.now() - this.startTime,
    };
  }
}

// ========== Singleton Instance ==========

export const diagnostics = new DiagnosticsLogger('alexz');

// Expose in window for console access
Object.defineProperty(window, '__alexz_diags__', {
  get: () => diagnostics,
  configurable: true,
});
```

---

## Usage Examples

### Using the State Store

```javascript
import { store } from './state/store.js';

// Subscribe to group changes
const unsubscribe = store.subscribe('selectedGroup', (groupId) => {
  console.log('Group changed to:', groupId);
  updateGroupUI(groupId);
});

// Update state
store.setState({ 
  selectedGroup: 'custom',
  visibility: 'visible'
});

// Batch updates
store.batchSetState({
  ui: { selectedGroup: 'custom', visibility: 'visible' },
  data: { catalog: newCatalog },
});

// Check expanded modules
if (store.isModuleExpanded('ComfyUI_ALEXZ_tools')) {
  console.log('Module is expanded');
}

// Unsubscribe when done
unsubscribe();
```

### Using the Diagnostic Logger

```javascript
import { diagnostics, DiagnosticsLogger } from './diagnostics/logger.js';

// Enable diagnostics (persists across page reloads)
diagnostics.enable();

// Log messages
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

// Time operations
const end = diagnostics.timeStart('Fetching catalog');
const data = await fetchCatalog();
end('- received ' + data.length + ' nodes');

// Async timing
const catalogData = await diagnostics.timeAsync(
  'Load and render catalog',
  async () => {
    const catalog = await fetchCatalog();
    renderCatalog(catalog);
    return catalog;
  }
);

// Download for sharing
diagnostics.downloadLogs('alexz-bug-report.txt');
```

---

## Integration Checklist

- [ ] Copy `web/state/store.js` into project
- [ ] Copy `web/diagnostics/logger.js` into project
- [ ] Import store in `module_node_picker.js`:
  ```javascript
  import { store } from './state/store.js';
  import { diagnostics, DiagnosticsLogger } from './diagnostics/logger.js';
  ```
- [ ] Start using `store.setState()` instead of `window[...]` assignments
- [ ] Add `diagnostics.info()` calls at key points in `renderPicker()` and `bindMinimalTabRelay()`
- [ ] Test in browser console: `window.__alexz_store_debug__`
- [ ] Enable diagnostics in console: `window.__alexz_diags__.enable()`

---

## Next Steps

Once Phase 1 is complete:
1. **Phase 2**: Refactor event handlers to use `store.subscribe()`
2. **Phase 3**: Extract render logic into `picker-ui/renderer.js`
3. **Phase 4**: Extract API calls into `api/` modules
4. **Phase 5**: Write unit tests for store and diagnostics

