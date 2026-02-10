/**
 * Module: web/diagnostics/logger.js
 * Author: AlexZ1967
 * Last updated: 2026-02-10
 *
 * Description:
 *   In-memory diagnostics logger for Module Node Picker.
 *
 * Purpose:
 *   Centralizes info/warn/error logging with bounded history and runtime
 *   debug toggle while preserving concise console output behavior.
 */

/**
 * Create diagnostics logger with bounded memory buffer.
 */
export function createModuleDiagnosticsLogger(options = {}) {
    const namespace = String(options.namespace || "ALEXZ_tools Node Picker");
    const maxEntries = Math.max(20, Number(options.maxEntries || 200));
    let debugEnabled = Boolean(options.debugEnabled);
    const entries = [];

    const pushEntry = (level, message, data) => {
        const entry = {
            ts: new Date().toISOString(),
            level: String(level || "info"),
            message: String(message || ""),
            data: data === undefined ? null : data,
        };
        entries.push(entry);
        if (entries.length > maxEntries) {
            entries.splice(0, entries.length - maxEntries);
        }
        return entry;
    };

    const consoleWrite = (level, message, data) => {
        const text = `[${namespace}] ${message}`;
        if (level === "warn") {
            console.warn(text, data ?? "");
            return;
        }
        if (level === "error") {
            console.error(text, data ?? "");
            return;
        }
        if (data !== undefined && data !== null) {
            console.log(text, data);
        } else {
            console.log(text);
        }
    };

    return {
        /**
         * Enable/disable debug logging for info-level messages.
         */
        setDebugEnabled(enabled) {
            debugEnabled = Boolean(enabled);
        },

        /**
         * Return active debug flag.
         */
        isDebugEnabled() {
            return Boolean(debugEnabled);
        },

        /**
         * Log info message. Printed to console only when debug is enabled,
         * unless `forceConsole` option is set.
         */
        info(message, data, opts = {}) {
            const entry = pushEntry("info", message, data);
            if (debugEnabled || Boolean(opts.forceConsole)) {
                consoleWrite("info", entry.message, data);
            }
            return entry;
        },

        /**
         * Log warning message and always print to console.
         */
        warn(message, data) {
            const entry = pushEntry("warn", message, data);
            consoleWrite("warn", entry.message, data);
            return entry;
        },

        /**
         * Log error message and always print to console.
         */
        error(message, data) {
            const entry = pushEntry("error", message, data);
            consoleWrite("error", entry.message, data);
            return entry;
        },

        /**
         * Return latest diagnostics entries from memory.
         */
        getEntries(limit = maxEntries) {
            const count = Math.max(0, Math.min(maxEntries, Number(limit || maxEntries)));
            if (count <= 0) {
                return [];
            }
            return entries.slice(-count);
        },

        /**
         * Clear buffered diagnostics entries.
         */
        clear() {
            entries.length = 0;
        },
    };
}
