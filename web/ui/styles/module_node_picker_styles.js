/**
 * Module: web/ui/styles/module_node_picker_styles.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   CSS source for Module Node Picker UI.
 *
 * Purpose:
 *   Centralizes picker styling in one place and groups rules by UI area so
 *   layout and visual changes can be made without touching orchestration logic.
 */

/**
 * Core container and global layout for the picker panel.
 */
const ROOT_LAYOUT_STYLES = `
    /* Root container and vertical spacing */
    .alexz-mod-picker {
        padding: 10px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        height: 100%;
        overflow: auto;
    }
`;

/**
 * Header row, title, warmup hint, and debug card styling.
 */
const HEADER_AND_DEBUG_STYLES = `
    /* Header row with title on the left and actions on the right */
    .alexz-mod-picker-head {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .alexz-mod-picker-title {
        font-size: 13px;
        font-weight: 700;
        opacity: 0.95;
    }
    .alexz-mod-picker-title-warmup {
        font-size: 10px;
        font-style: italic;
        opacity: 0.68;
        color: var(--input-text, #b3b3b3);
        margin-right: auto;
    }
    .alexz-mod-picker-head-right {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Optional debug diagnostics card */
    .alexz-mod-picker-debug-card {
        border: 1px dashed var(--border-color, #555);
        border-radius: 7px;
        padding: 6px;
        display: none;
        background: var(--comfy-input-bg, rgba(255,255,255,0.02));
    }
    .alexz-mod-picker-debug-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
    }
    .alexz-mod-picker-debug-title {
        font-size: 11px;
        font-weight: 700;
        opacity: 0.92;
    }
    .alexz-mod-picker-diag {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 10px;
        line-height: 1.25;
        opacity: 0.82;
        border: 1px dashed var(--border-color, #555);
        border-radius: 6px;
        padding: 6px;
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 130px;
        overflow: auto;
    }
`;

/**
 * Dividers and block-level composition helpers.
 */
const STRUCTURE_STYLES = `
    /* Section dividers and stacked blocks */
    .alexz-mod-picker-divider {
        border-top: 1px solid var(--border-color, #444);
        margin: 2px 0;
    }
    .alexz-mod-picker-update-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }
    .alexz-mod-picker-update-block {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .alexz-mod-picker-select {
        width: 100%;
    }
    .alexz-mod-picker-module-info-wrap {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
`;

/**
 * Inline help text, legend text, and refresh progress line.
 */
const HELP_AND_HINT_STYLES = `
    /* Help and hint text around module cards and node list */
    .alexz-mod-picker-help {
        display: flex;
        flex-direction: column;
        gap: 3px;
    }
    .alexz-mod-picker-help--module {
        min-height: 2.2em;
        justify-content: flex-end;
    }
    .alexz-mod-picker-help--selection {
        margin-top: 1px;
        margin-bottom: 1px;
    }
    .alexz-mod-picker-help-main {
        font-size: 13px;
        line-height: 1.3;
        opacity: 0.95;
        white-space: normal;
        word-break: break-word;
        overflow-wrap: anywhere;
    }
    .alexz-mod-picker-help-main strong {
        font-weight: 700;
    }
    .alexz-mod-picker-help-hint {
        font-size: 11px;
        line-height: 1.3;
        opacity: 0.78;
        font-style: italic;
        white-space: normal;
        word-break: break-word;
        overflow-wrap: anywhere;
        margin-bottom: 0;
    }
    .alexz-mod-picker-help-hint--warn {
        color: #ff6b6b;
        opacity: 0.95;
    }
    .alexz-mod-picker-help-hint--selection-legend {
        margin-top: 0;
        margin-bottom: 0;
        white-space: pre-line;
    }

    /* One-line refresh status/progress text */
    .alexz-mod-picker-refresh-line {
        font-size: 12px;
        opacity: 0.92;
        min-height: 1.2em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .alexz-mod-picker-refresh-line.alexz-mod-picker-refresh-line--ok {
        color: #3dbb7e;
        font-weight: 700;
    }
    .alexz-mod-picker-refresh-line.alexz-mod-picker-refresh-line--warn {
        color: #ff6b6b;
        font-weight: 700;
    }
`;

/**
 * Status cards for refresh/update results and inline process messages.
 */
const STATUS_CARD_STYLES = `
    /* Reusable status cards (neutral, warning, ok) */
    .alexz-mod-picker-status-card {
        border: 1px solid var(--border-color, #555);
        background: var(--comfy-input-bg, rgba(255,255,255,0.02));
        color: var(--input-text, inherit);
        border-radius: 7px;
        padding: 7px 8px;
        font-size: 12px;
        line-height: 1.3;
        display: block;
    }
    .alexz-mod-picker-status-card.alexz-mod-picker-status-card--warn {
        border-color: #b64040;
        background: rgba(180, 64, 64, 0.16);
        color: #ff6b6b;
        font-weight: 700;
    }
    .alexz-mod-picker-status-card.alexz-mod-picker-status-card--ok {
        border-color: #2e8f61;
        background: rgba(61, 187, 126, 0.16);
        color: #3dbb7e;
        font-weight: 700;
    }
    .alexz-mod-picker-status-card.alexz-mod-picker-status-card--neutral {
        border-color: var(--border-color, #555);
        background: rgba(120, 120, 120, 0.08);
        color: var(--input-text, inherit);
    }
    .alexz-mod-picker-status-card-actions {
        margin-top: 6px;
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }
    .alexz-mod-picker-process-inline {
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px dashed var(--border-color, #555);
        display: none;
    }
`;

/**
 * Module details card with metadata, notes, and action row.
 */
const MODULE_CARD_STYLES = `
    /* Selected module card */
    .alexz-mod-picker-module-card {
        border: 1px solid var(--border-color, #444);
        border-radius: 7px;
        padding: 8px;
        background: var(--comfy-input-bg, rgba(255,255,255,0.03));
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .alexz-mod-picker-module-card--updated {
        border-color: #3dbb7e;
        box-shadow: inset 0 0 0 1px rgba(61, 187, 126, 0.35);
    }
    .alexz-mod-picker-module-card--unknown {
        border-color: #f0b429;
        box-shadow: inset 0 0 0 1px rgba(240, 180, 41, 0.35);
    }
    .alexz-mod-picker-module-card--clickable {
        cursor: pointer;
        transition: filter 0.12s ease;
    }
    .alexz-mod-picker-module-card--clickable:hover {
        filter: brightness(1.08);
    }
    .alexz-mod-picker-module-title {
        font-size: 12px;
        font-weight: 700;
        word-break: break-all;
    }
    .alexz-mod-picker-module-meta {
        font-size: 11px;
        opacity: 0.9;
        word-break: break-all;
    }
    .alexz-mod-picker-module-meta a {
        color: var(--link-color, #87b5ff);
        text-decoration: underline;
    }
    .alexz-mod-picker-module-desc {
        font-size: 11px;
        opacity: 0.85;
        line-height: 1.28em;
        white-space: pre-wrap;
    }
    .alexz-mod-picker-module-note {
        font-size: 10px;
        opacity: 0.8;
        line-height: 1.25em;
        white-space: pre-wrap;
    }
    .alexz-mod-picker-module-note.alexz-mod-picker-module-note--ok {
        color: #3dbb7e;
        font-weight: 700;
    }
    .alexz-mod-picker-module-note.alexz-mod-picker-module-note--warn {
        color: #ff6b6b;
        font-weight: 700;
    }
    .alexz-mod-picker-module-row {
        font-size: 11px;
        opacity: 0.9;
        display: flex;
        gap: 6px;
        align-items: center;
        flex-wrap: wrap;
    }
    .alexz-mod-picker-module-row.notice {
        color: #f0b429;
    }
    .alexz-mod-picker-module-row.warn {
        color: #ff6b6b;
        font-weight: 700;
    }
    .alexz-mod-picker-module-row.ok {
        color: #3dbb7e;
        font-weight: 700;
    }
    .alexz-mod-picker-module-row.unknown {
        color: #f0b429;
        font-weight: 700;
    }
    .alexz-mod-picker-module-label {
        font-weight: 700;
        opacity: 0.95;
    }
`;

/**
 * Generic badges and interactive controls (small buttons/select groups).
 */
const CONTROLS_STYLES = `
    /* Small status badge in module rows */
    .alexz-mod-picker-status {
        display: inline-flex;
        align-items: center;
        border: 1px solid var(--border-color, #555);
        border-radius: 10px;
        padding: 1px 7px;
        font-size: 10px;
        line-height: 1.4;
        font-weight: 700;
    }
    .alexz-mod-picker-status.up-to-date {
        color: #3dbb7e;
    }
    .alexz-mod-picker-status.can-update {
        color: #f0b429;
    }
    .alexz-mod-picker-status.unknown {
        color: #b3b3b3;
    }

    /* Action buttons used in module and status cards */
    .alexz-mod-picker-action-row {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-top: 4px;
    }
    .alexz-mod-picker-btn-small {
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid var(--border-color, #555);
        background: var(--comfy-input-bg, rgba(255,255,255,0.03));
        cursor: pointer;
    }
    .alexz-mod-picker-btn-small:hover {
        filter: brightness(1.08);
    }
    .alexz-mod-picker-btn-small[disabled] {
        opacity: 0.55;
        cursor: default;
        filter: none;
    }
`;

/**
 * Node list container and per-node visual state (new/updated).
 */
const NODE_LIST_STYLES = `
    /* Node list panel */
    .alexz-mod-picker-group {
        border: 1px solid var(--border-color, #444);
        border-radius: 7px;
        padding: 8px;
        margin-top: 2px;
        background: var(--comfy-input-bg, rgba(255,255,255,0.03));
    }
    .alexz-mod-picker-node-legend {
        padding: 0;
        margin-top: 2px;
        margin-bottom: 6px;
        border: none;
        background: transparent;
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .alexz-mod-picker-node-legend-row {
        font-size: 11px;
        line-height: 1.3;
        opacity: 0.78;
        font-style: italic;
        white-space: normal;
        word-break: break-word;
        overflow-wrap: anywhere;
    }
    .alexz-mod-picker-legend-color-red {
        color: #d44f4f;
        font-weight: 700;
    }
    .alexz-mod-picker-legend-color-green {
        color: #3dbb7e;
        font-weight: 700;
    }
    .alexz-mod-picker-legend-color-yellow {
        color: #f0b429;
        font-weight: 700;
    }
    .alexz-mod-picker-group-title {
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 7px;
        word-break: break-all;
    }
    .alexz-mod-picker-node {
        width: 100%;
        text-align: left;
        margin-bottom: 6px;
        border: 1px solid var(--border-color, #555);
        border-radius: 6px;
        background: var(--comfy-input-bg, rgba(255,255,255,0.03));
        padding: 7px;
        cursor: pointer;
    }
    .alexz-mod-picker-node:hover {
        filter: brightness(1.12);
    }
    .alexz-mod-picker-node.alexz-mod-picker-node--updated {
        border-color: #3dbb7e;
        box-shadow: inset 0 0 0 1px rgba(61, 187, 126, 0.35);
    }
    .alexz-mod-picker-node.alexz-mod-picker-node--new {
        border-color: #d44f4f;
        box-shadow: inset 0 0 0 1px rgba(212, 79, 79, 0.35);
    }
    .alexz-mod-picker-node-name {
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 3px;
    }
    .alexz-mod-picker-node-desc {
        font-size: 11px;
        opacity: 0.85;
        line-height: 1.28em;
        word-break: break-all;
    }
`;

/**
 * Fallback floating button used when Sidebar API is unavailable.
 */
const FALLBACK_STYLES = `
    /* Floating launcher shown only in sidebar fallback mode */
    .alexz-mod-picker-floating-btn {
        position: fixed;
        left: 10px;
        bottom: 10px;
        z-index: 10005;
    }
`;

/**
 * Return full CSS text for Module Node Picker.
 */
export function getModuleNodePickerStyleText() {
    return [
        ROOT_LAYOUT_STYLES,
        HEADER_AND_DEBUG_STYLES,
        STRUCTURE_STYLES,
        HELP_AND_HINT_STYLES,
        STATUS_CARD_STYLES,
        MODULE_CARD_STYLES,
        CONTROLS_STYLES,
        NODE_LIST_STYLES,
        FALLBACK_STYLES,
    ].join("\n");
}
