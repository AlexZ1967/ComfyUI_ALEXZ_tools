/**
 * Module: web/show_json.js
 * Author: AlexZ1967
 * Last updated: 2026-02-10
 *
 * Description:
 *   Show/Save JSON node frontend helper.
 *
 * Purpose:
 *   Adds readonly JSON preview behavior for the node in the ComfyUI web interface.
 */

import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const TARGET_NODES = new Set(["Show/Save JSON", "JsonDisplayAndSave"]);

app.registerExtension({
    name: "ALEXZ.Tools.ShowJson",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!TARGET_NODES.has(nodeData.name)) {
            return;
        }

        /**
         * Lazily create and cache read-only multiline widget used for JSON preview.
         */
        function ensureWidget() {
            if (this._jsonWidget) {
                return this._jsonWidget;
            }

            const widget = ComfyWidgets["STRING"](
                this,
                "json_preview",
                ["STRING", { multiline: true }],
                app
            ).widget;

            widget.inputEl.readOnly = true;
            widget.inputEl.style.opacity = 0.6;
            widget.inputEl.style.fontFamily = "monospace";
            widget.serialize = false;

            this._jsonWidget = widget;
            return widget;
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const widget = ensureWidget.call(this);
            const payload = message?.text ?? "";
            const text = Array.isArray(payload) ? payload.join("\n") : String(payload);
            widget.value = text;
        };
    },
});
