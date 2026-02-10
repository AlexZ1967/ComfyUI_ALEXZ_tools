/**
 * Module: web/ui/module_node_picker_process.js
 * Author: AlexZ1967
 * Last updated: 2026-02-10
 *
 * Description:
 *   Inline process/progress controller for Module Node Picker status cards.
 *
 * Purpose:
 *   Centralizes mount/display logic for process line and action buttons.
 */

/**
 * Create a process UI controller used by top status cards.
 */
export function createProcessUiController(context) {
    const processHost = context?.processHost;
    const refreshLine = context?.refreshLine;
    const processActions = context?.processActions;
    const comfyAlert = context?.comfyAlert;
    const customAlert = context?.customAlert;
    const diagnosticsLogger = context?.diagnosticsLogger;
    const defaultTarget = typeof context?.defaultTarget === "function"
        ? context.defaultTarget
        : () => "custom";

    let processTarget = "";

    const normalizeTarget = (target) => {
        const normalized = String(target || "").trim().toLowerCase();
        if (normalized === "comfy" || normalized === "custom") {
            return normalized;
        }
        return "";
    };

    const setTarget = (target) => {
        if (!processHost) {
            return "";
        }
        const normalized = normalizeTarget(target);
        processTarget = normalized;
        const parent = processHost.parentElement;
        if (parent) {
            parent.removeChild(processHost);
        }
        if (normalized === "comfy" && comfyAlert) {
            comfyAlert.appendChild(processHost);
            comfyAlert.style.display = "block";
        } else if (normalized === "custom" && customAlert) {
            customAlert.appendChild(processHost);
            customAlert.style.display = "block";
        }
        return processTarget;
    };

    const ensureMounted = () => {
        if (!processHost?.parentElement) {
            setTarget(processTarget || defaultTarget());
        }
    };

    const setAction = (label, btnText, onClick, disabled = false) => {
        if (!processHost || !processActions || !refreshLine) {
            return;
        }
        processActions.innerHTML = "";
        if (!label) {
            if (!refreshLine.textContent) {
                processHost.style.display = "none";
            }
            return;
        }
        ensureMounted();
        processHost.style.display = "";
        const labelEl = document.createElement("div");
        labelEl.textContent = String(label);
        processActions.appendChild(labelEl);
        if (!btnText || typeof onClick !== "function") {
            return;
        }
        const actionBtn = document.createElement("button");
        actionBtn.type = "button";
        actionBtn.className = "alexz-mod-picker-btn-small";
        actionBtn.textContent = String(btnText);
        actionBtn.disabled = Boolean(disabled);
        actionBtn.onclick = onClick;
        processActions.appendChild(actionBtn);
    };

    const setLine = (text, tone = "neutral") => {
        if (!processHost || !processActions || !refreshLine) {
            return;
        }
        const value = String(text || "");
        refreshLine.textContent = value;
        refreshLine.classList.remove("alexz-mod-picker-refresh-line--ok", "alexz-mod-picker-refresh-line--warn");
        if (!value) {
            if (!processActions.children.length) {
                processHost.style.display = "none";
            }
            return;
        }
        ensureMounted();
        processHost.style.display = "";
        if (tone === "ok") {
            refreshLine.classList.add("alexz-mod-picker-refresh-line--ok");
        } else if (tone === "warn") {
            refreshLine.classList.add("alexz-mod-picker-refresh-line--warn");
        }
        diagnosticsLogger?.info?.(value, null, { forceConsole: true });
    };

    const setButtonsDisabled = (disabled) => {
        if (!processActions) {
            return;
        }
        for (const btn of processActions.querySelectorAll(".alexz-mod-picker-btn-small")) {
            btn.disabled = Boolean(disabled);
        }
    };

    return {
        setTarget,
        getTarget: () => processTarget,
        setAction,
        setLine,
        setButtonsDisabled,
    };
}
