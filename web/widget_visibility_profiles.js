/**
 * Module: web/widget_visibility_profiles.js
 * Author: AlexZ1967
 * Last updated: 2026-02-18
 *
 * Description:
 *   Reusable conditional widget-visibility profiles for ComfyUI nodes.
 *
 * Purpose:
 *   Hide/show optional widgets based on a controller widget value
 *   (e.g. show only LUT params when seam_model=v4_lut).
 */

import { app } from "../../../scripts/app.js";

const EXT_NAME = "ALEXZ.Tools.WidgetVisibilityProfiles";
const HIDDEN_STATE_KEY = "__alexz_hidden_state";
const WRAP_STATE_KEY = "__alexz_visibility_callback_wrapped";
const HIDDEN_INPUTS_KEY = "__alexz_hidden_inputs_by_widget";

/**
 * Add new reusable profiles here.
 */
const VISIBILITY_PROFILES = [
    {
        id: "seam_match_model_params",
        targetNodes: ["ImageSeamMatchToReference", "Seam Match To Reference"],
        controllerWidget: "seam_model",
        groups: {
            hybrid: ["hybrid_residual_strength", "hybrid_residual_reg", "hybrid_coherence_reg"],
            lut: ["lut_size", "lut_identity_reg", "lut_smooth_reg", "lut_lr_scale"],
        },
        modes: {
            v1_affine: { showGroups: [] },
            v2_tonal: { showGroups: [] },
            v3_hybrid: { showGroups: ["hybrid"] },
            v4_lut: { showGroups: ["lut"] },
        },
    },
];

function findWidget(node, name) {
    return node?.widgets?.find((w) => w?.name === name) || null;
}

function collectManagedWidgetNames(profile) {
    const names = new Set();
    Object.values(profile.groups || {}).forEach((arr) => {
        (arr || []).forEach((name) => names.add(name));
    });
    return names;
}

function resolveVisibleWidgetNames(profile, modeValue) {
    const mode = profile.modes?.[String(modeValue)];
    if (!mode) {
        // Unknown mode: keep all managed widgets visible for safety.
        return collectManagedWidgetNames(profile);
    }
    const visible = new Set();
    (mode.showGroups || []).forEach((groupName) => {
        (profile.groups?.[groupName] || []).forEach((widgetName) => visible.add(widgetName));
    });
    return visible;
}

function setWidgetVisible(widget, visible) {
    if (!widget) {
        return;
    }

    if (visible) {
        const state = widget[HIDDEN_STATE_KEY];
        if (!state) {
            return;
        }
        widget.type = state.type;
        widget.computeSize = state.computeSize;
        widget.serialize = state.serialize;
        delete widget[HIDDEN_STATE_KEY];
        return;
    }

    if (widget[HIDDEN_STATE_KEY]) {
        return;
    }

    widget[HIDDEN_STATE_KEY] = {
        type: widget.type,
        computeSize: widget.computeSize,
        serialize: widget.serialize,
    };
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
}

function findInputIndexByWidgetName(node, widgetName) {
    if (!node?.inputs?.length) {
        return -1;
    }
    return node.inputs.findIndex((slot) => {
        if (!slot) {
            return false;
        }
        if (slot.widget?.name === widgetName) {
            return true;
        }
        return slot.name === widgetName;
    });
}

function cloneSlotShallow(slot) {
    if (!slot || typeof slot !== "object") {
        return null;
    }
    const out = {};
    for (const [key, value] of Object.entries(slot)) {
        if (key === "link") {
            continue;
        }
        if (typeof value === "function") {
            continue;
        }
        if (Array.isArray(value)) {
            out[key] = [...value];
        } else if (value && typeof value === "object") {
            out[key] = { ...value };
        } else {
            out[key] = value;
        }
    }
    return out;
}

function ensureHiddenInputsStore(node) {
    if (!node[HIDDEN_INPUTS_KEY]) {
        node[HIDDEN_INPUTS_KEY] = {};
    }
    return node[HIDDEN_INPUTS_KEY];
}

function hideInputSlotForWidget(node, widgetName) {
    const idx = findInputIndexByWidgetName(node, widgetName);
    if (idx < 0) {
        return;
    }
    const slot = node.inputs[idx];
    // Do not silently drop active links. Keep visible connector if linked.
    if (slot?.link != null) {
        return;
    }
    const store = ensureHiddenInputsStore(node);
    if (!store[widgetName]) {
        store[widgetName] = {
            index: idx,
            slot: cloneSlotShallow(slot),
        };
    }
    node.removeInput(idx);
}

function showInputSlotForWidget(node, widgetName) {
    if (findInputIndexByWidgetName(node, widgetName) >= 0) {
        return;
    }
    const store = ensureHiddenInputsStore(node);
    const hidden = store[widgetName];
    if (!hidden?.slot) {
        return;
    }
    if (!Array.isArray(node.inputs)) {
        node.inputs = [];
    }
    const insertIndex = Math.max(0, Math.min(hidden.index, node.inputs.length));
    node.inputs.splice(insertIndex, 0, { ...hidden.slot, link: null });
    delete store[widgetName];
}

function applyProfileVisibility(node, profile) {
    const controller = findWidget(node, profile.controllerWidget);
    if (!controller) {
        return;
    }

    const visibleNames = resolveVisibleWidgetNames(profile, controller.value);
    const managedNames = collectManagedWidgetNames(profile);

    managedNames.forEach((name) => {
        const widget = findWidget(node, name);
        const visible = visibleNames.has(name);
        setWidgetVisible(widget, visible);
        if (visible) {
            showInputSlotForWidget(node, name);
        } else {
            hideInputSlotForWidget(node, name);
        }
    });

    node.setSize?.(node.computeSize());
    node.graph?.setDirtyCanvas?.(true, true);
}

function attachProfileToNode(node, profile) {
    if (!node) {
        return;
    }
    if (!node.__alexzVisibilityProfiles) {
        node.__alexzVisibilityProfiles = {};
    }
    if (node.__alexzVisibilityProfiles[profile.id]) {
        applyProfileVisibility(node, profile);
        return;
    }

    node.__alexzVisibilityProfiles[profile.id] = true;
    const controller = findWidget(node, profile.controllerWidget);
    if (controller && !controller[WRAP_STATE_KEY]) {
        const originalCallback = controller.callback;
        controller.callback = function () {
            originalCallback?.apply(this, arguments);
            applyProfileVisibility(node, profile);
        };
        controller[WRAP_STATE_KEY] = true;
    }

    applyProfileVisibility(node, profile);
}

app.registerExtension({
    name: EXT_NAME,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const matched = VISIBILITY_PROFILES.filter((profile) => profile.targetNodes.includes(nodeData.name));
        if (!matched.length) {
            return;
        }

        const patchKey = "__alexzVisibilityPatched";
        if (nodeType.prototype[patchKey]) {
            return;
        }
        nodeType.prototype[patchKey] = true;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            matched.forEach((profile) => attachProfileToNode(this, profile));
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            matched.forEach((profile) => attachProfileToNode(this, profile));
            return result;
        };
    },
});
