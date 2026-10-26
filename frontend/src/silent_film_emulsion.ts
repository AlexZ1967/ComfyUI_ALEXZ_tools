import { app } from "../../../scripts/app.js";

type NodeId = string | number;
interface Options { default?: unknown; min?: number; max?: number; forceInput?: boolean; control_after_generate?: boolean }
interface Schema {
    name: string;
    input: { required: Record<string, [string | string[], Options?]> };
    input_order?: { required?: string[] };
}
interface Field { name: string; type: string | string[]; options: Options; widget: boolean }
interface Input { name: string; type?: string; link?: NodeId | null; widget?: { name: string }; [key: string]: unknown }
interface Output { name?: string; type?: string; slot_index?: number; links?: NodeId[] | null }
interface SerializedNode {
    id: NodeId; type: string; inputs?: Input[]; outputs?: Output[]; properties?: Record<string, unknown>;
    widgets_values?: unknown[]; widgets_values_named?: Record<string, unknown>;
}
type Link = [NodeId, NodeId, number, NodeId, number, string]
    | { id: NodeId; target_id: NodeId; target_slot: number };
interface Graph { nodes: SerializedNode[]; links?: Link[] }
interface NodeInstance { widgets?: { name: string; value: unknown; options?: { serialize?: boolean } }[] }
interface NodeType { prototype: { onSerialize?: (this: NodeInstance, data: SerializedNode) => unknown } }

const MODEL = "alexz.neutral_bw_halation_v2";
const PRE_HALATION_MODEL = "alexz.neutral_bw_v1";
const PROJECTION_MODEL = "alexz.projection_glow_v1";
const LEGACY_WIDGETS: Record<string, string[]> = {
    VideoSilentFilmNegative: ["exposure_ev", "development_contrast", "base_fog", "density_range", "development_variation", "grain_strength", "grain_size", "seed", "control_after_generate", "compute_device", "batch_size"],
    VideoSilentFilmPrint: ["print_exposure_ev", "print_contrast", "reference_density", "base_fog", "density_range", "print_softness", "tone_mode", "grain_strength", "grain_size", "seed", "control_after_generate", "compute_device", "batch_size"],
    VideoSilentFilmProjector: ["projection_fps", "gate_weave_px", "softness", "vignette_strength", "edge_softness", "edge_start", "edge_end", "lamp_flicker", "lamp_breathing", "seed", "control_after_generate", "compute_device", "batch_size"],
};
const PRE_HALATION_WIDGETS: Record<string, string[]> = {
    VideoSilentFilmNegative: ["input_transfer", "exposure_ev", "development_contrast", "shadow_softness", "highlight_softness", "fog_strength", "development_variation", "grain_strength", "grain_size", "seed", "control_after_generate", "compute_device", "batch_size"],
    VideoSilentFilmPrint: ["output_transfer", "print_exposure_ev", "print_contrast", "shadow_softness", "highlight_softness", "fog_strength", "print_softness", "tone_mode", "grain_strength", "grain_size", "seed", "control_after_generate", "compute_device", "batch_size"],
};
const schemas = new Map<string, Schema>();

/** Read the current backend contract in the same order as ComfyUI widgets. */
function fields(schema: Schema): Field[] {
    return (schema.input_order?.required ?? Object.keys(schema.input.required)).map((name) => {
        const [type, options = {}] = schema.input.required[name];
        return { name, type, options, widget: type !== "IMAGE" && !options.forceInput };
    });
}

/** Preserve a compatible saved value or use the current backend default. */
function widgetValue(field: Field, saved: Record<string, unknown>): unknown {
    const value = saved[field.name];
    const { type, options } = field;
    if (Array.isArray(type)) return typeof value === "string" && type.includes(value) ? value : options.default ?? type[0];
    if (type === "FLOAT" || type === "INT") {
        if (typeof value === "number" && Number.isFinite(value)
            && value >= (options.min ?? -Infinity) && value <= (options.max ?? Infinity)
            && (type !== "INT" || Number.isInteger(value))) return value;
    } else if (typeof value === "string") return value;
    return options.default;
}

/** Migrate old widget positions and retarget connected inputs by name. */
function migrateNode(node: SerializedNode, graph: Graph, schema: Schema): void {
    const projector = node.type === "VideoSilentFilmProjector";
    const modelKey = projector ? "alexz_projection_model" : "alexz_emulsion_model";
    const model = projector ? PROJECTION_MODEL : MODEL;
    const beforeHalation = !projector && node.properties?.[modelKey] !== model
        && (node.properties?.[modelKey] === PRE_HALATION_MODEL
            || (!Object.hasOwn(node.widgets_values_named ?? {}, "halation_strength")
                && node.widgets_values?.length === PRE_HALATION_WIDGETS[node.type].length));
    const oldFields = beforeHalation ? PRE_HALATION_WIDGETS[node.type] : LEGACY_WIDGETS[node.type];
    const legacy = (!projector && (node.inputs?.some((input) => ["density_range", "reference_density", "base_fog"].includes(input.name))
        || Object.hasOwn(node.widgets_values_named ?? {}, "density_range")
    )) || beforeHalation || (node.properties?.[modelKey] !== model && Array.isArray(node.widgets_values) && node.widgets_values.length === oldFields.length);
    if (!legacy) return;
    const saved = { ...Object.fromEntries(oldFields.map((name, index) => [name, node.widgets_values?.[index]])), ...node.widgets_values_named };
    const currentFields = fields(schema);
    const values: Record<string, unknown> = {};
    node.widgets_values = [];
    for (const field of currentFields) {
        if (!field.widget) continue;
        values[field.name] = widgetValue(field, saved);
        node.widgets_values.push(values[field.name]);
        if (field.options.control_after_generate) {
            values.control_after_generate = saved.control_after_generate ?? "fixed";
            node.widgets_values.push(values.control_after_generate);
        }
    }
    node.widgets_values_named = values;
    const previous = node.inputs ?? [];
    const inputFields = [currentFields[0], { name: "film_json", type: "STRING", widget: false } satisfies Partial<Field>, ...currentFields.slice(1)];
    const inputs = inputFields.map((field) => {
        const old = previous.find((input) => input.name === field.name);
        const input = { ...old, name: field.name, type: Array.isArray(field.type) ? "COMBO" : field.type, link: old?.link ?? null };
        if (field.widget) input.widget = { ...old?.widget, name: field.name };
        return input;
    });
    node.inputs = inputs;
    const dropped = new Set<NodeId>();
    graph.links = (graph.links ?? []).filter((link) => {
        const array = Array.isArray(link);
        if (String(array ? link[3] : link.target_id) !== String(node.id)) return true;
        const oldSlot = array ? link[4] : link.target_slot;
        const name = previous[oldSlot]?.name;
        const slot = inputs.findIndex((input) => input.name === name);
        if (slot < 0) {
            dropped.add(array ? link[0] : link.id);
            return false;
        }
        if (array) link[4] = slot;
        else link.target_slot = slot;
        return true;
    });
    if (dropped.size) {
        for (const source of graph.nodes) for (const output of source.outputs ?? []) {
            if (Array.isArray(output.links)) output.links = output.links.filter((id) => !dropped.has(id));
        }
        console.warn(`[ALEXZ_tools] ${node.type}: отключены связи удалённых параметров материала; проверьте workflow.`);
    }
    if (node.type === "VideoSilentFilmNegative" && node.outputs?.length === 3) {
        node.outputs.push({ name: "preview_positive", type: "IMAGE", slot_index: 3, links: null });
    }
    node.properties = { ...node.properties, [modelKey]: model };
    console.info(projector
        ? "[ALEXZ_tools] Projector: сохранены параметры и связи, добавлен выключенный glow. softness остаётся расфокусом."
        : beforeHalation
            ? `[ALEXZ_tools] ${node.type}: сохранены параметры и связи, добавлен выключенный halation.`
            : `[ALEXZ_tools] ${node.type}: параметры перенесены в нейтральную модель. Проверьте экспозицию и контраст; старая калибровка плотности заменена эталоном.`);
}

app.registerExtension({
    name: "ALEXZ.Tools.SilentFilmEmulsion",
    beforeRegisterNodeDef(nodeTypeValue: unknown, nodeDataValue: unknown) {
        const nodeType = nodeTypeValue as NodeType;
        const nodeData = nodeDataValue as Schema;
        if (!Object.hasOwn(LEGACY_WIDGETS, nodeData.name)) return;
        // Старый работающий backend должен продолжать получать прежние параметры.
        const projector = nodeData.name === "VideoSilentFilmProjector";
        if (projector ? !nodeData.input?.required?.glow_strength
            : !nodeData.input?.required?.halation_strength) return;
        schemas.set(nodeData.name, nodeData);
        const original = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
            const result = original?.call(this, data);
            data.properties = { ...data.properties, [projector ? "alexz_projection_model" : "alexz_emulsion_model"]: projector ? PROJECTION_MODEL : MODEL };
            data.widgets_values_named = Object.fromEntries((this.widgets ?? [])
                .filter((widget) => widget.options?.serialize !== false)
                .map((widget) => [widget.name, widget.value]));
            return result;
        };
    },
    beforeConfigureGraph(value: unknown) {
        if (!value || typeof value !== "object" || !Array.isArray((value as Graph).nodes)) return;
        const data = value as Graph;
        for (const node of data.nodes) {
            const schema = schemas.get(node.type);
            if (schema) migrateNode(node, data, schema);
        }
    },
});
