// frontend/src/silent_film_emulsion.ts
import { app } from "../../../scripts/app.js";
var MODEL = "alexz.neutral_bw_halation_v2";
var PRE_HALATION_MODEL = "alexz.neutral_bw_v1";
var PROJECTION_MODEL = "alexz.projection_glow_v1";
var LEGACY_WIDGETS = {
  VideoSilentFilmNegative: ["exposure_ev", "development_contrast", "base_fog", "density_range", "development_variation", "grain_strength", "grain_size", "seed", "control_after_generate", "compute_device", "batch_size"],
  VideoSilentFilmPrint: ["print_exposure_ev", "print_contrast", "reference_density", "base_fog", "density_range", "print_softness", "tone_mode", "grain_strength", "grain_size", "seed", "control_after_generate", "compute_device", "batch_size"],
  VideoSilentFilmProjector: ["projection_fps", "gate_weave_px", "softness", "vignette_strength", "edge_softness", "edge_start", "edge_end", "lamp_flicker", "lamp_breathing", "seed", "control_after_generate", "compute_device", "batch_size"]
};
var PRE_HALATION_WIDGETS = {
  VideoSilentFilmNegative: ["input_transfer", "exposure_ev", "development_contrast", "shadow_softness", "highlight_softness", "fog_strength", "development_variation", "grain_strength", "grain_size", "seed", "control_after_generate", "compute_device", "batch_size"],
  VideoSilentFilmPrint: ["output_transfer", "print_exposure_ev", "print_contrast", "shadow_softness", "highlight_softness", "fog_strength", "print_softness", "tone_mode", "grain_strength", "grain_size", "seed", "control_after_generate", "compute_device", "batch_size"]
};
var schemas = /* @__PURE__ */ new Map();
function fields(schema) {
  return (schema.input_order?.required ?? Object.keys(schema.input.required)).map((name) => {
    const [type, options = {}] = schema.input.required[name];
    return { name, type, options, widget: type !== "IMAGE" && !options.forceInput };
  });
}
function widgetValue(field, saved) {
  const value = saved[field.name];
  const { type, options } = field;
  if (Array.isArray(type)) return typeof value === "string" && type.includes(value) ? value : options.default ?? type[0];
  if (type === "FLOAT" || type === "INT") {
    if (typeof value === "number" && Number.isFinite(value) && value >= (options.min ?? -Infinity) && value <= (options.max ?? Infinity) && (type !== "INT" || Number.isInteger(value))) return value;
  } else if (typeof value === "string") return value;
  return options.default;
}
function migrateNode(node, graph, schema) {
  const projector = node.type === "VideoSilentFilmProjector";
  const modelKey = projector ? "alexz_projection_model" : "alexz_emulsion_model";
  const model = projector ? PROJECTION_MODEL : MODEL;
  const beforeHalation = !projector && node.properties?.[modelKey] !== model && (node.properties?.[modelKey] === PRE_HALATION_MODEL || !Object.hasOwn(node.widgets_values_named ?? {}, "halation_strength") && node.widgets_values?.length === PRE_HALATION_WIDGETS[node.type].length);
  const oldFields = beforeHalation ? PRE_HALATION_WIDGETS[node.type] : LEGACY_WIDGETS[node.type];
  const legacy = !projector && (node.inputs?.some((input) => ["density_range", "reference_density", "base_fog"].includes(input.name)) || Object.hasOwn(node.widgets_values_named ?? {}, "density_range")) || beforeHalation || node.properties?.[modelKey] !== model && Array.isArray(node.widgets_values) && node.widgets_values.length === oldFields.length;
  if (!legacy) return;
  const saved = { ...Object.fromEntries(oldFields.map((name, index) => [name, node.widgets_values?.[index]])), ...node.widgets_values_named };
  const currentFields = fields(schema);
  const values = {};
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
  const inputFields = [currentFields[0], { name: "film_json", type: "STRING", widget: false }, ...currentFields.slice(1)];
  const inputs = inputFields.map((field) => {
    const old = previous.find((input2) => input2.name === field.name);
    const input = { ...old, name: field.name, type: Array.isArray(field.type) ? "COMBO" : field.type, link: old?.link ?? null };
    if (field.widget) input.widget = { ...old?.widget, name: field.name };
    return input;
  });
  node.inputs = inputs;
  const dropped = /* @__PURE__ */ new Set();
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
    console.warn(`[ALEXZ_tools] ${node.type}: \u043E\u0442\u043A\u043B\u044E\u0447\u0435\u043D\u044B \u0441\u0432\u044F\u0437\u0438 \u0443\u0434\u0430\u043B\u0451\u043D\u043D\u044B\u0445 \u043F\u0430\u0440\u0430\u043C\u0435\u0442\u0440\u043E\u0432 \u043C\u0430\u0442\u0435\u0440\u0438\u0430\u043B\u0430; \u043F\u0440\u043E\u0432\u0435\u0440\u044C\u0442\u0435 workflow.`);
  }
  if (node.type === "VideoSilentFilmNegative" && node.outputs?.length === 3) {
    node.outputs.push({ name: "preview_positive", type: "IMAGE", slot_index: 3, links: null });
  }
  node.properties = { ...node.properties, [modelKey]: model };
  console.info(projector ? "[ALEXZ_tools] Projector: \u0441\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u044B \u043F\u0430\u0440\u0430\u043C\u0435\u0442\u0440\u044B \u0438 \u0441\u0432\u044F\u0437\u0438, \u0434\u043E\u0431\u0430\u0432\u043B\u0435\u043D \u0432\u044B\u043A\u043B\u044E\u0447\u0435\u043D\u043D\u044B\u0439 glow. softness \u043E\u0441\u0442\u0430\u0451\u0442\u0441\u044F \u0440\u0430\u0441\u0444\u043E\u043A\u0443\u0441\u043E\u043C." : beforeHalation ? `[ALEXZ_tools] ${node.type}: \u0441\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u044B \u043F\u0430\u0440\u0430\u043C\u0435\u0442\u0440\u044B \u0438 \u0441\u0432\u044F\u0437\u0438, \u0434\u043E\u0431\u0430\u0432\u043B\u0435\u043D \u0432\u044B\u043A\u043B\u044E\u0447\u0435\u043D\u043D\u044B\u0439 halation.` : `[ALEXZ_tools] ${node.type}: \u043F\u0430\u0440\u0430\u043C\u0435\u0442\u0440\u044B \u043F\u0435\u0440\u0435\u043D\u0435\u0441\u0435\u043D\u044B \u0432 \u043D\u0435\u0439\u0442\u0440\u0430\u043B\u044C\u043D\u0443\u044E \u043C\u043E\u0434\u0435\u043B\u044C. \u041F\u0440\u043E\u0432\u0435\u0440\u044C\u0442\u0435 \u044D\u043A\u0441\u043F\u043E\u0437\u0438\u0446\u0438\u044E \u0438 \u043A\u043E\u043D\u0442\u0440\u0430\u0441\u0442; \u0441\u0442\u0430\u0440\u0430\u044F \u043A\u0430\u043B\u0438\u0431\u0440\u043E\u0432\u043A\u0430 \u043F\u043B\u043E\u0442\u043D\u043E\u0441\u0442\u0438 \u0437\u0430\u043C\u0435\u043D\u0435\u043D\u0430 \u044D\u0442\u0430\u043B\u043E\u043D\u043E\u043C.`);
}
app.registerExtension({
  name: "ALEXZ.Tools.SilentFilmEmulsion",
  beforeRegisterNodeDef(nodeTypeValue, nodeDataValue) {
    const nodeType = nodeTypeValue;
    const nodeData = nodeDataValue;
    if (!Object.hasOwn(LEGACY_WIDGETS, nodeData.name)) return;
    const projector = nodeData.name === "VideoSilentFilmProjector";
    if (projector ? !nodeData.input?.required?.glow_strength : !nodeData.input?.required?.halation_strength) return;
    schemas.set(nodeData.name, nodeData);
    const original = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function(data) {
      const result = original?.call(this, data);
      data.properties = { ...data.properties, [projector ? "alexz_projection_model" : "alexz_emulsion_model"]: projector ? PROJECTION_MODEL : MODEL };
      data.widgets_values_named = Object.fromEntries((this.widgets ?? []).filter((widget) => widget.options?.serialize !== false).map((widget) => [widget.name, widget.value]));
      return result;
    };
  },
  beforeConfigureGraph(value) {
    if (!value || typeof value !== "object" || !Array.isArray(value.nodes)) return;
    const data = value;
    for (const node of data.nodes) {
      const schema = schemas.get(node.type);
      if (schema) migrateNode(node, data, schema);
    }
  }
});
//# sourceMappingURL=silent_film_emulsion.js.map
