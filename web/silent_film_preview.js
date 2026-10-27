import { app } from "../../../scripts/app.js";

const FILM_STAGES = new Set(["VideoSilentFilmCamera", "VideoSilentFilmNegative", "VideoSilentFilmPrint", "VideoSilentFilmProjector"]);
const FILM_DEFECTS = new Set(["VideoSilentFilmCaptureDirt", "VideoSilentFilmNegativeDevelopment", "VideoSilentFilmNegativeDamage", "VideoSilentFilmPrintingDirt", "VideoSilentFilmPositiveDevelopment", "VideoSilentFilmPositiveDamage", "VideoSilentFilmProjectorDirt"]);
const redirected = new WeakSet();
let helperAvailable = false;

/** Resolve the dedicated preview output without following bypass substitutions. */
function isFilmPreview(node) {
    if (!node.graph) return false;
    let link = node.getInputLink?.(0);
    const seen = new Set();
    while (link) {
        const source = node.graph?.getNodeById(link.origin_id);
        if (!source || seen.has(source.id)) return false;
        seen.add(source.id);
        const sourceType = source.comfyClass ?? source.type;
        if (FILM_STAGES.has(sourceType) || FILM_DEFECTS.has(sourceType)) {
            const outputName = source.outputs?.[link.origin_slot]?.name;
            if (FILM_DEFECTS.has(sourceType)) return ["preview", "preview_positive", "mask_preview"].includes(outputName);
            return outputName === "preview" || (sourceType === "VideoSilentFilmNegative" && outputName === "preview_positive");
        }
        if (source.type !== "Reroute") return false;
        link = source.getInputLink?.(0);
    }
    return false;
}

/** Select the compatible single-frame preview backend only for Silent Film preview links. */
function updatePreview(node) {
    if (helperAvailable && isFilmPreview(node) && (node.comfyClass === "PreviewImage" || (redirected.has(node) && node.comfyClass === "VideoPreviewFrame"))) {
        node.comfyClass = "VideoPreviewFrame";
        redirected.add(node);
    } else if (redirected.has(node)) {
        if (node.comfyClass === "VideoPreviewFrame") node.comfyClass = "PreviewImage";
        redirected.delete(node);
    }
}

app.registerExtension({
    name: "ALEXZ.Tools.SilentFilmPreview",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "VideoPreviewFrame") helperAvailable = true;
        if (nodeData.name !== "PreviewImage") return;
        for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                updatePreview(this);
                return result;
            };
        }
    },
    afterConfigureGraph() {
        for (const node of app.graph?._nodes ?? []) if (node.type === "PreviewImage") updatePreview(node);
    },
});
