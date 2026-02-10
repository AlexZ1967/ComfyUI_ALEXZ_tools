/**
 * Match Video Cut Point frontend upload helper.
 *
 * Adds UI support for selecting the second video input file required by the
 * VideoCutMatch node workflow.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXT_NAME = "ALEXZ.Tools.VideoCutMatchUpload";
const TARGET_NODES = new Set(["VideoCutMatch", "Match Video Cut Point"]);
const VIDEO_ACCEPT = ["video/webm", "video/mp4", "video/x-matroska", "image/gif"];

/** Handle `ensureOption` workflow step. */
function ensureOption(widget, filename) {
    if (!widget?.options || !Array.isArray(widget.options.values)) {
        return;
    }
    if (!widget.options.values.includes(filename)) {
        widget.options.values.push(filename);
    }
}

/** Handle `setWidgetValue` workflow step. */
function setWidgetValue(widget, filename) {
    if (!widget) {
        return;
    }
    ensureOption(widget, filename);
    widget.value = filename;
    widget.callback?.(filename);
}

/** Handle `uploadVideoFile` workflow step. */
async function uploadVideoFile(file) {
    const body = new FormData();
    const newFile = new File([file], file.name, {
        type: file.type,
        lastModified: file.lastModified,
    });
    body.append("image", newFile);
    const response = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
    });
    if (!response.ok) {
        throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
    }
    const payload = await response.json();
    const name = payload?.name;
    const subfolder = payload?.subfolder || "";
    if (!name) {
        throw new Error("Upload response has no filename.");
    }
    return `${subfolder}${name}`;
}

/** Handle `addUploadWidget` workflow step. */
function addUploadWidget(node, widgetName, buttonLabel) {
    if (!node?.widgets?.length) {
        return;
    }
    const pathWidget = node.widgets.find((w) => w.name === widgetName);
    if (!pathWidget) {
        return;
    }
    const marker = `upload_${widgetName}`;
    if (node.widgets.find((w) => w.name === marker)) {
        return;
    }

    const fileInput = document.createElement("input");
    Object.assign(fileInput, {
        type: "file",
        accept: VIDEO_ACCEPT.join(","),
        style: "display: none",
    });

    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        fileInput.remove();
        onRemoved?.apply(this, arguments);
    };

    fileInput.onchange = async () => {
        if (!fileInput.files?.length) {
            return;
        }
        try {
            node.progress = 0;
            const filename = await uploadVideoFile(fileInput.files[0]);
            const widgetA = node.widgets.find((w) => w.name === "video_a");
            const widgetB = node.widgets.find((w) => w.name === "video_b");
            ensureOption(widgetA, filename);
            ensureOption(widgetB, filename);
            setWidgetValue(pathWidget, filename);
            node.graph?.setDirtyCanvas?.(true, true);
        } catch (err) {
            alert(String(err));
        } finally {
            node.progress = undefined;
            fileInput.value = "";
        }
    };

    document.body.append(fileInput);
    const uploadWidget = node.addWidget("button", buttonLabel, "image", () => {
        app.canvas.node_widget = null;
        fileInput.click();
    });
    uploadWidget.options.serialize = false;
    uploadWidget.name = marker;
}

app.registerExtension({
    name: EXT_NAME,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODES.has(nodeData.name)) {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            addUploadWidget(this, "video_a", "choose video_a to upload");
            addUploadWidget(this, "video_b", "choose video_b to upload");
        };
    },
});

