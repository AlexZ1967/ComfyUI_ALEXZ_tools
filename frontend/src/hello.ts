import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "ALEXZ.Tools.Hello",

    async setup() {
        console.log("[ALEXZ_tools] TypeScript frontend extension loaded.");
    },
});
