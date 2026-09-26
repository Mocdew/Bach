// Regenerate the API contract from the Python side:
//   recoup/api/schemas.py -> openapi.json -> src/api/schema.d.ts
// Needs the repo's Python env (pip install -e ..[api]). PYTHON overrides the interpreter.
import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repo = resolve(web, "..");
const python = process.env.PYTHON ?? "python";

const schema = execFileSync(python, ["-m", "recoup.api.openapi"], { cwd: repo, encoding: "utf8" });
writeFileSync(resolve(web, "openapi.json"), schema);
execFileSync(process.execPath, [resolve(web, "node_modules/openapi-typescript/bin/cli.js"),
  "openapi.json", "-o", "src/api/schema.d.ts"], { cwd: web, stdio: "inherit" });
