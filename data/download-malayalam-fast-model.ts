/**
 * Sets up the quick Malayalam engine for "Local (Malayalam, fast + accurate)":
 * AI4Bharat's official IndicConformer (600M, RNNT), Malayalam only, with an
 * int8 encoder (~0.9 GB). Made by scripts/chorus/ic_official_pack.py from
 * https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
 * (MIT licence) and published as a Chorus release file.
 *
 * Model folder: models/malayalam/indicconformer-official-ml/
 *
 * Run: bun run download:malayalam-fast
 */

import { join } from "node:path"
import {
  existsSync,
  mkdirSync,
  createWriteStream,
  renameSync,
  statSync,
  readdirSync,
  rmSync,
} from "node:fs"
import { spawnSync } from "node:child_process"

const PROJECT_ROOT = join(import.meta.dir, "..")
const MODEL_DIR = join(PROJECT_ROOT, "models", "malayalam", "indicconformer-official-ml")
const TAR_URL =
  "https://github.com/vectoredgex/Chorus/releases/download/models-ml-fast-v1/indicconformer-official-ml.tar"

// Downloads with resume: a dropped connection (common on home Wi-Fi for a
// ~480 MB file) picks up where it stopped instead of starting again.
async function download(url: string, dest: string) {
  const tmp = dest + ".tmp"
  const MAX_ATTEMPTS = 20
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const have = existsSync(tmp) ? statSync(tmp).size : 0
    try {
      const done = await downloadFrom(url, tmp, have)
      if (done) {
        renameSync(tmp, dest)
        console.log("")
        return
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      console.log(`\n  Connection dropped (${msg}). Resuming (try ${attempt + 1}/${MAX_ATTEMPTS})...`)
      await new Promise((r) => setTimeout(r, Math.min(2000 * attempt, 15000)))
    }
  }
  throw new Error(`Download kept failing after ${MAX_ATTEMPTS} tries. Run the same command again to continue.`)
}

// Returns true once the whole file is on disk.
async function downloadFrom(url: string, tmp: string, have: number): Promise<boolean> {
  const headers: Record<string, string> = have > 0 ? { Range: `bytes=${have}-` } : {}
  const response = await fetch(url, { redirect: "follow", headers })
  if (response.status === 416) return true // already complete
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status} ${response.statusText}`)
  }
  const resumed = response.status === 206
  const start = resumed ? have : 0
  const totalBytes = start + Number(response.headers.get("content-length") ?? 0)
  const totalMB = (totalBytes / 1_000_000).toFixed(0)
  if (start === 0) console.log(`Size: ${totalMB} MB`)
  const writer = createWriteStream(tmp, { flags: resumed ? "a" : "w" })
  const finished = new Promise<void>((resolve, reject) => {
    writer.on("finish", resolve)
    writer.on("error", reject)
  })
  const reader = response.body?.getReader()
  if (!reader) throw new Error("No response body")
  let downloaded = start
  let lastPercent = -1
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      writer.write(Buffer.from(value))
      downloaded += value.byteLength
      const percent = totalBytes > 0 ? Math.floor((downloaded / totalBytes) * 100) : 0
      if (percent !== lastPercent && percent % 5 === 0) {
        process.stdout.write(`\r  ${percent}% (${(downloaded / 1_000_000).toFixed(0)}/${totalMB} MB)`)
        lastPercent = percent
      }
    }
  } finally {
    writer.end()
    await finished
  }
  return totalBytes === 0 || downloaded >= totalBytes
}

const FILES = [
  "preprocessor.onnx",
  "encoder.onnx",
  "rnnt_decoder.onnx",
  "joint_enc.onnx",
  "joint_pred.onnx",
  "joint_pre_net.onnx",
  "joint_post_net_ml.onnx",
  "tokens.json",
]

function modelReady(): boolean {
  if (!existsSync(MODEL_DIR)) return false
  const names = readdirSync(MODEL_DIR)
  return FILES.every((f) => names.includes(f))
}

// Move every file found under `dir` (any depth) into MODEL_DIR.
function flattenInto(dir: string) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) flattenInto(path)
    else renameSync(path, join(MODEL_DIR, entry.name))
  }
}

async function main() {
  if (modelReady()) {
    console.log(`Already set up: ${MODEL_DIR}`)
  } else {
    mkdirSync(MODEL_DIR, { recursive: true })
    const tar = join(MODEL_DIR, "..", "indicconformer-official-ml.tar")
    if (!existsSync(tar)) {
      console.log("Downloading the official IndicConformer (about 0.9 GB)...")
      await download(TAR_URL, tar)
    }
    console.log("Unpacking...")
    const unpack = join(MODEL_DIR, "..", "indicconformer-official-unpack")
    rmSync(unpack, { recursive: true, force: true })
    mkdirSync(unpack, { recursive: true })
    const res = spawnSync("tar", ["-xf", tar, "-C", unpack], { stdio: "inherit" })
    if (res.status !== 0) throw new Error("Could not unpack the model (is `tar` available?)")
    flattenInto(unpack)
    rmSync(unpack, { recursive: true, force: true })
    if (!modelReady()) throw new Error(`Unpacked, but the model files are missing in ${MODEL_DIR}`)
    rmSync(tar, { force: true })
    console.log(`Official IndicConformer ready: ${MODEL_DIR}`)
  }

  console.log("\nDone. In Chorus: Settings, choose Local (Malayalam, fast + accurate).")
}

main().catch((e) => {
  console.error("Official IndicConformer setup failed:", e)
  process.exit(1)
})
