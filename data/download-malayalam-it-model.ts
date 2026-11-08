/**
 * Sets up the Indic-Transcribe Malayalam speech engine for Chorus:
 *  1. Downloads Bodhan AI Indic-Transcribe (int8 ONNX for the CPU, ~1.7 GB)
 *     and unpacks it.
 *  2. Adds the Malayalam Bible to rhema.db if it is not there yet.
 *
 * Model folder: models/malayalam/indic-transcribe-core-int8/
 * Source: https://huggingface.co/Mvkd2k06/indic-transcribe-core-onnx
 *         (ONNX export of bodhan-ai/indic-transcribe-core)
 * Licence: Indic Open Model License v1.0 (free to self-host; see
 *          Bodhan_AI_Open_Model_License.md in the model folder)
 *
 * Run: bun run download:malayalam-it
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
const MODEL_DIR = join(PROJECT_ROOT, "models", "malayalam", "indic-transcribe-core-int8")
const TAR_URL =
  "https://huggingface.co/Mvkd2k06/indic-transcribe-core-onnx/resolve/main/variants/int8/indic-transcribe-core-int8.tar"

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
  if (process.env.HF_TOKEN) headers.Authorization = `Bearer ${process.env.HF_TOKEN}`
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

function modelReady(): boolean {
  if (!existsSync(MODEL_DIR)) return false
  const names = readdirSync(MODEL_DIR)
  return (
    names.some((n) => n.startsWith("encoder") && n.endsWith(".onnx")) &&
    names.some((n) => n.startsWith("decoder") && n.endsWith(".onnx")) &&
    names.includes("tokenizer.json")
  )
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
    const tar = join(MODEL_DIR, "..", "indic-transcribe-core-int8.tar")
    if (!existsSync(tar)) {
      console.log("Downloading Indic-Transcribe (about 1.7 GB, this takes a while)...")
      await download(TAR_URL, tar)
    }
    console.log("Unpacking...")
    const unpack = join(MODEL_DIR, "..", "indic-transcribe-unpack")
    rmSync(unpack, { recursive: true, force: true })
    mkdirSync(unpack, { recursive: true })
    const res = spawnSync("tar", ["-xf", tar, "-C", unpack], { stdio: "inherit" })
    if (res.status !== 0) throw new Error("Could not unpack the model (is `tar` available?)")
    flattenInto(unpack)
    rmSync(unpack, { recursive: true, force: true })
    if (!modelReady()) throw new Error(`Unpacked, but the model files are missing in ${MODEL_DIR}`)
    rmSync(tar, { force: true })
    console.log(`Indic-Transcribe ready: ${MODEL_DIR}`)
  }

  console.log("\nAdding the Malayalam Bible (skipped if already there)...")
  const bible = spawnSync("bun", ["run", join(import.meta.dir, "download-malayalam-bible.ts")], {
    stdio: "inherit",
  })
  if (bible.status !== 0) {
    throw new Error("Could not add the Malayalam Bible (see the message above)")
  }
  console.log("\nDone. In Chorus: Settings, choose Local (Malayalam, Indic-Transcribe).")
}

main().catch((e) => {
  console.error("Indic-Transcribe setup failed:", e)
  process.exit(1)
})
