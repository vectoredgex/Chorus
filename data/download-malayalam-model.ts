/**
 * Sets up Malayalam for Chorus in one go:
 *  1. Downloads AI4Bharat IndicConformer 600M (Malayalam, int8, sherpa-onnx
 *     format), the offline Malayalam speech engine (~660 MB).
 *  2. Adds the Malayalam Bible (Sathyavedapusthakam 1910) to rhema.db.
 *
 * Model folder: models/malayalam/indicconformer-600m-ml-int8/
 * Source: https://huggingface.co/Docglasses-AI/indic-conformer-600m-sherpa-onnx-int8
 *         (conversion of ai4bharat/indic-conformer-600m-multilingual)
 * Licence: MIT (AI4Bharat)
 *
 * Run: bun run download:malayalam
 */

import { join } from "node:path"
import {
  existsSync,
  mkdirSync,
  createWriteStream,
  renameSync,
  statSync,
} from "node:fs"
import { spawnSync } from "node:child_process"

const PROJECT_ROOT = join(import.meta.dir, "..")
const MODEL_DIR = join(PROJECT_ROOT, "models", "malayalam", "indicconformer-600m-ml-int8")
const REPO = "https://huggingface.co/Docglasses-AI/indic-conformer-600m-sherpa-onnx-int8/resolve/main"
const FILES = ["indicconformer-ml.int8.onnx", "indicconformer-tokens.txt"]

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

async function main() {
  mkdirSync(MODEL_DIR, { recursive: true })
  for (const f of FILES) {
    const dest = join(MODEL_DIR, f)
    if (existsSync(dest)) {
      console.log(`Already downloaded: ${f}`)
      continue
    }
    console.log(`Downloading ${f}...`)
    await download(`${REPO}/${f}`, dest)
  }
  console.log(`Malayalam speech model ready: ${MODEL_DIR}`)

  console.log("\nAdding the Malayalam Bible...")
  const bible = spawnSync("bun", ["run", join(import.meta.dir, "download-malayalam-bible.ts")], {
    stdio: "inherit",
  })
  if (bible.status !== 0) {
    throw new Error("Could not add the Malayalam Bible (see the message above)")
  }
  console.log("\nMalayalam is ready. In Chorus: Settings, choose Malayalam.")
}

main().catch((e) => {
  console.error("Malayalam setup failed:", e)
  process.exit(1)
})
