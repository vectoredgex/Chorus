/**
 * Sets up the quick Malayalam engine for "Local (Malayalam, fast + accurate)":
 * AI4Bharat's official IndicConformer (600M, RNNT), Malayalam only, with an
 * int8 encoder (~0.9 GB). Made by scripts/chorus/ic_official_pack.py from
 * https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
 * (MIT licence) and kept in a PRIVATE Hugging Face repo, <you>/chorus-ml-fast,
 * so it needs your Hugging Face token:
 *
 *   PowerShell:  $env:HF_TOKEN="hf_..."; bun run download:malayalam-fast
 *
 * Model folder: models/malayalam/indicconformer-official-ml/
 */

import { join } from "node:path"
import {
  existsSync,
  mkdirSync,
  createWriteStream,
  renameSync,
  statSync,
  readdirSync,
} from "node:fs"

const PROJECT_ROOT = join(import.meta.dir, "..")
const MODEL_DIR = join(PROJECT_ROOT, "models", "malayalam", "indicconformer-official-ml")
const REPO_NAME = "chorus-ml-fast"

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
  headers.Authorization = `Bearer ${process.env.HF_TOKEN}`
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

async function main() {
  if (modelReady()) {
    console.log(`Already set up: ${MODEL_DIR}`)
    return
  }
  const token = process.env.HF_TOKEN
  if (!token) {
    throw new Error(
      'This needs your Hugging Face token. In PowerShell: $env:HF_TOKEN="hf_..."; bun run download:malayalam-fast',
    )
  }
  const who = await fetch("https://huggingface.co/api/whoami-v2", {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!who.ok) throw new Error(`Hugging Face did not accept the token (${who.status})`)
  const user = ((await who.json()) as { name: string }).name
  const base = `https://huggingface.co/${user}/${REPO_NAME}/resolve/main`
  mkdirSync(MODEL_DIR, { recursive: true })
  console.log(`Downloading the official IndicConformer from ${user}/${REPO_NAME} (about 0.9 GB)...`)
  for (const f of [...FILES, "NOTICE.txt"]) {
    const dest = join(MODEL_DIR, f)
    if (existsSync(dest)) continue
    console.log(`  ${f}`)
    await download(`${base}/${f}`, dest)
  }
  if (!modelReady()) throw new Error(`Downloaded, but the model files are missing in ${MODEL_DIR}`)
  console.log(`\nDone: ${MODEL_DIR}`)
  console.log("In Chorus: Settings, choose Local (Malayalam, fast + accurate). It uses this engine for quick guesses.")
}

main().catch((e) => {
  console.error("Official IndicConformer setup failed:", e)
  process.exit(1)
})
