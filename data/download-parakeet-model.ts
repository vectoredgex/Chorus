/**
 * Downloads NVIDIA Parakeet TDT 0.6B v2 (int8, sherpa-onnx format), the
 * offline English speech engine Chorus uses by default.
 *
 * Model folder: models/parakeet/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/
 * Source: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
 * Licence: CC-BY-4.0 (NVIDIA)
 *
 * Run: bun run download:parakeet
 */

import { join } from "node:path"
import { existsSync, mkdirSync, createWriteStream, renameSync, rmSync } from "node:fs"
import { spawnSync } from "node:child_process"

const PROJECT_ROOT = join(import.meta.dir, "..")
const MODELS_DIR = join(PROJECT_ROOT, "models", "parakeet")
const MODEL_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
const MODEL_DIR = join(MODELS_DIR, MODEL_NAME)
const FILES = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
const ARCHIVE_URL = `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_NAME}.tar.bz2`

async function download(url: string, dest: string) {
  const response = await fetch(url, { redirect: "follow" })
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status} ${response.statusText}`)
  }
  const totalBytes = Number(response.headers.get("content-length") ?? 0)
  const totalMB = (totalBytes / 1_000_000).toFixed(0)
  console.log(`Size: ${totalMB} MB`)
  const writer = createWriteStream(dest + ".tmp")
  const reader = response.body?.getReader()
  if (!reader) throw new Error("No response body")
  let downloaded = 0
  let lastPercent = -1
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
  writer.end()
  await new Promise<void>((resolve, reject) => {
    writer.on("finish", resolve)
    writer.on("error", reject)
  })
  renameSync(dest + ".tmp", dest)
  console.log("")
}

async function main() {
  if (FILES.every((f) => existsSync(join(MODEL_DIR, f)))) {
    console.log(`Parakeet model already exists: ${MODEL_DIR}`)
    return
  }
  mkdirSync(MODELS_DIR, { recursive: true })
  const archive = join(MODELS_DIR, `${MODEL_NAME}.tar.bz2`)
  console.log(`Downloading Parakeet model from ${ARCHIVE_URL}`)
  await download(ARCHIVE_URL, archive)

  // tar ships with Windows 10+, macOS and Linux and reads .tar.bz2.
  console.log("Unpacking...")
  const result = spawnSync("tar", ["-xjf", archive, "-C", MODELS_DIR], { stdio: "inherit" })
  if (result.status !== 0) {
    throw new Error(`Could not unpack ${archive} (tar exit code ${result.status})`)
  }
  rmSync(archive, { force: true })

  const missing = FILES.filter((f) => !existsSync(join(MODEL_DIR, f)))
  if (missing.length > 0) {
    throw new Error(`Model folder is missing: ${missing.join(", ")}`)
  }
  console.log(`Parakeet model ready: ${MODEL_DIR}`)
}

main().catch((e) => {
  console.error("Failed to download Parakeet model:", e)
  process.exit(1)
})
