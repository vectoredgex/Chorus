/**
 * Proper Malayalam test on a real sermon.
 *
 *   bun run test:sermon <YouTube link | audio file> [--old] [--minutes N]
 *
 * Downloads the sermon's audio (YouTube links need yt-dlp, fetched once into
 * test-audio/), then plays it through the real Chorus pipeline (speech
 * detection, the Malayalam engine, the verse finder) at normal speed, with
 * no speakers or microphone. Writes a report per engine to test-audio/.
 *
 * --old       also run the old Malayalam engine (IndicConformer) afterwards
 * --minutes N only play the first N minutes
 *
 * Everything stays in test-audio/, which git ignores: the audio never goes
 * into the repo. Only the text reports are meant to be shared.
 */
import { spawn, spawnSync } from "child_process"
import { chmodSync, createWriteStream, existsSync, mkdirSync, readdirSync, unlinkSync } from "fs"
import { cpus } from "os"
import { join, resolve } from "path"

const ROOT = resolve(import.meta.dir, "..")
const OUT = join(ROOT, "test-audio")
const WIN = process.platform === "win32"
mkdirSync(OUT, { recursive: true })

const args = process.argv.slice(2)
const runOld = args.includes("--old")
const mi = args.indexOf("--minutes")
const minutes = mi >= 0 ? args[mi + 1] : undefined
const source = args.find((a, i) => !a.startsWith("--") && args[i - 1] !== "--minutes")
if (!source) {
  console.log("Usage: bun run test:sermon <YouTube link | audio file> [--old] [--minutes N]")
  process.exit(1)
}

const ENGINES = [
  { name: "new", label: "Indic-Transcribe (new)", dir: join(ROOT, "models", "malayalam", "indic-transcribe-core-int8"), download: "bun run download:malayalam-it" },
  { name: "old", label: "IndicConformer (old)", dir: join(ROOT, "models", "malayalam", "indicconformer-600m-ml-int8"), download: "bun run download:malayalam" },
].filter((e) => e.name === "new" || runOld)

for (const e of ENGINES) {
  if (!existsSync(e.dir)) {
    console.error(`The ${e.label} model is missing. Run first: ${e.download}`)
    process.exit(1)
  }
}

async function ytDlp(): Promise<string> {
  const exe = join(OUT, WIN ? "yt-dlp.exe" : "yt-dlp")
  if (existsSync(exe)) return exe
  const asset = WIN ? "yt-dlp.exe" : process.platform === "darwin" ? "yt-dlp_macos" : "yt-dlp_linux"
  console.log("Downloading yt-dlp (once)...")
  const res = await fetch(`https://github.com/yt-dlp/yt-dlp/releases/latest/download/${asset}`)
  if (!res.ok) throw new Error(`yt-dlp download failed: ${res.status}`)
  await Bun.write(exe, res)
  if (!WIN) chmodSync(exe, 0o755)
  return exe
}

let audio = source
if (/^https?:\/\//.test(source)) {
  const exe = await ytDlp()
  for (const f of readdirSync(OUT)) if (f.startsWith("sermon.")) unlinkSync(join(OUT, f))
  console.log("Downloading the sermon audio...")
  const r = spawnSync(exe, ["-f", "bestaudio[ext=m4a]/bestaudio[ext=mp3]", "--no-playlist", "-o", join(OUT, "sermon.%(ext)s"), source], { stdio: "inherit" })
  if (r.status !== 0) {
    console.error("Could not download that link. Try another video, or download the audio yourself and pass the file.")
    process.exit(1)
  }
  const got = readdirSync(OUT).find((f) => f.startsWith("sermon.") && !f.endsWith(".part"))
  if (!got) throw new Error("download produced no file")
  audio = join(OUT, got)
}
if (!existsSync(audio)) {
  console.error(`File not found: ${audio}`)
  process.exit(1)
}

console.log("Building the test program (first time takes a few minutes)...")
const build = spawnSync("cargo", ["build", "--release", "-p", "chorus-replay"], { cwd: join(ROOT, "src-tauri"), stdio: "inherit" })
if (build.status !== 0) process.exit(build.status ?? 1)
const replay = join(ROOT, "src-tauri", "target", "release", WIN ? "chorus-replay.exe" : "chorus-replay")

const threads = String(cpus().length)
for (const e of ENGINES) {
  const report = join(OUT, `report-${e.name}.txt`)
  console.log(`\n=== ${e.label}: playing the sermon at normal speed (takes as long as the sermon) ===`)
  console.log(`Report: ${report}\n`)
  const file = createWriteStream(report)
  file.write(`engine: ${e.label}\nsource: ${source}\nminutes: ${minutes ?? "all"}\nthreads: ${threads}\n`)
  const env = { ...process.env, ...(minutes ? { CHORUS_REPLAY_MINUTES: minutes } : {}) }
  const code: number = await new Promise((done) => {
    const p = spawn(replay, [e.dir, threads, audio, "-"], { env })
    p.stdout.on("data", (d) => { process.stdout.write(d); file.write(d) })
    p.stderr.on("data", (d) => file.write(d))
    p.on("close", (c) => done(c ?? 1))
  })
  file.end()
  if (code !== 0) console.error(`${e.label} stopped with an error (code ${code}); see ${report}`)
}

console.log(`\nDone. Send these report files to Claude:`)
for (const e of ENGINES) console.log(`  ${join(OUT, `report-${e.name}.txt`)}`)
