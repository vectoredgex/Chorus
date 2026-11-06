# Chorus — Handover for the next AI (or developer)

Read this whole file before touching anything. It is written so you can pick up
the work immediately, without asking the owner questions.

Last updated: 2026-11-07, 02:25 Asia/Qatar (UTC+3).

---

## 1. What Chorus is

Chorus is a desktop app for churches. It **listens to the preacher through a
microphone, recognises the Bible references they say ("John three sixteen",
"turn to Romans chapter eight verse twenty-eight"), and puts the verse on the
projector / livestream automatically**.

- It is a fork of **Rhema** by OpenBezal (<https://github.com/openbezal/rhema>,
  MIT licence). The MIT copyright notice must stay in `LICENSE`; the app's
  Help section credits "Rhema by OpenBezal".
- Repository: **<https://github.com/vectoredgex/Chorus>** (note capital C; the
  old URL `vectoredgex/chorus` redirects). It is a **public** fork.
- Stack: **Tauri v2** desktop app. Frontend React + TypeScript + Vite (Bun as
  package manager). Backend **Rust** workspace in `src-tauri/`.
- Speech-to-text: **Whisper large-v3-turbo (q8_0, 874 MB)** run locally through
  `whisper-rs 0.16` / whisper.cpp. Deepgram (cloud) also exists in the code
  but the owner does **not** want it for now (offline only).

### Roadmap the owner agreed to
1. **Phase 1 — English verses, as accurate and fast as possible** (in progress).
2. **Phase 2 — Malayalam**: speech engine chosen by testing (Google/Azure/AI4Bharat
   were discussed but owner wants offline, so test Whisper Malayalam and
   AI4Bharat IndicConformer), Malayalam script + Manglish book names
   (Yohannan, Ulpathi, Sankeerthanangal, Mathai, Romar, Apostolapravarthikal …)
   and number words (onnu, randu, moonnu … adhyayam = chapter, vakyam = verse),
   Malayalam Bible text (check licence; godlytalias/Bible-Database is GPL-3 and
   from Wordproject), bilingual display.
3. **Phase 3 — Songs**: setlist + follow-along lyrics (not song identification;
   discussed as too unreliable live).

---

## 2. The owner (how to work with them)

- Not a programmer. Write to them in **plain, simple English**, short
  sentences, no jargon. They type fast with typos; read intent, not spelling.
- They want it **"maxed out / the best ever"**, and they want you to **work
  autonomously**: "I should get everything finished and final later when I
  check." Do not stop to ask questions unless something is irreversible or
  needs a decision only they can make.
- They often interrupt mid-tool-call to ask questions. When they do, answer
  the question directly first.
- Be honest about numbers. Never claim "100% accurate". They respond well to
  measured results (before/after tables).
- **Privacy decision already made:** their own voice recording must **not**
  go into the public GitHub repo. (Encrypting it needs a GitHub Actions
  secret, which cannot be created from the Claude cloud session because the
  proxy blocks `/actions/secrets`. So it was dropped; do not ask them again
  unless they bring it up.)
- They are in **Qatar (UTC+3)**.

### Their hardware
- Windows laptop, **Intel CPU + NVIDIA GeForce MX550** (2 GB, Turing, sm_75).
- Built-in "Microphone Array (Intel Smart Sound Technology)" — the weakest
  link for accuracy. A sound-desk feed or USB mic would help most.
- Repo cloned at **`C:\Users\georg\Documents\Chorus`**. Bun, Rust, LLVM, CMake,
  Visual Studio (C++ tools, "18 Community") and the **CUDA Toolkit** are
  installed. Setup (`bun run setup:all`) was done: Bible DB (7 translations)
  and the Whisper model in `models/whisper/`.
- App log file: **`C:\Users\georg\AppData\Local\com.vectoredgex.chorus\logs\rhema.log`**.
- Their test recording: `C:\Users\georg\Documents\Sound Recordings\Recording (3).m4a`
  (143 s, them reading the 20-phrase list in section 9). Keep it local only.

### Commands the owner runs (PowerShell)
```
cd ~\Documents\Chorus
git pull
bun run dev:gpu          # NVIDIA build (first build 20–40 min)
bun run tauri dev        # CPU build
```
Known Windows traps (tell the owner if relevant):
- **Clicking inside PowerShell freezes the program** (QuickEdit mode) until a
  key is pressed. This caused fake "Not Responding" / "stuck at 615/616".
  Fix: right-click title bar → Properties → untick QuickEdit.
- If `git pull` refuses because `src-tauri/Cargo.toml` or `Cargo.lock` is
  "modified": `git checkout -- src-tauri/Cargo.toml src-tauri/Cargo.lock`, then
  pull. (Caused by line endings / Cargo rewriting the lock; `.gitattributes`
  now forces LF for `*.toml` and `*.lock`, and CI checks the lock.)

---

## 3. Branches

| Branch | Purpose |
|---|---|
| `main` | Untouched upstream Rhema. Do not push here. |
| `chorus/phase1` | Detector + rebrand + early CPU speed work. Historical. |
| `chorus/whisper-partials` | Intermediate Whisper work. Historical. |
| **`chorus/gpu`** | **What the owner currently pulls and runs.** Only fast-forward it to tested work. |
| **`chorus/wip`** | **Active work.** Push here; when CI is green, fast-forward `chorus/gpu` to it. |

Rule: **never push untested changes to `chorus/gpu`.** Workflow:
`chorus/wip` → CI green → `git checkout chorus/gpu && git merge --ff-only chorus/wip && git push`.

Commit trailer required on every commit:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018gD4MFkA72rx7CA2JZPpb8
```

---

## 4. Code map (what lives where)

### Backend workspace `src-tauri/`
- `Cargo.toml` — workspace. Members: `.`, `crates/audio`, `crates/stt`,
  `crates/bible`, `crates/detection`, `crates/broadcast`, `crates/api`,
  `crates/notes`, **`crates/chorus-refparse`**, **`crates/replay`**.
  Features on the app: `whisper` (default), **`cuda`**, **`vulkan`**.
  `[profile.dev.package.whisper-rs-sys]` and `whisper-rs` are `opt-level = 3`
  so Whisper is fast even in `tauri dev`.
- `src/commands/stt.rs` — starts audio + STT, consumes transcript events,
  runs detection on partials and finals.
  - `PARTIAL_CONFIRMER` (static Mutex<PartialConfirmer>): references from
    partial transcripts are only emitted when the next partial agrees.
  - After each direct detection it calls
    `rhema_stt::whisper::set_context_hint(&verse_ref)` so Whisper's prompt
    knows the current passage.
  - Whisper threads = half of logical CPUs.
- `src/commands/detection.rs` — DetectionResult, merger → UI.

### `crates/chorus-refparse` (NEW, Chorus's own reference engine)
- Pure Rust, **no dependencies** (so it can be tested offline anywhere).
- `src/books.rs` — 66 books, chapter counts, ambiguity flags, alias table
  (abbreviations + speech-to-text mishearings), weak aliases, ordinal prefixes.
- `src/numbers.rs` — spoken numbers ("one hundred and nineteen", "twenty
  third", "23rd", "8a").
- `src/lib.rs` — `Detector` with: written citations, spoken numbers,
  chapter/verse words, ranges ("through", "to", "and" for consecutive),
  numbered books, one-chapter books, "of" forms ("the third chapter of John",
  "verse 16 of John 3"), "the twenty third psalm", context follow-ups ("verse
  17", "the next verse", "chapter 4 verse 5"), split chapter digits ("Psalm 1
  36 verse 1" → 136:1), "Book for N" → chapter 4, **lists** (a 1–3 before a
  numbered book is part of its name: "Revelation 21, 2 Corinthians 13"),
  **fuzzy misheard book names** (`try_fuzzy`, Jaro-Winkler, only when chapter
  AND verse follow; source = `Source::Fuzzy`), `is_song_verse()` ("verse two of
  the hymn" → nothing). Confidence High/Medium/Low.
- `tests/cases.txt` — **the sermon phrase test set** (~240 phrases incl. trap
  phrases and multi-line sermon context). Format documented at top of file.
  **Keep `src-tauri/crates/detection/tests/chorus_cases.txt` an exact copy**
  (`cp` after editing).
- Run offline here: copy the crate to a scratch dir, append `[workspace]` to
  its Cargo.toml, `cargo test --offline`.

### `crates/detection` (Rhema's detector, extended)
- `src/direct/detector.rs` — `DirectDetector::detect()` now runs the Chorus
  engine first (`chorus_hits`): complete, versification-valid verse-level
  references win (confidence 1.0 for High, 0.85 Medium, **0.75 for Fuzzy** so
  it stays below the 0.80 auto-queue threshold → operator confirms).
  Otherwise falls through to Rhema's original `detect_base()` (held
  chapter-only references, streaming completion, corrections, bare-colon
  context, "previous verse" commands, Rhema's own fuzzy matcher).
  Partials of the same utterance are snapshotted (`is_same_utterance`) so
  "the next verse" doesn't advance twice.
  Song-verse check runs first.
- `src/partial_confirm.rs` (NEW) — `PartialConfirmer` shared by app and replay.
- `tests/chorus_benchmark.rs` (NEW) — runs `DirectDetector` over
  `chorus_cases.txt`. Chapter-only expectations also pass when nothing is
  emitted (Rhema holds them waiting for a verse). `CHORUS_BENCH_MIN=0.99` in CI.
- Note: Rhema's **bare-colon context rule** is intentional: after "Exodus
  14:14", a bare "8:28" means Exodus 8:28. Put trap phrases in their own
  blank-line-separated block in the cases file.

### `crates/stt` (speech-to-text)
- `src/whisper.rs` — **most Chorus speed/accuracy work is here**:
  - Speech detection (`rhema_audio::Vad`, energy-based) now fed **fixed 20 ms
    frames** (`VAD_FRAME_SAMPLES = 320`); phrase ends after **0.7 s** silence,
    min voice 120 ms, 300 ms pre-buffer. (Windows delivers ~10 ms chunks; the
    old 12-frame setting meant 0.12 s and chopped phrases at every breath.)
  - Short pieces 0.5–1 s ("John" + pause) are kept and joined to the next
    piece if it starts within 2.5 s; < 0.5 s dropped.
  - **Live partials**: every 1.5 s of new audio during speech, the utterance
    so far is transcribed and sent as `TranscriptEvent::Partial`; stale
    partials are skipped if newer audio is queued. Finals never skipped.
  - `audio_ctx_for()` — encoder window sized to the clip: frames = clip×50
    + 256, min **512**, max 1500. (256 min made Whisper invent words on a real
    voice. Full 1500 window costs ~21 s/phrase on the owner's CPU and ~2.5 s
    on the MX550.) `encoder_window()` uses it for CPU and GPU.
  - `BIBLE_PROMPT` + `set_context_hint()` + previous 20 words of the last
    final (dropped after 20 s) → `current_prompt(previous)`.
  - `level_volume()` (uncommitted at time of writing → see §8) — raises quiet
    phrases so the 99.5th-percentile level reaches 0.5, max 8×.
  - `cleanup::remove_repeats()` (`src/cleanup.rs`) — removes Whisper loops
    of ≥3 words ("Romans 8:28, Romans 8:28, …"), keeps "holy, holy, holy".
  - `GPU_BUILD` = cfg(cuda|vulkan): `use_gpu(true)` + `flash_attn(true)`.
  - `Cargo.toml` features: `cuda = ["whisper","whisper-rs/cuda"]`,
    `vulkan = ["whisper","whisper-rs/vulkan"]`. Example `whisper_bench`.
- `examples/whisper_bench.rs` — times/transcribes clips with full window,
  sized window + prompt, full window + prompt.

### `crates/replay` (NEW) — end-to-end test
- `chorus-replay <model.bin> <threads> <recording.wav> <expected.txt>`:
  feeds a 16 kHz mono WAV in 10 ms chunks **in real time** through
  `WhisperProvider` (the real pipeline) and `DirectDetector` +
  `PartialConfirmer`, prints every transcript and verse with timestamps, then
  `OK / MISSED / WRONG` and a score line `==== CHORUS REPLAY: …`.
  `CHORUS_REPLAY_MIN` env makes it fail below a rate.
- `scripts/chorus/make-sermon.py` — builds a 21-line test sermon with
  pauses (Piper neural voice `en_US-ryan-high` if available, else espeak-ng),
  plus an echo/noise version, and `expected.txt`.

### Frontend `src/`
- Branding changed to Chorus (`transport-bar.tsx`, `settings-dialog.tsx`,
  NDI names, tutorial text, `index.html`, `broadcast-output.html`).
- App identifier `com.vectoredgex.chorus` (`src-tauri/tauri.conf.json`).
- Internal crate names stay `rhema-*` so upstream fixes merge easily.

### Scripts
- `package.json`: `dev:gpu` = `CUDAARCHS=native CUDAFLAGS=-allow-unsupported-compiler tauri dev --features cuda`; `dev:vulkan`.

---

## 5. CI (GitHub Actions) — your main test bench

The cloud session **cannot compile the app locally** (no crates.io access;
only dependency-free crates build with `--offline`). Use GitHub Actions:
push to `chorus/**`, then read results with `gh api`.

| Workflow | What it does |
|---|---|
| `chorus-tests.yml` | Lock file up to date (`cargo metadata --locked`); chorus-refparse tests; **detector benchmark** (fails < 99%); detection unit tests (219 upstream tests); full app `cargo check --all-targets --locked` on Linux; frontend typecheck + vitest. |
| `chorus-whisper-speed.yml` | Synthesises clips (espeak, clean + noisy), downloads the real model, runs `whisper_bench`. CPU only, slow (~10 min). |
| `chorus-gpu-build.yml` | Compiles Whisper with CUDA (in `nvidia/cuda:12.8.0-devel-ubuntu24.04`, CUDAARCHS=75) and Vulkan. Build-only (runners have no GPU). |
| `chorus-replay.yml` | Builds the test sermon (natural + robot voice), runs `chorus-replay` on clean and noisy audio. ~40–60 min. |
| `chorus-inspect.yml` | Prints dependency source APIs and any Cargo.lock diff into the log (dev tool). |
| `build-release.yml` | Upstream release workflow (tags `v*`). Not yet adapted to Chorus. |

### Reading CI from the cloud session
```
gh api "repos/vectoredgex/Chorus/actions/runs?branch=chorus/wip&per_page=10" \
  --jq '.workflow_runs[] | "\(.id) \(.name) | \(.status) \(.conclusion) \(.head_sha[0:7])"'
gh api repos/vectoredgex/Chorus/actions/runs/<RUN>/jobs --jq '.jobs[] | "\(.id) \(.name) \(.conclusion)"'
gh api repos/vectoredgex/Chorus/actions/jobs/<JOB>/logs | sed 's/^[0-9TZ:.-]* //'
```
- Log download sometimes returns "ServerBusy / Egress is over the account
  limit" — just retry after ~30 s.
- `POST .../runs/<id>/rerun-failed-jobs` works.
- **Blocked by the proxy:** `/actions/secrets`, `/actions/permissions`,
  GraphQL, `user/repos`. `gh api` must use repo-scoped REST paths.
- **Pushing a commit cancels in-progress runs of the same workflow on that
  branch** (concurrency groups). Path filters: replay/speed/gpu only run when
  their paths change; `chorus-tests` always runs. Don't push stt/detection
  changes while a replay you need is running.
- WebFetch is allowlisted for github.com pages but not docs.rs; use
  `chorus-inspect.yml` to read crate source instead.
- Waiting: a helper script was used:
  `/tmp/.../scratchpad/waitruns.sh <branch> <sha> <workflow-name-to-skip>`
  (scratchpad is ephemeral — recreate if needed; it polls runs for a commit
  until all non-skipped are completed). Long waits must use
  `run_in_background`, not chained sleeps.

---

## 6. Measured results so far

### Detector (text → verse)
- Rhema original on the Chorus phrase set: **193/210 (91.9%)**.
- Chorus: **231/231 (100%)** at commit b0a1c83 (set has grown since; CI
  enforces ≥ 99%). All 219 upstream detection unit tests pass.

### Whisper speed (owner's laptop)
| Setup | Per phrase |
|---|---|
| Original (CPU, full 30 s window, unoptimised dev build) | 20–26 s |
| CPU, sized window min 256 | ~2.5 s, but accuracy bad on real voice |
| MX550 GPU, full window | ~2.5 s, up to 5–7 s when many tiny pieces queued |
| MX550 GPU, sized window min 512 (on chorus/wip) | expected ~1 s (not yet measured on laptop) |

### Whisper accuracy (synthetic, 16 clips: 8 clear, 8 echo+noise, espeak)
- Full window: ~94%. Sized 512 + prompt: ~88–94% (noise is random per run).
- All clear-speech clips correct in every setting; misses = noise
  ("Genesis"→"Geddes", "Psalm"→"song"). Do **not** map "song N" to Psalm N
  (hymn numbers).

### Owner's live test on GPU (log 22:53–22:55 UTC, chorus/gpu before wip fixes)
- Correct: "John 3 verse 16" (several times), "Revelation chapter 5 … verse
  5" joined → Revelation 5:5.
- Problems found (all addressed on `chorus/wip`): phrases chopped at 0.12 s
  pauses; a partial flashed "John 3:28"; "Revelation 21, 2 Corinthians 13"
  read as 21:2; 2.5–7.6 s per phrase.

---

## 7. Decisions & reasons (don't undo without evidence)
- Chorus engine runs **before** Rhema's detector; Rhema keeps streaming/
  correction logic. Both are tested by the benchmark.
- Fuzzy (misheard) book names never auto-display (0.75 < 0.80 threshold).
- "Delibians" is deliberately **not** guessed (closer to Ephesians than
  Philippians → would show wrong book).
- No noise-suppression filter yet: speech enhancement can hurt Whisper; only
  add it if the noisy benchmark/replay improves.
- Single large model chosen over two-model setups for now. Options discussed:
  "two-model fast" (tiny + large, instant first guess, possible brief wrong
  flash) and "two-model accurate" (two strong models voting, ~2× GPU time).
- Copyright: Rhema's setup downloads NIV/ESV/NKJV/NLT/NASB/AMP from a bundle.
  Fine for testing at the owner's church; **before sharing Chorus or
  publishing installers, ship only free translations** (KJV, WEB, BSB …).
- The repo is public: no owner recordings, keys, or personal data in it.

---

## 8. State at handover & immediate next steps

Uncommitted at time of writing: `level_volume()` in `crates/stt/src/whisper.rs`
(+ unit test). It was verified standalone. Commit it to `chorus/wip`.

Running on GitHub when this was written: **Chorus replay** run 42310342635
(commit 73dcf1d — has time-based VAD, sized GPU window, partial confirmation,
lists fix, previous-sentence prompt; NOT volume levelling). Read its logs
for `CHORUS REPLAY` lines, `MISSED`, `WRONG`.

### Next steps, in order
1. Read replay results; fix what they show (accuracy first). If phrase
   boundaries still cause misses, implement **Silero VAD** (see §10).
2. When `chorus-tests` + `chorus-replay` + GPU builds are green on
   `chorus/wip`, **fast-forward `chorus/gpu`** and push.
3. Update the project doc `claude/chorus-status.md` and send the owner a
   short plain-English summary (SendUserMessage) of what changed and the
   measured numbers. Tell them: `git pull` then `bun run dev:gpu`.
4. Then (each measured before keeping): streaming LocalAgreement mode
   (re-transcribe sliding window every ~0.5 s, commit words two passes agree
   on), noise filter test, operator confidence colours (green auto / amber
   confirm), reading-along tuning (`crates/detection/src/reading_mode.rs`),
   two-model voting if replay shows misses, Windows installer build (CUDA or
   Vulkan variant, free translations only), more realistic voices (Indian
   English if a Piper voice exists).
5. Phase 2: Malayalam (see §1).

### Scheduled automatic check-ins (this session)
- `trig_01NcKDQanRPFtHP67acnFyh6` at 2026-11-07 04:49 UTC (07:49 Qatar).
- `trig_01AApqYEmBNmN3cfbDemkjbc` at 2026-11-07 09:49 UTC (12:49 Qatar, backup).
They resume this session to continue §8 without the owner.

---

## 9. The owner's 20 test phrases (what Recording (3).m4a should contain)
1 John three sixteen · 2 Turn with me to Romans chapter eight, verse twenty-eight ·
3 First Corinthians thirteen, verses four to seven · 4 Let's read Psalm twenty-three ·
5 Look at verse four · 6 Philippians four thirteen · 7 Genesis chapter one verse one ·
8 Ephesians two, eight and nine · 9 Second Timothy three sixteen ·
10 Isaiah forty, verse thirty-one · 11 Matthew chapter six, verse thirty-three ·
12 Jeremiah twenty-nine eleven · 13 Hebrews eleven one · 14 Proverbs three five and six ·
15 Revelation chapter twenty-one, verse four · 16 Mark my words, God is faithful (no verse) ·
17 My job this morning is to encourage you (no verse) · 18 Joshua one nine ·
19 First John one nine · 20 Acts two thirty-eight.
Expected verses: John 3:16, Romans 8:28, 1 Corinthians 13:4-7, Psalms 23:4,
Philippians 4:13, Genesis 1:1, Ephesians 2:8-9, 2 Timothy 3:16, Isaiah 40:31,
Matthew 6:33, Jeremiah 29:11, Hebrews 11:1, Proverbs 3:5-6, Revelation 21:4,
Joshua 1:9, 1 John 1:9, Acts 2:38. (Not verified that the recording follows
the list exactly.)

---

## 10. Silero VAD notes (whisper-rs 0.16, confirmed from source)
- `whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams}`.
- `WhisperVadContext::new(model_path: &str, WhisperVadContextParams)`;
  ctx params: `set_n_threads`, `set_use_gpu`, `set_gpu_device`.
- `ctx.detect_speech(&[f32])`, `ctx.probabilities() -> &[f32]`,
  `ctx.segments_from_probabilities(params)`, `ctx.segments_from_samples(params, &[f32])`.
- `WhisperVadParams`: `set_threshold(f32)`, `set_min_speech_duration(ms)`,
  `set_min_silence_duration(ms)`, `set_max_speech_duration(secs f32)`,
  `set_speech_pad(ms)`, `set_samples_overlap(secs f32)`.
- Segments: `num_segments()`, `get_segment_start_timestamp(i)` /
  `get_segment_end_timestamp(i)` → `Option<f32>` in **centiseconds**.
- Also `FullParams::set_vad_model_path(Some(path))`, `enable_vad(true)`,
  `set_vad_params(..)` (skips non-speech inside a buffer).
- Model: `ggml-silero-v5.1.2.bin` (~0.9 MB) from Hugging Face `ggml-org/whisper-vad`.
- Streaming approach: every ~100 ms run `detect_speech` on the last ~1–2 s,
  use the trailing probabilities to decide speech/silence instead of RMS.
  Add a download step to `data/download-whisper-model.ts` and the replay CI.

---

## 11. Environment facts for an AI in the Claude cloud session
- Workspace: `/home/claude`; repo clone at `/home/claude/chorus` (remote
  `https://github.com/vectoredgex/Chorus.git`; extra fetch refspecs were added
  for `chorus/*` branches because the clone was shallow single-branch).
- Rust (cargo 1.97), Bun, Node 22, Python 3.13, ffmpeg available. **No
  network for package installs** (crates.io/pip/npm unreachable from the
  shell); GitHub git push/fetch and `gh api` (REST, repo-scoped) work.
- Owner's computer is reachable only sometimes (Claude desktop app must be
  open). Connected folders: `C:\Users\georg\Documents\Chorus`,
  `C:\Users\georg\AppData\Local\com.vectoredgex.chorus`,
  `C:\Users\georg\Documents\Sound Recordings`. Computer-use (screen control)
  was **declined** by the owner — don't ask again.
- Project (claude.ai) "Chorus" has `claude/chorus-status.md` (short status)
  and this handover mirrored as `claude/chorus-handover.md`.
