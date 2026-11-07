/**
 * Adds the Malayalam Bible (Sathyavedapusthakam, 1910 edition, public
 * domain) to rhema.db as translation "ML1910".
 *
 * Source: eBible.org, "mal1910" (verse-per-line text). Safe to run again:
 * an existing ML1910 is replaced.
 *
 * Run: bun run download:malayalam-bible
 *      bun run data/download-malayalam-bible.ts [path/to/rhema.db]
 */

import { Database } from "bun:sqlite"
import { existsSync, mkdirSync, readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"

const DATA_DIR = import.meta.dir
const DB_PATH = process.argv[2] ?? join(DATA_DIR, "rhema.db")
const WORK_DIR = join(DATA_DIR, "sources", "mal1910")
const ZIP_URL = "https://ebible.org/Scriptures/mal1910_vpl.zip"
const ABBREV = "ML1910"

// USFM book code -> [book number, Malayalam name, short name]
const BOOKS: Record<string, [number, string, string]> = {
  GEN: [1, "ഉല്പത്തി", "ഉല്പ"],
  EXO: [2, "പുറപ്പാട്", "പുറ"],
  LEV: [3, "ലേവ്യപുസ്തകം", "ലേവ്യ"],
  NUM: [4, "സംഖ്യാപുസ്തകം", "സംഖ്യ"],
  DEU: [5, "ആവർത്തനപുസ്തകം", "ആവ"],
  JOS: [6, "യോശുവ", "യോശു"],
  JDG: [7, "ന്യായാധിപന്മാർ", "ന്യായ"],
  RUT: [8, "രൂത്ത്", "രൂത്ത്"],
  "1SA": [9, "1 ശമൂവേൽ", "1 ശമൂ"],
  "2SA": [10, "2 ശമൂവേൽ", "2 ശമൂ"],
  "1KI": [11, "1 രാജാക്കന്മാർ", "1 രാജാ"],
  "2KI": [12, "2 രാജാക്കന്മാർ", "2 രാജാ"],
  "1CH": [13, "1 ദിനവൃത്താന്തം", "1 ദിന"],
  "2CH": [14, "2 ദിനവൃത്താന്തം", "2 ദിന"],
  EZR: [15, "എസ്രാ", "എസ്രാ"],
  NEH: [16, "നെഹെമ്യാവ്", "നെഹെ"],
  EST: [17, "എസ്ഥേർ", "എസ്ഥേ"],
  JOB: [18, "ഇയ്യോബ്", "ഇയ്യോ"],
  PSA: [19, "സങ്കീർത്തനങ്ങൾ", "സങ്കീ"],
  PRO: [20, "സദൃശവാക്യങ്ങൾ", "സദൃ"],
  ECC: [21, "സഭാപ്രസംഗി", "സഭാ"],
  SNG: [22, "ഉത്തമഗീതം", "ഉത്ത"],
  ISA: [23, "യെശയ്യാവ്", "യെശ"],
  JER: [24, "യിരെമ്യാവ്", "യിരെ"],
  LAM: [25, "വിലാപങ്ങൾ", "വിലാ"],
  EZK: [26, "യെഹെസ്കേൽ", "യെഹെ"],
  DAN: [27, "ദാനീയേൽ", "ദാനീ"],
  HOS: [28, "ഹോശേയ", "ഹോശേ"],
  JOL: [29, "യോവേൽ", "യോവേ"],
  AMO: [30, "ആമോസ്", "ആമോ"],
  OBA: [31, "ഓബദ്യാവ്", "ഓബ"],
  JON: [32, "യോനാ", "യോനാ"],
  MIC: [33, "മീഖാ", "മീഖാ"],
  NAM: [34, "നഹൂം", "നഹൂം"],
  HAB: [35, "ഹബക്കൂക്ക്", "ഹബ"],
  ZEP: [36, "സെഫന്യാവ്", "സെഫ"],
  HAG: [37, "ഹഗ്ഗായി", "ഹഗ്ഗാ"],
  ZEC: [38, "സെഖര്യാവ്", "സെഖ"],
  MAL: [39, "മലാഖി", "മലാ"],
  MAT: [40, "മത്തായി", "മത്താ"],
  MRK: [41, "മർക്കൊസ്", "മർക്കൊ"],
  LUK: [42, "ലൂക്കൊസ്", "ലൂക്കൊ"],
  JHN: [43, "യോഹന്നാൻ", "യോഹ"],
  ACT: [44, "പ്രവൃത്തികൾ", "പ്രവൃ"],
  ROM: [45, "റോമർ", "റോമ"],
  "1CO": [46, "1 കൊരിന്ത്യർ", "1 കൊരി"],
  "2CO": [47, "2 കൊരിന്ത്യർ", "2 കൊരി"],
  GAL: [48, "ഗലാത്യർ", "ഗലാ"],
  EPH: [49, "എഫെസ്യർ", "എഫെ"],
  PHP: [50, "ഫിലിപ്പിയർ", "ഫിലി"],
  COL: [51, "കൊലൊസ്സ്യർ", "കൊലൊ"],
  "1TH": [52, "1 തെസ്സലൊനീക്യർ", "1 തെസ്സ"],
  "2TH": [53, "2 തെസ്സലൊനീക്യർ", "2 തെസ്സ"],
  "1TI": [54, "1 തിമൊഥെയൊസ്", "1 തിമൊ"],
  "2TI": [55, "2 തിമൊഥെയൊസ്", "2 തിമൊ"],
  TIT: [56, "തീത്തൊസ്", "തീത്തൊ"],
  PHM: [57, "ഫിലേമോൻ", "ഫിലേ"],
  HEB: [58, "എബ്രായർ", "എബ്രാ"],
  JAS: [59, "യാക്കോബ്", "യാക്കോ"],
  "1PE": [60, "1 പത്രൊസ്", "1 പത്രൊ"],
  "2PE": [61, "2 പത്രൊസ്", "2 പത്രൊ"],
  "1JN": [62, "1 യോഹന്നാൻ", "1 യോഹ"],
  "2JN": [63, "2 യോഹന്നാൻ", "2 യോഹ"],
  "3JN": [64, "3 യോഹന്നാൻ", "3 യോഹ"],
  JUD: [65, "യൂദാ", "യൂദാ"],
  REV: [66, "വെളിപ്പാട്", "വെളി"],
}

async function download(): Promise<string> {
  mkdirSync(WORK_DIR, { recursive: true })
  const zip = join(WORK_DIR, "mal1910_vpl.zip")
  if (!existsSync(zip)) {
    console.log(`  Downloading ${ZIP_URL}...`)
    const res = await fetch(ZIP_URL)
    if (!res.ok) throw new Error(`Download failed: ${res.status} ${res.statusText}`)
    await Bun.write(zip, await res.arrayBuffer())
  }
  const proc = Bun.spawn(["unzip", "-o", "-q", zip, "-d", WORK_DIR], { stderr: "inherit" })
  if ((await proc.exited) !== 0) throw new Error("unzip failed")
  const txt = readdirSync(WORK_DIR).find((f) => f.endsWith("_vpl.txt") || f.endsWith("vpl.txt"))
  if (!txt) throw new Error(`No verse text in ${readdirSync(WORK_DIR).join(", ")}`)
  return join(WORK_DIR, txt)
}

interface Verse {
  book: number
  chapter: number
  verse: number
  text: string
}

export function parseVpl(raw: string): Verse[] {
  const out: Verse[] = []
  for (const line of raw.split(/\r?\n/)) {
    // "GEN 1:1 ആദിയിൽ ദൈവം ആകാശവും ഭൂമിയും സൃഷ്ടിച്ചു."
    const m = line.match(/^([1-3]?[A-Z]{2,3})\s+(\d+):(\d+)\s+(.*)$/)
    if (!m) continue
    const book = BOOKS[m[1]]
    if (!book) continue // apocrypha or front matter
    const text = m[4].trim()
    if (!text) continue
    out.push({ book: book[0], chapter: Number(m[2]), verse: Number(m[3]), text })
  }
  return out
}

async function main() {
  if (!existsSync(DB_PATH)) {
    throw new Error(`${DB_PATH} not found. Build the Bible database first (bun run build:bible).`)
  }
  console.log("\n📖 Malayalam Bible (Sathyavedapusthakam 1910)\n")
  const verses = parseVpl(readFileSync(await download(), "utf-8"))
  const books = new Set(verses.map((v) => v.book))
  console.log(`  ${verses.length} verses in ${books.size} books`)
  if (books.size !== 66 || verses.length < 30000) {
    throw new Error("Unexpected source contents; nothing was changed")
  }

  const db = new Database(DB_PATH)
  db.exec("BEGIN TRANSACTION")
  const old = db.query("SELECT id FROM translations WHERE abbreviation = ?").get(ABBREV) as
    | { id: number }
    | null
  if (old) {
    const hasFts = db.query("SELECT name FROM sqlite_master WHERE name = 'verses_fts'").get()
    if (hasFts) {
      db.run(
        "INSERT INTO verses_fts(verses_fts, rowid, text) SELECT 'delete', id, text FROM verses WHERE translation_id = ?",
        [old.id]
      )
    }
    db.run("DELETE FROM verses WHERE translation_id = ?", [old.id])
    db.run("DELETE FROM books WHERE translation_id = ?", [old.id])
    db.run("DELETE FROM translations WHERE id = ?", [old.id])
  }
  db.run(
    "INSERT INTO translations (abbreviation, title, language, license) VALUES (?, ?, ?, ?)",
    [ABBREV, "സത്യവേദപുസ്തകം (Malayalam Bible 1910)", "ml", "Public Domain"]
  )
  const tid = (db.query("SELECT last_insert_rowid() AS id").get() as { id: number }).id
  const insertBook = db.prepare(
    "INSERT INTO books (translation_id, book_number, name, abbreviation, testament) VALUES (?, ?, ?, ?, ?)"
  )
  const bookIds = new Map<number, number>()
  for (const [num, name, short] of Object.values(BOOKS)) {
    insertBook.run(tid, num, name, short, num <= 39 ? "OT" : "NT")
    bookIds.set(num, (db.query("SELECT last_insert_rowid() AS id").get() as { id: number }).id)
  }
  const byNum = new Map(Object.values(BOOKS).map(([n, name, short]) => [n, { name, short }]))
  const insertVerse = db.prepare(
    "INSERT INTO verses (translation_id, book_id, book_number, book_name, book_abbreviation, chapter, verse, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
  )
  for (const v of verses) {
    const b = byNum.get(v.book)!
    insertVerse.run(tid, bookIds.get(v.book)!, v.book, b.name, b.short, v.chapter, v.verse, v.text)
  }
  const hasFts = db.query("SELECT name FROM sqlite_master WHERE name = 'verses_fts'").get()
  if (hasFts) {
    db.run(
      "INSERT INTO verses_fts(rowid, text) SELECT id, text FROM verses WHERE translation_id = ?",
      [tid]
    )
  }
  db.exec("COMMIT")
  const john316 = db
    .query("SELECT text FROM verses WHERE translation_id = ? AND book_number = 43 AND chapter = 3 AND verse = 16")
    .get(tid) as { text: string } | null
  console.log(`  ✓ Added ${ABBREV}. John 3:16: ${john316?.text ?? "(missing!)"}`)
  db.close()
}

if (import.meta.main) {
  await main()
}
