import { invoke } from "@tauri-apps/api/core"
import type { Translation, Verse, VerseRenderData } from "@/types"

/**
 * Chorus: show every verse in English with the same verse in Malayalam
 * directly under it. The English Bible is the one picked in Settings, or KJV
 * when a Malayalam Bible is picked (the Malayalam engines switch to it).
 * When no Malayalam Bible is installed, or the setting is off, verses show
 * in the picked Bible only, exactly as before.
 */

type TranslationLike = Pick<Translation, "id" | "abbreviation" | "language">

export function isMalayalam(t: TranslationLike | undefined): boolean {
  return t?.language === "ml"
}

export function findMalayalamTranslation<T extends TranslationLike>(
  translations: T[]
): T | undefined {
  return (
    translations.find((t) => t.abbreviation === "ML1910") ??
    translations.find((t) => isMalayalam(t))
  )
}

/** The Bible shown on top: the picked one, or English KJV when Malayalam is picked. */
export function topTranslation<T extends TranslationLike>(
  translations: T[],
  activeId: number,
  bilingual: boolean
): T | undefined {
  const active = translations.find((t) => t.id === activeId)
  if (!bilingual || !isMalayalam(active) || !findMalayalamTranslation(translations)) {
    return active
  }
  return (
    translations.find((t) => t.abbreviation === "KJV") ??
    translations.find((t) => t.language === "en") ??
    active
  )
}

/** Build what the screen draws: English verse, then the Malayalam verse on a new line. */
export function buildVerseRenderData(
  verse: Verse,
  abbreviation: string,
  malayalamText?: string | null
): VerseRenderData {
  const segments: VerseRenderData["segments"] = [
    { verseNumber: verse.verse, text: verse.text },
  ]
  const ml = malayalamText?.trim()
  if (ml && ml !== verse.text.trim()) {
    segments.push({ text: ml, newBlock: true })
  }
  return {
    reference: `${verse.book_name} ${verse.chapter}:${verse.verse} (${abbreviation})`,
    segments,
  }
}

const cache = new Map<string, Verse | null>()

/** get_verse with a small cache (verses never change while the app runs). */
export async function getVerseCached(
  translationId: number,
  bookNumber: number,
  chapter: number,
  verse: number
): Promise<Verse | null> {
  const key = `${translationId}:${bookNumber}:${chapter}:${verse}`
  if (cache.has(key)) return cache.get(key) ?? null
  const v = await invoke<Verse | null>("get_verse", {
    translationId,
    bookNumber,
    chapter,
    verse,
  })
  if (cache.size > 2000) cache.clear()
  cache.set(key, v ?? null)
  return v ?? null
}

/**
 * Turn any verse (English, Malayalam, or a partial queue/detection item) into
 * the English-over-Malayalam render data. Returns the English verse used on
 * top so callers can keep it as the source verse.
 */
export async function resolveBilingual(
  verse: Verse,
  translations: TranslationLike[],
  activeId: number,
  bilingual: boolean
): Promise<{ top: Verse; render: VerseRenderData }> {
  const top = topTranslation(translations, activeId, bilingual)
  const topAbbrev = top?.abbreviation ?? "KJV"
  let topVerse = verse
  if (top && (verse.translation_id !== top.id || !verse.text?.trim())) {
    try {
      topVerse =
        (await getVerseCached(top.id, verse.book_number, verse.chapter, verse.verse)) ??
        verse
    } catch {
      topVerse = verse
    }
  }

  let malayalamText: string | null = null
  const ml = bilingual ? findMalayalamTranslation(translations) : undefined
  if (ml && !isMalayalam(top)) {
    try {
      const v = await getVerseCached(ml.id, verse.book_number, verse.chapter, verse.verse)
      malayalamText = v?.text ?? null
    } catch {
      malayalamText = null
    }
  }

  return { top: topVerse, render: buildVerseRenderData(topVerse, topAbbrev, malayalamText) }
}
