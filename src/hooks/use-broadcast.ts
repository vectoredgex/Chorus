import { invoke } from "@tauri-apps/api/core"
import { mark } from "@/lib/latency-marks"
import { useBroadcastStore } from "@/stores/broadcast-store"
import { useBibleStore } from "@/stores/bible-store"
import { useSettingsStore } from "@/stores/settings-store"
import { resolveBilingual, topTranslation } from "@/lib/bilingual"
import type { VerseRenderData } from "@/types"
import type { Verse } from "@/types"

export function toVerseRenderData(verse: Verse, translation: string): VerseRenderData {
  return {
    reference: `${verse.book_name} ${verse.chapter}:${verse.verse} (${translation})`,
    segments: [{ verseNumber: verse.verse, text: verse.text }],
  }
}

// Monotonic ticket so an older present whose refetch resolves late can't
// overwrite a newer one (detections can present in quick succession).
let presentSeq = 0

/**
 * Refetch the full verse from the backend and present it to preview + live
 * output. Queue/detection items may carry partial verse objects (empty text,
 * synthetic id 0, stale translation_id), so always look the verse up by
 * book/chapter/verse against the active translation. Falls back to the passed
 * verse if the lookup fails — never blanks the output on error.
 *
 * This is the ONLY path that changes the live output verse. Manual selection
 * (verse click, arrow navigation, search) presents directly; detections update
 * only the preview unless the Auto toggle is on.
 */
export async function presentVerse(verse: Verse): Promise<void> {
  presentSeq += 1
  const ticket = presentSeq
  mark(`presentVerse start ${verse.book_name} ${verse.chapter}:${verse.verse}`)

  const bilingual = useSettingsStore.getState().showMalayalam
  const startState = useBibleStore.getState()
  const topId =
    topTranslation(startState.translations, startState.activeTranslationId, bilingual)?.id ??
    startState.activeTranslationId

  let verseToPresent = verse
  try {
    const fullVerse = await invoke<Verse | null>("get_verse", {
      translationId: topId,
      bookNumber: verse.book_number,
      chapter: verse.chapter,
      verse: verse.verse,
    })
    verseToPresent = fullVerse ?? verse
  } catch (e) {
    console.warn("[broadcast] get_verse refetch failed, using cached verse:", e)
  }
  mark(`presentVerse get_verse done ${verse.book_name} ${verse.chapter}:${verse.verse}`)

  // A newer present started while we were refetching — let it win.
  if (ticket !== presentSeq) return

  // Never present a verse with no text (e.g. a citation that doesn't exist
  // in the active translation) — keep whatever is currently live instead.
  if (!verseToPresent.text?.trim()) {
    console.warn(
      "[broadcast] refusing to present verse with empty text:",
      `${verseToPresent.book_name} ${verseToPresent.chapter}:${verseToPresent.verse}`,
    )
    return
  }

  const bibleState = useBibleStore.getState()
  const { render } = await resolveBilingual(
    verseToPresent,
    bibleState.translations,
    bibleState.activeTranslationId,
    bilingual,
  )
  if (ticket !== presentSeq) return

  // Keep the preview in sync with what was just presented.
  bibleState.selectVerse(verseToPresent)
  useBroadcastStore.getState().setLiveVerse(render, verseToPresent)
  mark(`presentVerse end ${verseToPresent.book_name} ${verseToPresent.chapter}:${verseToPresent.verse}`)
}

export const broadcastActions = {
  setLiveVerse: (verse: VerseRenderData | null) =>
    useBroadcastStore.getState().setLiveVerse(verse),
  setLive: (live: boolean) =>
    useBroadcastStore.getState().setLive(live),
  getActiveTheme: () => {
    const s = useBroadcastStore.getState()
    return s.themes.find((t) => t.id === s.activeThemeId) ?? s.themes[0]
  },
}
