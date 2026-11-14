import { useEffect, useState } from "react"
import { PanelHeader } from "@/components/ui/panel-header"
import { CanvasVerse } from "@/components/ui/canvas-verse"
import { useBibleStore, useBroadcastStore, useSettingsStore } from "@/stores"
import { resolveBilingual } from "@/lib/bilingual"
import type { VerseRenderData } from "@/types"
import { bibleActions } from "@/hooks/use-bible"
import { toVerseRenderData } from "@/hooks/use-broadcast"

export function PreviewPanel() {
  const selectedVerse = useBibleStore((s) => s.selectedVerse)
  const translations = useBibleStore((s) => s.translations)
  const activeTranslationId = useBibleStore((s) => s.activeTranslationId)

  // When translation changes, re-fetch the selected verse in the new translation
  useEffect(() => {
    const verse = useBibleStore.getState().selectedVerse
    if (verse && verse.book_number > 0 && verse.chapter > 0 && verse.verse > 0) {
      bibleActions
        .fetchVerse(verse.book_number, verse.chapter, verse.verse)
        .then((v) => {
          if (v) bibleActions.selectVerse(v)
        })
        .catch(() => {})
    }
  }, [activeTranslationId])
  const themes = useBroadcastStore((s) => s.themes)
  const activeThemeId = useBroadcastStore((s) => s.activeThemeId)

  const activeTheme = themes.find((t) => t.id === activeThemeId) ?? themes[0]
  const translation = translations.find((t) => t.id === activeTranslationId)?.abbreviation ?? "KJV"
  const showMalayalam = useSettingsStore((s) => s.showMalayalam)

  // English on top, Malayalam under it. Show the plain verse at once, then
  // the paired version as soon as the lookups finish.
  const [paired, setPaired] = useState<{ key: string; data: VerseRenderData } | null>(null)
  const verseKey = selectedVerse
    ? `${selectedVerse.translation_id}:${selectedVerse.book_number}:${selectedVerse.chapter}:${selectedVerse.verse}:${activeTranslationId}:${showMalayalam}`
    : ""
  useEffect(() => {
    if (!selectedVerse) return
    let cancelled = false
    resolveBilingual(selectedVerse, translations, activeTranslationId, showMalayalam)
      .then(({ render }) => {
        if (!cancelled) setPaired({ key: verseKey, data: render })
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [verseKey, translations])

  const verseData = selectedVerse
    ? paired?.key === verseKey
      ? paired.data
      : toVerseRenderData(selectedVerse, translation)
    : null

  return (
    <div
      data-slot="preview-panel"
      className="flex min-h-0 flex-col overflow-hidden rounded-lg border border-border bg-card"
    >
      <PanelHeader title="Program preview" />
      <div className="flex min-h-0 flex-1 items-center justify-center p-3">
        <CanvasVerse theme={activeTheme} verse={verseData} />
      </div>
    </div>
  )
}
