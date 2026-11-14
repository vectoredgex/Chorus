import { beforeEach, describe, expect, it, vi } from "vitest"
import type { Verse } from "@/types"
import {
  buildVerseRenderData,
  resolveBilingual,
  topTranslation,
} from "./bilingual"
import { splitVerseBlocks } from "./verse-renderer"

const mockInvoke = vi.fn()
vi.mock("@tauri-apps/api/core", () => ({
  invoke: (...args: unknown[]) => mockInvoke(...args),
}))

const KJV = { id: 1, abbreviation: "KJV", language: "en" }
const NKJV = { id: 2, abbreviation: "NKJV", language: "en" }
const ML = { id: 9, abbreviation: "ML1910", language: "ml" }

function verse(translation_id: number, text: string): Verse {
  return {
    id: 1,
    translation_id,
    book_number: 43,
    book_name: translation_id === 9 ? "യോഹന്നാൻ" : "John",
    book_abbreviation: "",
    chapter: 3,
    verse: 16,
    text,
  }
}

const EN = "For God so loved the world"
const MLT = "തന്റെ ഏകജാതനായ പുത്രനിൽ വിശ്വസിക്കുന്ന ഏവനും"

describe("bilingual verses", () => {
  beforeEach(() => {
    mockInvoke.mockReset()
    mockInvoke.mockImplementation((_cmd: string, args: { translationId: number }) =>
      Promise.resolve(verse(args.translationId, args.translationId === 9 ? MLT : EN)),
    )
  })

  it("puts Malayalam on its own block under English", () => {
    const data = buildVerseRenderData(verse(1, EN), "KJV", MLT)
    expect(data.reference).toBe("John 3:16 (KJV)")
    const blocks = splitVerseBlocks(data)
    expect(blocks).toHaveLength(2)
    expect(blocks[0].segments[0]).toMatchObject({ verseNumber: 16, text: EN })
    expect(blocks[1].segments[0].text).toBe(MLT)
  })

  it("uses KJV on top when the Malayalam Bible is picked", () => {
    expect(topTranslation([KJV, NKJV, ML], 9, true)?.abbreviation).toBe("KJV")
    expect(topTranslation([KJV, NKJV, ML], 2, true)?.abbreviation).toBe("NKJV")
    expect(topTranslation([KJV, NKJV, ML], 9, false)?.abbreviation).toBe("ML1910")
    // No Malayalam Bible installed: unchanged.
    expect(topTranslation([KJV, NKJV], 2, true)?.abbreviation).toBe("NKJV")
  })

  it("turns a Malayalam verse into English over Malayalam", async () => {
    const { top, render } = await resolveBilingual(verse(9, MLT), [KJV, ML], 9, true)
    expect(top.text).toBe(EN)
    expect(render.reference).toBe("John 3:16 (KJV)")
    expect(render.segments.map((s) => s.text)).toEqual([EN, MLT])
  })

  it("shows only the picked Bible when switched off", async () => {
    const { render } = await resolveBilingual(verse(1, EN), [KJV, ML], 1, false)
    expect(render.segments.map((s) => s.text)).toEqual([EN])
    expect(mockInvoke).not.toHaveBeenCalled()
  })
})
