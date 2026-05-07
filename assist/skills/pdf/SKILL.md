---
name: pdf
description: Reading PDF files — manuals, reports, papers, brochures, handbooks, whitepapers, RFPs, spec sheets, and anything else with a .pdf extension. TRIGGER WORDS — PDF, .pdf, page, document, extract, manual, report, paper, brochure, handbook, whitepaper, RFP, spec sheet. MUST load before answering any question about a PDF in the workspace, or when the user pastes a path that ends in .pdf.
---

# PDF — orient, find, read

## The rule

Use `read_pdf` for any `.pdf` file. **Never** use `read_file` on a PDF — it returns base64-encoded bytes the model can't actually read.

`read_pdf` has three modes. Pick the smallest one that answers the question.

| Mode | Call | Use when |
| --- | --- | --- |
| **Orient** | `read_pdf("foo.pdf")` | First contact. Returns page count + first page text. |
| **Find** | `read_pdf("foo.pdf", search="dosage")` | You have a keyword. Returns matching page numbers + ~3 lines per hit. |
| **Read** | `read_pdf("foo.pdf", pages="5")` or `pages="5-10"` | You know which pages you want. Returns full text of those pages. |

`search` and `pages` together is a 400 — pick one mode per call.

## The pattern: orient → narrow → read

For a PDF you've never seen:

1. `read_pdf("foo.pdf")` to learn what you're looking at.
2. If the question has a keyword, `read_pdf("foo.pdf", search="...")` to find the right pages.
3. If you need more than the search context, `read_pdf("foo.pdf", pages="N-M")` for the specific section.

**Don't dump the whole PDF.** A 200-page document doesn't fit in your useful context. Search or read in chunks.

## When the result got evicted

The agent's tool-result eviction middleware will replace any large `read_pdf` output with a short preview pointing at `/large_tool_results/<id>`. **Don't `read_file` that path** — it'll round-trip the same big text and re-evict. Instead, re-call `read_pdf` with a narrower scope: a smaller `pages` range, or a `search` for the specific term.

## Edge cases

- **Encrypted PDF** — `read_pdf` returns "Error: ... is password-protected — cannot extract." Tell the user; don't guess at contents.
- **Image-only / scanned PDF** — orient mode returns "No extractable text on the first page — this PDF may be image-based. OCR is not currently supported." Don't pretend you read it.
- **Wrong file extension** — magic-byte check refuses non-PDFs.

## Examples

> User: "How many pages is the treatment guide?"
> 
> ```
> read_pdf("treatment-guide.pdf")
> ```
> Returns header with page count. Answer the user.

> User: "What does the report say about Q3 revenue?"
> 
> ```
> read_pdf("report.pdf", search="Q3 revenue")
> ```
> Returns a few matching pages. If they answer the question, cite the page numbers in your response. If you need more context, follow up with `read_pdf("report.pdf", pages="N")`.

> User: "Summarize section 2 of the manual."
> 
> First orient to find where section 2 starts:
> ```
> read_pdf("manual.pdf")
> ```
> Then if the table of contents on page 1 says section 2 starts at page 12:
> ```
> read_pdf("manual.pdf", pages="12-20")
> ```

## Page references in answers

When citing a PDF in your response to the user, always include the page number: *"Section 2 (page 12) covers..."*. The user can jump straight to the page; you can re-read it later if the conversation turns back to it.
