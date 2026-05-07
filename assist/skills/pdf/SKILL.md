---
name: pdf
description: Reading PDF files — manuals, reports, papers, brochures, handbooks, whitepapers, RFPs, spec sheets, and anything else with a .pdf extension. TRIGGER WORDS — PDF, .pdf, page, document, extract, manual, report, paper, brochure, handbook, whitepaper, RFP, spec sheet. MUST load before answering any question about a PDF in the workspace, or when the user pastes a path that ends in .pdf.
---

# PDF — orient, find, read via `execute`

## The rule

For any `.pdf` file, run shell commands via `execute`. **Never** use `read_file` on a PDF — `read_file` returns the file as a multimodal content block this model cannot consume, which causes the next API call to 400.

The sandbox has `pdftotext` and `pdfinfo` (poppler) installed. Use them.

## The pattern: orient → narrow → read

Three operations match how a person uses a big PDF. Pick the smallest one that answers the question:

### Orient — page count + first page

```
execute('pdfinfo foo.pdf')
```

Returns metadata including a `Pages: N` line. If the user just asks "how many pages", grep for it:

```
execute('pdfinfo foo.pdf | grep Pages')
```

For a quick taste of the contents, also extract page 1:

```
execute('pdftotext -f 1 -l 1 foo.pdf -')
```

### Find — search for a keyword

`pdftotext` writes plain text to stdout. Pipe to `grep` to find pages containing a term, with surrounding context:

```
execute('pdftotext foo.pdf - | grep -i -A2 -B1 "dosage"')
```

`-i` case-insensitive, `-A2 -B1` two lines after / one line before each match. If you also need page numbers, ask `pdftotext` for one page at a time and grep each — but typically the matched lines themselves locate you.

### Read — extract specific pages

```
execute('pdftotext -f 5 -l 10 foo.pdf -')
```

`-f` first page, `-l` last page (both 1-based, inclusive). Trailing `-` writes to stdout.

## Anti-patterns

- **Never `read_file('foo.pdf')`.** Returns multimodal — model can't parse, next request 400s.
- **Don't dump the whole PDF.** A 200-page document blown into a single `pdftotext foo.pdf -` call drowns your context. Orient first; narrow second.
- **Don't `cat` a PDF.** Same problem as `read_file` — binary garbage.

## Edge cases

- **Encrypted PDF** — `pdftotext` exits with `Error: Incorrect password`. Tell the user the document is password-protected; don't guess at contents.
- **Image-only / scanned PDF** — `pdftotext` returns empty output per page. The sandbox doesn't have OCR; tell the user.
- **Wrong file type** — `pdftotext` errors with `Syntax Error: May not be a PDF file`. Confirm the extension and ask the user.

## When the result got evicted

The agent's tool-result eviction middleware will replace any large `execute` output with a short preview pointing at `/large_tool_results/<id>`. **Don't `read_file` that path** — it'll round-trip the same big text and re-evict. Instead, re-run `execute` with a narrower scope: a smaller `-f`/`-l` range, or a `grep` for the specific term.

## Examples

> User: "How many pages is the treatment guide?"
>
> ```
> execute('pdfinfo treatment-guide.pdf | grep Pages')
> ```
> Returns `Pages: 47`. Answer the user.

> User: "What does the report say about Q3 revenue?"
>
> ```
> execute('pdftotext report.pdf - | grep -i -A2 "Q3 revenue"')
> ```
> Returns matching lines with two-line context per hit. Cite the figures in your response; if you need a fuller section, follow up with a page-range `pdftotext`.

> User: "Summarize section 2 of the manual."
>
> Orient first to find where section 2 starts:
> ```
> execute('pdftotext -f 1 -l 1 manual.pdf -')
> ```
> Then if the table of contents on page 1 says section 2 starts at page 12:
> ```
> execute('pdftotext -f 12 -l 20 manual.pdf -')
> ```

## Page references in answers

When citing a PDF in your response to the user, include a page number where reasonable: *"Section 2 (page 12) covers..."*. The user can jump straight to the page; you can re-run the right `pdftotext -f N -l M` later if the conversation turns back to it.
