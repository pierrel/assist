---
name: explore-website
description: "Explore ONE specific website to find and download a file — a manual, PDF, spec sheet, dataset, image, export. Navigate with read_url, download with curl. EXAMPLES — 'download the user manual PDF from the fellow website'; 'get the CSV linked on this vendor dashboard page'; 'save the spec sheet from acme.com/products/x'. This is NOT general web research (that's the research agent, which searches). MUST load before fetching pages or files from a specific website with read_url or curl."
allowed-tools: read_url
---

# Explore a website — read to navigate, curl to download

You have a specific website (or a page URL) and need to find and download one
or more files from it. The rule is simple: **`read_url` finds; `curl`
downloads.** Never swap them.

## The failure this skill exists to prevent — read this first

Do **not** hunt for a file by curling page after page of a site's raw HTML.
Modern sites (Shopify, most storefronts, docs sites) return HTML stuffed with
dozens of asset URLs — `/cdn/…/assets/*.js`, CSS, fonts, thumbnails. If you
`curl` a page and start following those looking for your file, you will
follow link after link, each a dead end, and **never converge**. You will burn
the step limit and the user gets nothing after a long wait. There is no hope
down that path — a file is never found by crawling a site's assets.

`read_url` exists precisely to avoid this: it strips the asset noise and hands
you the page's **real links**. Use it.

## Find the file — with `read_url`

`read_url(url)` returns the page's readable text **plus** a "Links on this
page:" list of real URLs (absolute). That link list is how you navigate.

1. `read_url` the URL the user gave (or the site homepage).
2. In the surfaced links, pick the one section that would hold your file —
   Support, Downloads, Manuals, Resources, or the specific product page.
   Ignore `/assets/`, `.js`, `.css`, font, and thumbnail links entirely.
3. `read_url` that page. Repeat until a surfaced link **is** your file — it
   ends in `.pdf` / `.csv` / `.zip` / `.xlsx` etc., or sits under a downloads
   / files / CDN-files path.
4. **Budget: 2–4 `read_url` hops.** read_url is host-side and safe. But if a
   few hops don't surface the file, **stop** — tell the user you couldn't find
   it and name the pages you checked. Do not keep hopping.

## Download the file — with `curl`

Once `read_url` gave you the file's real URL, download it in the sandbox:

```
curl -L -o /workspace/<name> "<file-url>"
```

`-L` follows redirects, `-o` writes into `/workspace`.

- Downloading uses the sandbox's network, which is allowlisted. If curl gets a
  **403 from the proxy** ("tunnel"/"proxy" in the error), the host isn't
  approved: call `request_egress(host, port, task)`, tell the user approval is
  waiting, and **stop** — do not retry until approved. (Load the `egress`
  skill for that flow.)
- **Multiple files:** `read_url` the page once, collect **all** the file URLs
  from its surfaced links, then `curl` each. Don't re-read the page between
  downloads.
- After downloading, inspect it (e.g. `pdfinfo file.pdf`, `head`, `wc -l`) and
  report what you got — filename, size, page/row count.

## Do not

- **Do not `curl` a page to read it.** curl returns raw asset-soup HTML;
  `read_url` returns clean links. curl is only for downloading the file you
  already found.
- **Do not web-search for a file on a site you already have.** For a known
  site, `read_url` it directly — searching hits our rate limits and isn't
  needed. (If you genuinely don't know the site at all, **one** search to find
  the official site is fine — then switch to `read_url`.)
- **Do not crawl `/assets/`, `.js`, `.css`, font, or thumbnail URLs.** They
  never contain your file.
- **Do not loop.** Re-reading the same page or re-downloading makes no
  progress; the reread guard will stop you. Treat that stop as a signal to
  change approach or report failure — not to try the same URL again.

## Worked example (fictional site)

User: *"Download the Fathom Tide Kettle manual PDF from
`https://www.fathomcoffee.example/`."*

1. `read_url("https://www.fathomcoffee.example/")` → links include
   `…/pages/support  (Support & manuals)`.
2. `read_url(".../pages/support")` → links include
   `…/pages/tide-manual  (Tide Kettle manual)`.
3. `read_url(".../pages/tide-manual")` → link
   `…/cdn/shop/files/Fathom_Tide_Manual_v3.pdf`.
4. `curl -L -o /workspace/Fathom_Tide_Manual.pdf ".../Fathom_Tide_Manual_v3.pdf"`
5. `pdfinfo /workspace/Fathom_Tide_Manual.pdf` → report the page count.

Three reads, one download, zero asset crawling.

## When a site needs JavaScript

Some sites build their links or download buttons with JavaScript, so
`read_url` (which runs no JS) and `curl` won't see the file. If `read_url`
shows the page but none of its surfaced links lead to the file and you're
confident it exists, **say so** — that page needs a headless browser, which
isn't available yet. Don't curl-crawl trying to work around it; tell the user
it needs browser rendering.
