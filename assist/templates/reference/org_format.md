Org format — quick reference

- Headings/Subheadings: Start lines with one or more asterisks. One * for top-level, ** for second-level, and so on. Example:
  * Project
  ** Tasks
  *** Subtask

- Headings and section body: All of the content between a heading and the next heading of the same level is considered the body of the first heading. Whenever you're adding a new heading, do so **before the next heading** and not just under the previous heading of the same level. Example:
  * Heading 1
  This is the content of heading 1.
  
  I am also heading 1 content
  ** sub-section of heading 1
  This is the content of both heading 1 its subsection
  * Heading 2
  This content does not belong to heading 1

- Bullets/Lists: Use - or + followed by a space. Indent to nest sublists. Ordered lists use 1., 2., 3. (indent similarly).
  - Item A
    - Subitem A1
  1. First
  2. Second

- Bold/Italics/Underline: Wrap text with the corresponding markers.
  - *bold*
  - /italic/
  - _underline_
  - +strike-through+
  - =verbatim= and ~code~

- URLs/Links: Plain URLs are recognized automatically. For labeled links, use [[URL][Label]].
  - https://orgmode.org
  - [[https://orgmode.org][Org Mode site]]

Notes:
- Leave a blank line between blocks for clarity.


