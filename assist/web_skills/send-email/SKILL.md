---
name: send-email
description: Drafting or sending a plain-text email for the user. EXAMPLES — "email the contractor to reschedule"; "write this person a note about the meeting"; "send a thank-you email to this address". MUST load before drafting or sending an email.
allowed-tools: send_email
---

# Email drafting and approval

Draft an email with one recipient, a clear subject, and a complete plain-text body.
When the user has supplied enough information to send it, call
`send_email(to, subject, body)` in THIS turn with exactly what should be delivered.
Do not ask for a separate chat confirmation or end with "does this look good?": the
web approval card is the user's review and confirmation.

The call proposes the email. It does not send until the user reviews and approves the
web approval card. Never say an email was sent until the tool confirms it. The service
adds the fixed sender and oversight CC itself; do not ask for, invent, or try to set
From, CC, BCC, attachments, HTML, provider settings, or multiple recipients.

For several recipients, prepare and propose one separate email at a time so each
recipient receives an independently reviewed message.
