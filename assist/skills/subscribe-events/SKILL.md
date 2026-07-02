---
name: subscribe-events
description: Subscribe this conversation to inbound messages (texts) so the agent triages each one against your instructions — e.g. "watch my texts and if it's from the plumber tell me", "when Ana messages about pickup, draft a reply", "stop watching texts from that number", "change how you handle messages from work". Load whenever the user wants incoming messages handled automatically by rules in this thread.
---

# Subscribing to inbound message events

This thread can watch inbound messages and, for each one whose **sender** matches a
pattern, run a turn **here** where you triage the message against instructions you wrote.
The subscription belongs to **this** thread. Use the subscription tools — don't invent your
own matching.

## Tools
- `create_subscription(sender_regexp, template)` — start watching.
- `list_subscriptions()` / `modify_subscription(id, ...)` / `pause_subscription(id)` /
  `resume_subscription(id)` / `delete_subscription(id)`.

## sender_regexp — match the sender
A Python regexp searched against the sender string (a phone number like `+15551234567`, or
an alphanumeric shortcode). Examples: a specific number `^\+15551234567$`; a prefix
`^\+1555`; **every** sender `.*`. **If the sender rule is at all complex (multiple numbers,
alternation, anchoring you're unsure about), load the `regexp` skill first** and use it to
build and sanity-check the pattern before creating the subscription — a wrong regexp
silently matches nothing or the wrong senders.

## template — your whole instruction to yourself
`template` is the entire prompt each matching message runs. Write it as instructions to
*yourself* for triaging one message: what to look for, the user's rules, and what to do.
Put the literal slots `{sender}` and `{text}` where the message's sender and body belong.
You get the sender as its raw string (e.g. a phone number) — if a rule is about a person by
name, say so in the template ("if {sender} is Ana (+15551234567), …") and resolve it
yourself at triage time. A good template ends by telling yourself to **state your decision
every time** — either propose a reply or say explicitly that no action is needed — so a
message never vanishes silently.

Example template:
```
A new text arrived from {sender}:
{text}

Decide what to do:
- If it's about a delivery or appointment, note the key details for me.
- If it clearly needs a quick reply I'd approve, propose one with send_reply.
- Otherwise, say briefly why no action is needed.
Always end by stating your decision.
```

## What you can DO with a matched message
You can read context and **propose a reply** with `send_reply(text)` — the user approves it
before anything is sent; you cannot send on your own. Everything else you do stays in this
thread. Never claim you sent a reply — you proposed it.

## After any change
Relay back the sender pattern and a short description of the template so the user can catch
a misread. If a tool returns an error or "couldn't", tell the user — don't claim it worked.
