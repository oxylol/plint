---
name: spec-build
description: Turn a build request into a checked list before any code is written, build against that list, and report against it. Use when Filip asks for something bigger than a small fix - a new app or service, a clone or "like X", "all the features", several features in one message - or when a word in the request could be read two ways. Also when he types /spec-build followed by what he wants.
---

# Spec, then build

Filip's requests are short and the misses here have been misreadings, not bad code (see *Before you build* in
`CLAUDE.md`). This is the long form of that section: five steps, in order. The request is whatever follows
`/spec-build`, or the message that brought you here.

He is usually on a phone. Everything you show him in steps 1 to 3 fits on one screen: short lines, no tables.

## 1. Read it back

Write the request as a numbered list of things he will be able to see or do when it is done ("TALK stays on after
a silence", not "refactor the recogniser loop"). Each line must be something that can be checked and found wrong.

- **"All the features", "like X", "a clone of X":** open X itself (the page, the app, its docs) and list every part
  of it: tabs, modes, settings, what it does on a phone. Mark each one *build* or *not this time*. Do not list from
  memory: the Code tab was missed that way.
- **Words that could mean two things:** for each, one line under **Read another way** with the readings you
  considered and the one you took ("measure: a ruler he drags in the view; not the probe cycle"). If you cannot
  tell which he means, it becomes a question in step 3.
- **Left out:** what a reasonable person might expect from the request that you do not plan to build.
- **Not said by him:** anything that shapes what he will see or do and that he has not said (which device comes
  first, who uses it, where it saves, sound or none, how he will play or use it) is a question in step 3. It is
  never a line you take for granted: "phone first" was assumed that way and he wanted PC first.
- **Plumbing:** what he never sees (port, folder, stack name) is yours to pick; list it in a line.

Keep the list in `.scratch/spec-<name>.md` so it survives a long session. It is the thing you build and report against.

## 2. Can it be done

Before installing or building anything, look for what would make the result useless to him, and say it plainly:
formats, consoles or devices it cannot handle, a file or key or licence that is missing, hardware it needs, what it
costs to run, how slow it is. Reading its docs, checking that a file exists or a one-minute probe is fine here;
installing is not. RomM was set up and deleted the same day because this step was skipped.

If the thing he pictured cannot be done, stop and tell him now, with the nearest thing that can.

## 3. One message, then wait

Send the list from step 1, anything from step 2, and your questions together, once. Ask with `AskUserQuestion`:
four questions at most, the readings as choices he can tap, your own pick first. Then wait for his answer.

Do not wait when he has already said to go ahead without asking, or when the list holds no open question and the
whole thing is small and easily undone: then say how you read it and build.

When his answer changes the list, change the file and carry on. Do not send the whole list back for a second OK
unless the change is large.

## 4. Build against the list

- Every line of the list gets a check that can fail: a script against the API, a browser check (the Playwright
  image; the top of `apps/ultron/test/page.js` shows how it is run), or, where neither is possible, a named thing
  to look at by hand. Write the checks as you go, not at the end.
- Build and test on a scratch copy with its own port and its own data. Never on the live app: nothing is sent to
  it, reset in it or cleaned out of it while he may be using it.
- Work until every check passes, or a line turns out not to be possible. Then say so; do not quietly drop it.
- A long stretch of work gets a one-line note every few minutes: what is done, what is next. Silence reads as stuck.
- If you find yourself building something that is not on the list, stop: either it belongs on the list (tell him)
  or it does not belong in this build.

## 5. Report against the list

Lead with the result in a line. Then the list, each line marked **done**, **not done** (and why) or **not
checked** (and what it would take), with where he can see it: the address, or the path of a screenshot. Then what
was left out on purpose, and anything he has to do himself (reload, flip a switch, accept a licence).

A line you could not check is said to be unchecked. It is never reported as done.
