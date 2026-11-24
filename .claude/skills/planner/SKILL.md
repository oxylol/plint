---
name: planner
description: Create a comprehensive feature plan with deep codebase analysis and research.
---

# Plan a new task

Use this after we've aligned on the goal. We do not write code in this phase.

Write the plan as one file with these sections: Goal and done, Read first, Tasks, Checkpoints, How to run it, Proof, Progress. Leave Progress empty at the start.

Parts 4 to 6 are instructions for the session that runs the plan, so write them into the plan for it to follow.

## 1. Start from the agreed goal

Write the goal as a user story and list the acceptance criteria. If anything is unclear, ask me before you plan.

## 2. Read before you write

Before you write the plan, read the code, docs and patterns this work touches. For each thing you'll change, find out what it's used for and what depends on it. If you can't establish it, ask me. In the plan, list the files the builder must read first, with line numbers and why, the docs with the exact section and why, and the existing code to copy. Someone who has never seen this project must be able to build from the plan alone.

## 3. Cut the work for sub-agents

Split the work into tasks by file. Each task owns its files, and no two tasks in the same checkpoint share a file. For each task, write what to change, in which file, the existing code to copy (file and line), the known pitfalls, and the command that proves it works. Mark which tasks can run in parallel, and give the reason whenever one has to wait. Cut the tasks so as many as possible run in parallel.

## 4. Set the checkpoints

Group the tasks into checkpoints. Each checkpoint must deliver something that works on its own and name the check that proves it. Build one checkpoint at a time: run its check, commit it, write the result and the evidence into the plan, and only then start the next. If the change is a single step, skip checkpoints.

## 5. Write how it runs

Read the whole plan before you start. Run it as the lead: you write no code. At the start of each checkpoint, give every task that can run in parallel to its own sub-agent, all at once. Each sub-agent's brief names its task, its files and only those, the bar to meet and the format to report back in, and tells it not to run git. You alone run the checkpoint's check, commit, and write the result into this plan. If something comes up that isn't in the plan, write it into the plan before anyone builds it. After each checkpoint, if your context window is half full, write which checkpoint is next and stop; a fresh session continues from this plan.

## 6. Define the proof

For each acceptance criterion, name the test that proves it and the one-line change to the code that should make that test fail. List the commands that check the whole build: lint, tests and the manual steps. When a test is written, make that one-line change, confirm the test fails, then undo the change with the exact reverse edit, never by resetting files. Before each commit, check the actual files, never the sub-agent's report. Anything that doesn't match goes back to its sub-agent or is fixed, never accepted on trust. Run the checks; if one fails, fix it and run it again, and commit only when all pass. Before you call it done, use it through its real entry point the way a user would, and check that normal use still works.

## 7. Check the plan, then hand it over

When the plan is written, give it to a fresh sub-agent to review. It reads the plan file itself and returns only its findings on four questions: does the plan serve the goal, is it complete, is it safe, and can it be built first time. Write every finding into the plan, never only into the chat. Save the plan as one file. End by telling me to clear the context and run the plan in a fresh session.
