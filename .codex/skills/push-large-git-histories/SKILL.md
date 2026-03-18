---
name: push-large-git-histories
description: Push large Git branches and repositories to remotes when normal pushes stall, time out, or hit provider size limits. Use when Codex must move a local branch to a remote exactly, batch existing commits forward, diagnose oversized blobs, decide whether Git LFS is required, and verify that local and remote refs end at the intended commit.
---

# Push Large Git Histories

## Overview

Use this skill to push difficult Git branches without guessing. Start by identifying whether the job is:

- a direct push of existing history
- a batched push through the existing commit chain
- a content-preserving sync that may require Git LFS or another size-handling change

Read [references/playbook.md](references/playbook.md) when you need exact commands, large-blob diagnostics, or PowerShell-friendly batching patterns.

## Workflow

1. Read current state from Git before deciding anything.
2. Determine the user's hard constraints:
   - exact history must be preserved
   - content/state must match, but history rewrite is acceptable
   - force-push is allowed or forbidden
3. Prefer the least invasive successful path:
   - direct exact refspec push first
   - batched exact refspec pushes next if the history itself is acceptable
   - LFS migration or other size-handling repair only when the user allows it
4. Verify the remote ref after each important step with `git ls-remote`.

## Required Checks

Always gather these before pushing:

```powershell
git status --short --branch
git rev-parse refs/heads/<local-branch>
git ls-remote <remote> refs/heads/<remote-branch>
git log --oneline --reverse <remote>/<remote-branch>..<local-branch>
```

If the local and remote refs are already equal, stop and report that nothing needs pushing.

## Direct Push

Prefer an exact refspec push instead of relying on inferred branch names:

```powershell
git push --progress <remote> refs/heads/<local-branch>:refs/heads/<remote-branch>
```

Use this first when:

- local history is a normal fast-forward of the remote
- there is no evidence of large-object rejection
- the user has not asked for special handling

## Batch Existing History

If a full push stalls or times out, batch the already-existing local commit chain forward without inventing a replacement history.

Rules:

- Push only commits that already exist on the local branch.
- Push them in chronological order.
- Use exact refspecs like `<sha>:refs/heads/<remote-branch>`.
- Check the remote SHA after every successful step.
- Finish with a final exact branch refspec push.

Use this only when:

- the user wants the current local history preserved
- the problem is transfer size, pack time, or flaky transport
- GitHub or another remote has not rejected the history because of forbidden blob sizes

## Diagnose Oversized Objects

If the remote rejects the push, identify whether the branch contains blobs above provider policy limits.

Check the ahead-range objects first:

```powershell
git rev-list --objects <remote>/<remote-branch>..<local-branch> |
  git cat-file --batch-check="%(objecttype) %(objectname) %(objectsize) %(rest)"
```

If needed, filter for large blobs and map object IDs back to file paths. Use the playbook reference for a ready-made PowerShell pipeline.

## LFS Boundary

Do not assume batching can solve every large push. Batching helps with large transfers, but it does not bypass provider blob-size policy.

If the existing history contains blobs over the remote's hard limit:

- stop immediately if the user required exact existing history
- report the offending blob and file path
- state plainly that pushing the branch as-is is impossible to that remote without history changes

If the user only cares about the current codebase state matching remotely, and they allow history changes, Git LFS migration is a valid repair. In that case:

- create a safety branch first
- migrate the offending paths to LFS
- verify the oversized Git blobs are gone from the branch's ahead-range
- push with `--force-with-lease`, not blind `--force`

## Verification

Success means all three refs agree:

```powershell
git rev-parse refs/heads/<local-branch>
git rev-parse refs/remotes/<remote>/<remote-branch>
git ls-remote <remote> refs/heads/<remote-branch>
```

Also confirm:

```powershell
git status --short --branch
```

The expected end state is that the local branch and its remote-tracking branch are up to date, or any remaining divergence is explicitly explained.

## Reporting

When done, report:

- the final local branch SHA
- the final remote branch SHA
- whether history was preserved or rewritten
- whether Git LFS was introduced
- any remaining warnings such as files above recommended but not blocked limits
