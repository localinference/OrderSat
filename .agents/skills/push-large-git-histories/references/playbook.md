# Push Playbook

## Basic State

```powershell
git status --short --branch
git rev-parse refs/heads/<local-branch>
git ls-remote <remote> refs/heads/<remote-branch>
git log --oneline --reverse <remote>/<remote-branch>..<local-branch>
```

## Exact Refspec Push

```powershell
git push --progress <remote> refs/heads/<local-branch>:refs/heads/<remote-branch>
```

## Force With Lease After Allowed History Rewrite

Fetch or read the current remote SHA first:

```powershell
$remote = (git ls-remote <remote> refs/heads/<remote-branch> | ForEach-Object { ($_ -split "`t")[0] })
git push --progress --force-with-lease=refs/heads/<remote-branch>:$remote <remote> refs/heads/<local-branch>:refs/heads/<remote-branch>
```

## Batch Existing Commits Forward

Get ordered SHAs:

```powershell
$commits = git log --format=%H --reverse <remote>/<remote-branch>..<local-branch>
```

Push checkpoints:

```powershell
foreach ($sha in $commits | Select-Object -Skip 99 -Step 100) {
  git push --progress <remote> "$sha:refs/heads/<remote-branch>"
  git ls-remote <remote> refs/heads/<remote-branch>
}
```

Finish with the branch tip:

```powershell
git push --progress <remote> refs/heads/<local-branch>:refs/heads/<remote-branch>
git ls-remote <remote> refs/heads/<remote-branch>
```

Reduce the checkpoint spacing if intermediate pushes still fail.

## Find Oversized Blobs In The Ahead Range

```powershell
git rev-list --objects <remote>/<remote-branch>..<local-branch> |
  git cat-file --batch-check="%(objecttype) %(objectname) %(objectsize) %(rest)" |
  Select-String '^blob ' |
  ForEach-Object { $_.Line } |
  Where-Object { [int64](($_ -split ' ')[2]) -ge 100000000 } |
  Sort-Object { [int64](($_ -split ' ')[2]) } -Descending
```

Use a smaller threshold like `50000000` to find files above GitHub's warning level.

## Map A Known Blob To Commits

```powershell
git log --oneline --find-object=<blob-sha> <local-branch>
git rev-list --objects <local-branch> | Select-String "<blob-sha>"
```

## LFS Migration For Specific Paths

Only do this when the user allows history changes.

```powershell
git branch backup/<branch>-before-lfs
git lfs install --local
git lfs migrate import --verbose --include="<path1>,<path2>" <local-branch>
git lfs checkout
```

Re-run the oversized-blob check before pushing.

## Final Verification

```powershell
git fetch <remote> refs/heads/<remote-branch>:refs/remotes/<remote>/<remote-branch>
git rev-parse refs/heads/<local-branch>
git rev-parse refs/remotes/<remote>/<remote-branch>
git ls-remote <remote> refs/heads/<remote-branch>
git status --short --branch
```
