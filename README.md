# tian-scripts

A collection of handy shell utilities for everyday Git workflows and development.

## Installation

Clone this repo and add it to your `PATH`:

```bash
git clone https://github.com/wangtian24/tian-scripts.git ~/workspace/tian-scripts
export PATH="$PATH:$HOME/workspace/tian-scripts"
```

You can also copy settings from `.bashrc-example` into your `~/.bashrc`.

## Scripts

| Script | Description |
|--------|-------------|
| `co`   | **Smart git checkout.** Fuzzy-match branch names, auto-stash, interactive drill-down. Sorts by most recent commit (use `-s` for alphabetical). |
| `cdo`  | **Smart cd into a `~/workspace` repo.** Ranks matches: exact > prefix > infix > regex. Multiple matches: pick by number or refine with another pattern. Requires the wrapper function from `.bashrc-example` to actually change directory. Override search root with `$CDO_WORKSPACE`. |
| `gib`  | **Create and checkout** a new branch in one step. |
| `gibdel` | **Interactive branch deletion.** Lists branches sorted by age, lets you pick by number/range, protects `main`/`master`. |
| `gmerge` | **Merge main into current branch.** Handles stashing, pulling latest main, and restoring your stash on success or failure. |
| `gpo`  | **Push current branch to origin** with a safety check for uncommitted changes. |
| `gwip` | **Quick WIP commit.** Stages tracked files and commits with a message (default: "WIP"). |

## Configuration

See `.bashrc-example` for recommended aliases and prompt setup.
