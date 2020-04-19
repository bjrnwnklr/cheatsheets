---
title: Bjoern's Git cheat sheet
author: Bjoern Winkler
date: 19-April-2020
---

# Git cheat sheet

# Initialize a new remote repo with an existing local repo from command line

    ```
    $ git remote add origin https://bjrnwnklr@dev.azure.com/bjrnwnklr/BulmaTest/_git/BulmaTest
    $ git push -u origin --all
    ```

# Migrate a Git repo from Azure DevOps to Github

1. Create a new repo on GitHub
1. Create a local clean copy of the repo (incl. history) in a new directory (as described [here](https://help.github.com/en/github/importing-your-projects-to-github/importing-a-git-repository-using-the-command-line))

```console
$ git clone --bare https://external-host.com/extuser/repo.git
```

3. Push the local cloned repo to GitHub using the 'mirror' option.

```console
$ cd repo.git
$ git push --mirror https://github.com/ghuser/repo.git
```

4. Remove the temp local repo

```console
$ cd ..
$ rm -rf repo.git
```

5. Change the remote URL of the existing local git repo to the new URL (e.g. on GitHub) (as described [here](https://help.github.com/en/github/using-git/changing-a-remotes-url))

```console
# list existing remotes
$ git remote -v

# change remote URL
$ git remote set-url origin https://github.com/USERNAME/REPOSITORY.git

# verify the URL has changed
$ git remote -v
```

# Working with branches

## Create local branch

Create a local branch, change code and merge with `master`

```
$ git checkout -b <branch-name>
# do the code changes...
$ git add --all
$ git commit -m "Commit message"
$ git checkout master
$ git merge <branch-name>
$ git push
$ git branch -d <branch-name>
```

## Create local branch and push to remote

Create a local branch, commit locally, then push the branch to master

```
$ git checkout -b <branch-name>
$ git push -u origin <branch-name>
# do your code changes...
$ git add --all
$ git commit -m "Commit message"
$ git push

$ git checkout master
$ git pull
$ git pull origin <branch-name>
$ git push
```

# Use `git log` to find changes to a specific function

With the `git log -L :<funcname>:<filename>`, you can see the version history of a specific function:

```
$ git log -L :get_regex:regex-search.py
```

# Use `git grep` to search through repository (or other files)

Useful options:

- -i: case insensitive
- -n: line numbers
- --perl-regexp: use Python like regex 
- --no-index: search through files not indexed by git e.g. in .gitignore
- -e <regex>: search for regex e.g. \d+ to find numbers

```
$ git grep -i -n --perl-regexp --no-index -e pass(word|phrase)
```

# Revert a single file back to a previous version
1) Find commit history of the file:

    ```
    $ git log <filename>
    commit 491ffa31940dc14cb300eb5307da5ce0578381cb
    Author: Bjoern Winkler <git@web.bjoern-winkler.de>
    Date:   Sat Jan 12 18:59:21 2019 +0100

    Advent 14, part 1 (finally working!)

    commit 999695de8a019b69457fa13c58b5a3bbd1a5d587
    Author: Bjoern Winkler <git@web.bjoern-winkler.de>
    Date:   Sat Jan 12 18:54:40 2019 +0100

    Advent 14 part 1 - long version
    ```

2) Revert file back to previous commit

    ```
    $ git checkout <commit-id> <filename>
    ```