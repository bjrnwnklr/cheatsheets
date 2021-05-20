---
title: Bjoern's Git cheat sheet
author: Bjoern Winkler
date: 19-April-2020
---

# Git cheat sheet

# Initialize a directory with a git repo

Directory with existing code.

```shell
$ git init
```
You might then have to connect to a remote repo, see next tip.

# Initialize a new remote repo with an existing local repo from command line

    ```
    $ git remote add origin https://bjrnwnklr@dev.azure.com/bjrnwnklr/BulmaTest/_git/BulmaTest
    $ git push -u origin --all
    ```

# Clone a new repo from github

```shell
$ git clone https://github.com/bjrnwnklr/algs1.git
```

# Move existing code (without .git) into a new github repo

1. Rename existing folder to a temp name (assuming the github repo should have the same name)
1. Create a new github repo on github.com (incl. a README and .gitignore file)
1. Clone the new repo
    ```shell
    $ git clone https://github.com/bjrnwnklr/algs1.git
    ```
1. Copy the files from the temp directory into the new local repo directory
1. Add / commit / push
1. Delete the temp directory (old directory with files)



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

# Updating personal access token e.g. after expiring

On Windows:
- generate a new access token from github.com > Settings > Developer Settings > Personal Access Tokens
- grant at least repo, workflow and gist access rights
- copy the token
- from the command line, run

```shell
$ git config --global credential.helper wincred
```

and paste the new token in there to authenticate.

[Creating a personal access token](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token)
[Caching your GitHub credentials in Git](https://docs.github.com/en/github/getting-started-with-github/getting-started-with-git/caching-your-github-credentials-in-git)