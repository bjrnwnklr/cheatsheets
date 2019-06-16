---
title: Node.js / nvm / npm cheatsheet
author: Bjoern Winkler
date: 16-June-2019
---

# Managing versions with `nvm`

`nvm` is a node.js version manager for Windows.

### Checking version of installed nvm

Show the installed versions:

    > nvm list

Show all available versions:

    > nvm list available

### Updating nvm

Install a specific version and re-install all packages from previous versions:

    > nvm install <new version e.g. 12.4.0> --reinstall-packages-from=<old version e.g. 12.1.0>
    > nvm use 12.4.0
    > nvm uninstall 12.1.0

# Using `npm`



