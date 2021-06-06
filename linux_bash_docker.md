# Linux, bash and Docker

# Resources

## Linux

[Ubuntu - command line for beginners](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview)
[Linux command line cheatsheet](https://cheatography.com/davechild/cheat-sheets/linux-command-line/)

## VIM

[Vim cheat sheet](https://vim.rtorr.com/)
[basic.vim](https://github.com/amix/vimrc/blob/master/vimrcs/basic.vim)
[Learn vim for the last time](https://danielmiessler.com/study/vim/)
[Python and vim](https://realpython.com/vim-and-python-a-match-made-in-heaven/)

[Vundle - extension manager](https://github.com/gmarik/Vundle.vim)



## Dotfiles

[Managing your dotfiles](https://www.anishathalye.com/2014/08/03/managing-your-dotfiles/)
[Using Git and Github to manage your dotfiles](http://blog.smalleycreative.com/tutorials/using-git-and-github-to-manage-your-dotfiles/)
[Getting started with Dotfiles](https://medium.com/@webprolific/getting-started-with-dotfiles-43c3602fd789)
[Git in Bash](https://git-scm.com/book/id/v2/Appendix-A%3A-Git-in-Other-Environments-Git-in-Bash)

## Bash


## WSL

[Microsoft - install WSL on Windows 10 Home](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
[Microsoft - manually downloading a WSL ditro](https://docs.microsoft.com/en-us/windows/wsl/install-manual)

## Docker


# Questions

## Linux

- Cheat sheet for common Linux commands
- Cheat sheet for vim DONE 
- Config for vim DONE
- Dotfiles - find a good repository
- Dotfiles - how to synch using git DONE

## Bash

- customize bash prompt
- find a good .bashrc

## WSL

- How to develop using WSL
- Where to store source code
- How to use Pycharm with WSL

# Docker

- What are some useful images (e.g. the network troubleshooting one)
- What is a useful data science set up using Docker?
- What are other use cases where it makes sense to use Docker
- How to deploy to Azure / Kubernetes
- 


# Set-up

## Git

[Git on WSL](https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-git)

Set up Git Credential Manager for use with WSL:

```shell
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/libexec/git-core/git-credential-manager.exe"
```


## Dotfiles

Clone my dotfile repository:

```shell
git clone https://github.com/bjrnwnklr/dotfiles.git
```

Change the `install.sh` script to executable and run it:

```shell
cd dotfiles
chmod +x ./install.sh
./install.sh
```

### Dotfiles structure

- Dotfiles are in the `dotfiles` directory, saved with their original filenames i.e. with a leading dot.
- The `install.sh` script 
    - takes a list of files named in the script, 
    - copies any existing versions in the homedir to `dotfiles.old`
    - creates symlinks from the homedir to the version in `dotfiles` 

I currently only use a `.bashrc` file and a `.profile` file. The `.profile` file looks for `~/dotfiles/bin` and appends that to the path if it exists.
