# Git cheat sheet

### Revert a single file back to a previous version
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