# Basics

```powershell
$PSVersionTable # prints version information about Powershell

$a | Get-Member # prints info about all member methods and properties available for the object. Also shows the object type.
```

## Single quotes vs double quotes

Single quotes assign a variable literal text, while double quotes evaluates / expands variables.

```powershell
$name = 'Bjoern'

$a = "My namy is $name" # My name is Bjoern
$b = 'My name is $name' # My name is $name
```

## Comparison operators

```powershell
-eq # equal (==)
-ne # not equal (!=)
-gt # greater than (>)
-lt # less than (<)

# string comparison
"a" -eq "A" # True - normal -eq is case insensitive
"a" -ceq "A" # False - prefix operators with "c" to make them case sensitive

# wildcards -  '*' for multiple chars, '?' for single char
"Apple" -like "A*" # True
"Apple" -like "A??le" # True

# match substring
"My name is Bjoern" -match "Bjoern" # True
```

## Cmdlets

- Cmdlets follow the structure `verb`-`object`, e.g. `Get-Member`, `Write-Host` etc.
- Aliases for cmdlets can be checked with `get-alias <cmd>`
- Help can be displayed by `help <cmd>`. Online help with `help <cmd> -online`

# Working with objects

Lists can be indexed with square brackets (0 indexing, lists start with 0)

```powershell
$files = dir    # get elements of current directory - a list
$files[0]       # get the first entry of the list i.e. the first file
```

## Sorting

Objects can be sorted with `sort-object`

```powershell
Get-ChildItem | Sort-Object     # sorts current directory by name (default)
Get-ChildItem | Sort-Object -Property length -Descending # sort by length property and descending
```

## Filtering

Lists can be filtered using `where-object`, `where` or `?`. 

Using curly brackets, the condition can be spelled using the `$_.<property>` syntax. This is called the _script block syntax_. Everything inside the curly braces is executed for every element of the list passed to `Where-Object`. The current element is identified with the `$_` variable.

```powershell
dir | where length -gt 500000   # all files with length > 500k
dir | Where-Object {$_.length -gt 500000} # block code syntax
dir | Where-Object {($_.length -gt 500000) -and ($_.Name -like "k*")} # multiple criteria combined
```

## Foreach loops

`foreach` to loop over lists.
- `{}` is used for blocks of code
- `$_` is used to for the current element
- `($_ % 2)` returns the current element mod 2 and is True if result != 0 (i.e. if there is a remainder other than 0)

```powershell
1..10 | foreach {$_ * 2} # multiplies list elements by 2

1..10 | foreach {if ($_ % 2) {"$_ is odd"}} # print all odd numbers
```

Foreach can be replaced by `%`.

## Arrays

Arrays are defined with `@()`

```powershell
$strComputers = @("Server1", "Server2", "Server3")
```

## Hashmap (dictionary)

Uses @ notation, similar to arrays.

```powershell
$empNumbers = @{"John Doe" = 112233; "Bob Jones" = 223344; "Sally Smith" = 334455}
```

Elements can be referred to by their key:

```powershell
$empNumbers["Bob Jones"]

$empNumbers["Bob Jones"] = 123456
```

Elements can be removed using the `.remove` method

```powershell
$empNumbers.remove("Bob Jones")
```

## Formatting

Format output as a table:

```powershell
get-process | format-table -property path, name, id, company

get-process | sort -Property company | format-table -property path, name, id -groupby company
```

## Saving data

- `Out-File` to output to a file
- `ConvertTo-HTML` to output to html format
- `export-csv` to save as CSV

## Importing data

Import from csv file:

```powershell
$names = import-csv C:\temp\census.csv
$names | format-table
```

# Functions and scripts

Functions are defined with the `functions` keyword.

```powershell
function Add-Numbers
{
    param([int]$num1, [int]$num2)
    # do something
    return $num1 + $num2
}

$result = Add-Numbers 1 2 # call the function with parameters
$result = Add-Numbers -num1 1 -num2 2 # with named parameters

function Get-DirInfo($dir)
{
    Get-ChildItem $dir -Recurse | Measure-Object -Property length -Sum
}

(Get-DirInfo D:\temp).sum/1GB # just get the sum property and divide it by 1GB to get GB
```

## Calling .NET functions

.NET functions can be called with square brackets (library) and then `::` before the function.

```powershell
[math]::round((Get-DirInfo d:\temp).sum/1GB)
```

## Passing parameters to a script

Pass parameters to a script using the `param` keyword at the beginning of the script.

```powershell
# paramaters
param([string]$dir="c:\") # default value

# functions
#...

# main
# use the script parameter in here by calling one of the functions with the param
```

# Examples

## Get all local user accounts, then select just those in a list and display some attributes in a table

```powershell
Get-LocalUser               # displays all local user accounts

$accounts = @("Bjoern", "Guest")
foreach ($account in $accounts)
{ get-localuser -Name $account | select Name, LastLogon}

```

## Get number of processes running for a list of processes

```powershell
$processes = @("Chrome", "svchost")
foreach ($p in $processes)
{ $plist = Get-Process -Name $p
    New-Object psobject -Property @{
    Process = $p
    Count = $plist.length}}
```

## Count elements in a group

Count the number of processes for each company. This uses `group-object` to group the list by the company.

```powershell
get-process | group-object -property company  | sort-object -property count -descending
```

## Get Active Directory group membership for a list of people and export to CSV

[https://www.reddit.com/r/PowerShell/comments/71d2rm/get_display_names_and_group_membership_for_list/](https://www.reddit.com/r/PowerShell/comments/71d2rm/get_display_names_and_group_membership_for_list/)