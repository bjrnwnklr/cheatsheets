# Rust

[Main Rust page](https://www.rust-lang.org/)

[Getting started](https://www.rust-lang.org/learn/get-started)

[Rust on Github](https://github.com/rust-lang)

[The Rust book](https://doc.rust-lang.org/book/)

[Rust by example](https://doc.rust-lang.org/stable/rust-by-example/)

# Installation

Install Rustup, the Rust installer and version management tool.


# Rustup commands

Update Rust:

```console
rustup update
```

# Cargo

Cargo is the Rust build tool and package manager.

- build your project with `cargo build`
- run your project with `cargo run`
- test your project with `cargo test`
- build documentation for your project with `cargo doc`
- publish a library to crates.io with `cargo publish`

To test that you have Rust and Cargo installed, you can run this in your terminal of choice:

```console
cargo --version
```

## Compiling a program manually

Use `rustc` to compile a program manually without building a project.

```console
rustc hello-world.rs
```

# Syntax

## Printing

- `format!`: write formatted text to `String`
- `print!`: same as `format!` but the text is printed to the console (io::stdout).
- `println!`: same as `print!` but a newline is appended.
- `eprint!`: same as `format!` but the text is printed to the standard error (io::stderr).
- `eprintln!`: same as eprint!but a newline is appended.