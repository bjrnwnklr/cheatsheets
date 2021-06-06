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

## Creating a new project incl. cargo.toml

```console
$ cargo new hello_cargo
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

### Debug printing

Debug output (e.g. contents of a structure) can be printed using the `{:?}` and `{:#?}` formats (implementing the fmt::Debug trait). Structs will have to implement the Debug attribute with `#[derive(Debug)]`:

```rust
#[derive(Debug)]
struct Person<'a> {
    name: &'a str,
    age: u8
}

fn main() {
    let name = "Peter";
    let age = 27;
    let peter = Person { name, age};

    println!("{:?}", peter);
    println!("{:#?}", peter);
}
```
