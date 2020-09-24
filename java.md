---
title: Java cheatsheet
author: Bjoern Winkler
date: 23-September-2020
---

# Java basics

## Running a test program from the `main` method

Basic class structure:

```java
import java.util.HashMap;
import java.util.Random;

public class CTest {

    // class variables, declare here and instantiate in the constructor
    private int size;
    private HashMap<Integer, Integer> hm;

    public CTest(int n) {
        size = n;
        Random rn = new Random();
        // initiate a class variable
        hm = new HashMap<>();

        for (int i = 0; i < size; i++) {
            hm.put(i, rn.nextInt(size));
        }
    }

    public void printMap() {
        hm.forEach((k, v) -> {
            System.out.printf("Key: %d, Val: %d%n", k, v);
        });
    }

    // main method - can run without creating any other methods etc
    public static void main(String[] args) {
        int n = 100;

        // create a new instance of the class
        CTest c = new CTest(n);
        c.printMap();
    }
}
```

# Collections

## HashMap

- A `HashMap` is similar to a Python dictionary.
- Key and value types are defined in the initiation and need to be objects not primitive types:
    - `Integer` for `int`
    - `Boolean` for `boolean`
    - `Character` for `char`
    - `Double` for `double`
    - `String` for `string`

Key points:

```java
// Defining and initiating
import java.util.HashMap;

// initiate - key and value need to be objects (e.g. Integer) not primitive types (int)
HashMap<Integer, Integer> hm = new HashMap<>();

// add an element - use put
hm.put(k, v);

// get an element
int n = hm.get(k);

// remove an element
hm.remove(k);

// remove all items
hm.clear();

// how many elements are in the map?
hm.size();

// loop through a hashmap using for loop
for (int i = 0; i < hm.size(); i++) {
    System.out.println(hm.get(i));
}

// loop through a hashmap using foreach
hm.foreach((k, v) -> {
    System.out.printf("Key: %d, value: %d%n", k, v)
});
```


# Advent Of Code 

## Structure of programs

[Example from thcathy on GitHub](https://github.com/thcathy/advent-of-code/blob/master/src/main/java/com/adventofcode/year2019/Day1Part1.java)

- Maven project
- code is in `src/main/com.bjoern_winkler.aoc2019`
- input files are in `src/main/resources/2019`
- classes are named `Day01.java`
- main method instantiates a `Day01` class and calls the `run` method (maybe do a `run1` and `run2` method for parts 1 and 2?)
- input file is referenced a static class variable
```java
final static String inputFile = "2019/day1.txt";
```
- Examples are verified as tests
```java
    @Test
    public void testcases() {
        assertEquals(2, fuelRequiredForModule(12));
        assertEquals(2, fuelRequiredForModule(14));
        Assert.assertEquals(654, fuelRequiredForModule(1969));
        Assert.assertEquals(33583, fuelRequiredForModule(100756));
    }
```

Example project.

- using a `Stream` object to 
    - read in the file and have a stream of strings (`Stream<String>`)
    - convert each line into an `Int` (`Stream<Integer>`)
    - map the fuelRequired function to each integer (`IntStream`)
    - sum it all up.

```java
public class Day01 {

    private int fuelRequired(int mass) {
        return (int)floor(mass / 3) -  2;
    }

    public static void main(String[] args) throws IOException, URISyntaxException {
        Day01 solution = new Day01();
        solution.run();
    }

    private void run() throws IOException, URISyntaxException {
        // get a stream of the file
        AoCFileOps aoc = new AoCFileOps("2019");
        int part1 = aoc.readAsStream(1)
                .map(num -> Integer.parseInt(num))
                .mapToInt(this::fuelRequired)
                .sum();

        System.out.printf("Part 1: %d", part1);
    }

    @Test
    public void testcases() {
        Assertions.assertEquals(2, fuelRequired(12));
        Assertions.assertEquals(2, fuelRequired(14));
        Assertions.assertEquals(654, fuelRequired(1969));
        Assertions.assertEquals(33583, fuelRequired(100756));
    }

}
```

## Reading a file

[how-to-read-a-file-line-by-line-in-java](https://attacomsian.com/blog/how-to-read-a-file-line-by-line-in-java)
[Baeldung - reading files in java](https://www.baeldung.com/reading-file-in-java)
[Read a file from a resources folder](https://mkyong.com/java/java-read-a-file-from-resources-folder/)


### Scanner including regular expressions

[Scanner - Oracle](https://docs.oracle.com/javase/8/docs/api/java/util/Scanner.html)

Getting long numbers from a file:
```java
Scanner sc = new Scanner(new File("myNumbers"));
while (sc.hasNextLong()) {
    long aLong = sc.nextLong();
}
```

```java
String input = "1 fish 2 fish red fish blue fish";
Scanner s = new Scanner(input);
s.findInLine("(\\d+) fish (\\d+) fish (\\w+) fish (\\w+)");
MatchResult result = s.match();
for (int i=1; i<=result.groupCount(); i++)
    System.out.println(result.group(i));
s.close();
```

### Stream

[Stream - Oracle](https://docs.oracle.com/javase/8/docs/api/java/util/stream/Stream.html)
