---
title: JavaScript cheat sheet
author: Bjoern Winkler
date: 29-05-2019
---

# JavaScript cheat sheet

# HTML / DOM manipulation

## Hiding HTML elements, e.g. `div`  

1) set the `style="display: none"` attribute on the HTML element to be hidden, e.g.

```html
<div id="hidden" style="display: none">This is hidden!</div>
<div id="visible" style="display: block">This is visible!</div>
```

2) Use js to change the display style:

```js
document.getElementById('hidden').style.display = "block";
```

# APIs, HTTPRequest etc

## PUT request using XMLHTTPRequest

```js
var xhr = new XMLHttpRequest();
xhr.open("POST", '/sentiment', true);

//Send the proper header information along with the request
xhr.setRequestHeader("Content-Type", "application/json");

xhr.onreadystatechange = function() { // Call a function when the state changes.
    if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
        // Request finished. Do processing here.
        const response = JSON.parse(this.response);          
    }
}
xhr.send(sentimentRequest);
```

### Use `JSON.stringify` to convert string / object into a JSON object

```js
const textInput = document.getElementById('input_text').value;
if (textInput !== "") {
    const sentimentRequest = JSON.stringify({
        'text' : textInput
    });
}
```

## PUT request using `fetch()` and `async` / `await`

Details [at MDN](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch#Uploading_JSON_data)

Just using `fetch()`, `.then()` and `.catch()`

```js
var url = 'https://example.com/profile';
var data = {username: 'example'};

fetch(url, {
    method: 'POST', // or 'PUT'
    body: JSON.stringify(data), // data can be `string` or {object}!
    headers:{
        'Content-Type': 'application/json'
    }
}).then(res => res.json())
.then(response => console.log('Success:', JSON.stringify(response)))
.catch(error => console.error('Error:', error));
```

Using `async` and `await` (_still need to review how to do this!_).

Details how to convert from `.then()` using `async` and `await` [at MDN](https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Asynchronous/Async_await#Rewriting_promise_code_with_asyncawait)

```js
async function myFetch() {
    let response = await fetch('coffee.jpg');
    let myBlob = await response.blob();

    let objectURL = URL.createObjectURL(myBlob);
    let image = document.createElement('img');
    image.src = objectURL;
    document.body.appendChild(image);
}

myFetch();
```

Full example with a POST request and error handling:

```js
function submit() {
    // Get text input
    const textInput = document.getElementById('input_text').value;
    if (textInput !== "") {
        const sentimentRequest = JSON.stringify({
            'text' : textInput
        });

        // call the fetch function (similar to XMLHttpRequest)
        fetchSentiment(sentimentRequest)
            .then(response => {
                // now process the response and call the function to update the website
                updateWebsite(response);
            }) 
            .catch(error => {
                // error handling
                console.error(error);
            });
    }
}

// execute a POST request using the async / await / fetch functions
// return the json response object
async function fetchSentiment(data) {
    const config = {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: data
    }

    const response = await fetch('/sentiment', config);
    const res_json = await response.json();
    return res_json;
}
```