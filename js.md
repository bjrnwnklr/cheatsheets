---
title: JavaScript cheat sheet
author: Bjoern Winkler
date: 27-05-2019
---

# JavaScript cheat sheet

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