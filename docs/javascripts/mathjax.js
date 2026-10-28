window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

/* MathJax 3 cannot line-break inline math, so on a phone a long inline formula
   can be wider than its paragraph and push the whole page sideways. Mark only
   those formulas; CSS lets them scroll in place. Re-checked on resize. */
var fitWideInlineMath = function () {
  document.querySelectorAll(".md-typeset span.arithmatex").forEach(function (span) {
    span.classList.remove("gb-math-scroll");
    var box = span.parentElement;  // nearest block ancestor; inline ones report width 0
    while (box && getComputedStyle(box).display.indexOf("inline") === 0) box = box.parentElement;
    if (box && span.getBoundingClientRect().width > box.clientWidth + 1) {
      span.classList.add("gb-math-scroll");
    }
  });
};

var fitTimer = null;
window.addEventListener("resize", function () {
  clearTimeout(fitTimer);
  fitTimer = setTimeout(fitWideInlineMath, 150);
});

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise().then(fitWideInlineMath);
});
