/* Small UX layer for Material for MkDocs.
   - reading progress bar
   - word count / reading time on article pages
   - external-link markers
   - keyboard shortcut "/" focuses search
*/
(function () {
  "use strict";

  var isArticle = function () {
    var content = document.querySelector(".md-content__inner");
    return content && !content.querySelector(".gb-hero");
  };

  var addProgress = function () {
    if (document.querySelector(".gb-progress")) return;
    var bar = document.createElement("div");
    bar.className = "gb-progress";
    bar.setAttribute("aria-hidden", "true");
    var fill = document.createElement("span");
    bar.appendChild(fill);
    document.body.appendChild(bar);

    var update = function () {
      var root = document.documentElement;
      var max = root.scrollHeight - root.clientHeight;
      var ratio = max > 0 ? root.scrollTop / max : 0;
      fill.style.transform = "scaleX(" + Math.min(1, Math.max(0, ratio)) + ")";
    };

    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
  };

  var addReadingTime = function () {
    if (!isArticle()) return;
    var content = document.querySelector(".md-content__inner");
    var h1 = content.querySelector("h1");
    if (!h1 || content.querySelector(".gb-page-meta")) return;

    var text = content.innerText || "";
    var words = text.trim().split(/\s+/).filter(Boolean).length;
    if (words < 120) return;

    var minutes = Math.max(1, Math.round(words / 220));
    var meta = document.createElement("p");
    meta.className = "gb-page-meta";
    meta.textContent = minutes + " min read · " + words.toLocaleString() + " words";
    h1.insertAdjacentElement("afterend", meta);
  };

  var markExternalLinks = function () {
    var content = document.querySelector(".md-content__inner");
    if (!content) return;
    content.querySelectorAll('a[href^="http"]').forEach(function (link) {
      if (link.hostname && link.hostname !== window.location.hostname) {
        link.target = "_blank";
        link.rel = "noopener";
        link.classList.add("gb-external");
      }
    });
  };

  var makeDiagramsZoomable = function () {
    var content = document.querySelector(".md-content__inner");
    if (!content) return;
    content.querySelectorAll('img[src$=".svg"]').forEach(function (img) {
      if (img.closest("a.gb-zoom")) return;
      var link = document.createElement("a");
      link.className = "gb-zoom";
      link.href = img.src;
      link.target = "_blank";
      link.rel = "noopener";
      link.setAttribute("aria-label", "Open diagram at full size");
      img.parentNode.insertBefore(link, img);
      link.appendChild(img);
    });
  };

  var makeTablesFocusable = function () {
    var content = document.querySelector(".md-content__inner");
    if (!content) return;
    content.querySelectorAll(".md-typeset__scrollwrap").forEach(function (wrap) {
      if (wrap.getAttribute("tabindex")) return;
      wrap.setAttribute("tabindex", "0");
      wrap.setAttribute("role", "region");
      wrap.setAttribute("aria-label", "Scrollable table");
    });
  };

  var labelTaskCheckboxes = function () {
    var content = document.querySelector(".md-content__inner");
    if (!content) return;
    content.querySelectorAll('.task-list-control input[type="checkbox"]').forEach(function (input) {
      if (input.getAttribute("aria-label")) return;
      var item = input.closest("li");
      var text = item ? item.innerText.trim().replace(/\s+/g, " ").slice(0, 90) : "";
      input.setAttribute("aria-label", text || "Task item");
    });
  };

  var addSearchShortcut = function () {
    if (window.__gbSearchShortcut) return;
    window.__gbSearchShortcut = true;
    document.addEventListener("keydown", function (event) {
      var tag = (document.activeElement && document.activeElement.tagName) || "";
      var typing = /INPUT|TEXTAREA|SELECT/.test(tag);
      if (event.key === "/" && !typing && !event.metaKey && !event.ctrlKey && !event.altKey) {
        var input = document.querySelector(".md-search__input");
        if (input) {
          event.preventDefault();
          input.focus();
        }
      }
    });
  };

  document$.subscribe(function () {
    addProgress();
    addReadingTime();
    markExternalLinks();
    makeDiagramsZoomable();
    makeTablesFocusable();
    labelTaskCheckboxes();
    addSearchShortcut();
  });
})();
