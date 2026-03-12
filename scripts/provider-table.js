(function () {
  var LEGEND = {
    Responses:
      "Supports the <a href='https://www.openresponses.org/' target='_blank'>OpenResponses</a> specification for agentic AI systems.",
    Completion: "Supports the completions endpoint.",
    Streaming: "Can stream completion results back as an iterator.",
    Reasoning:
      "Can return reasoning traces alongside the assistant message. Does not indicate whether the provider offers separate reasoning models.",
    Image:
      "Supports passing image data for vision capabilities (OpenAI-compatible <code>image_data</code> parameter).",
    Embedding: "Supports the embedding endpoint.",
    "List Models": "Supports listing available models programmatically via <code>list_models()</code>.",
    Batch: "Supports batch completion requests.",
  };

  document.addEventListener("DOMContentLoaded", function () {
    var input = document.getElementById("provider-search");
    if (!input) return;

    var content = document.querySelector(".sl-markdown-content");
    if (!content) return;

    var table = content.querySelector("table");
    if (!table) return;

    // Parse headers
    var headerCells = Array.from(table.querySelectorAll("thead th"));
    if (headerCells.length === 0) {
      headerCells = Array.from(table.querySelector("tr").querySelectorAll("th"));
    }
    var headers = headerCells.map(function (th) {
      return th.textContent.trim();
    });

    // Parse rows into provider objects
    var providers = [];
    var dataRows = Array.from(table.querySelectorAll("tbody tr"));
    if (dataRows.length === 0) {
      dataRows = Array.from(table.querySelectorAll("tr")).filter(function (row) {
        return row.querySelector("td");
      });
    }

    dataRows.forEach(function (row) {
      var cells = Array.from(row.querySelectorAll("td"));
      var nameCell = cells[0];
      var link = nameCell ? nameCell.querySelector("a") : null;

      var provider = {
        name: nameCell ? nameCell.textContent.trim() : "",
        url: link ? link.href : null,
        config: [],
        features: [],
        searchText: row.textContent.toLowerCase(),
      };

      for (var i = 1; i <= 2 && i < cells.length; i++) {
        provider.config.push({
          label: headers[i],
          value: cells[i] ? cells[i].textContent.trim() : "",
        });
      }

      for (var j = 3; j < headers.length && j < cells.length; j++) {
        provider.features.push({
          label: headers[j].replace(/\(Completions\)/g, "").replace(/\s+/g, " ").trim(),
          supported: cells[j].textContent.indexOf("\u2705") !== -1,
        });
      }

      providers.push(provider);
    });

    // Hide the original table
    var tableEl = table.parentElement !== content && table.parentElement.tagName === "DIV" ? table.parentElement : table;
    tableEl.style.display = "none";

    // Update provider count in intro text
    var intro = content.querySelector("p");
    if (intro && intro.innerHTML.indexOf("{n}") !== -1) {
      intro.innerHTML = intro.innerHTML.replace("{n}", providers.length);
    }

    // Build provider grid
    var grid = document.createElement("div");
    grid.className = "provider-grid";

    var cards = [];
    providers.forEach(function (provider) {
      var card = document.createElement("button");
      card.className = "provider-card";
      card.type = "button";

      var supported = provider.features.filter(function (f) {
        return f.supported;
      }).length;
      var total = provider.features.length;

      // Dots row showing feature support at a glance
      var dots = provider.features
        .map(function (f) {
          return '<span class="provider-card-dot ' + (f.supported ? "dot-yes" : "dot-no") + '"></span>';
        })
        .join("");

      card.innerHTML =
        '<span class="provider-card-name">' +
        provider.name +
        "</span>" +
        '<span class="provider-card-dots">' +
        dots +
        "</span>" +
        '<span class="provider-card-count">' +
        supported +
        "/" +
        total +
        " features</span>";

      card.addEventListener("click", function () {
        showModal(provider);
      });

      grid.appendChild(card);
      cards.push({ el: card, provider: provider });
    });

    var searchContainer = input.closest(".provider-search-container");
    searchContainer.parentNode.insertBefore(grid, searchContainer.nextSibling);

    // No results
    var noResults = document.createElement("p");
    noResults.textContent = "No providers match your search.";
    noResults.className = "provider-no-results";
    noResults.style.display = "none";
    grid.parentNode.insertBefore(noResults, grid.nextSibling);

    // Modal
    var overlay = document.createElement("div");
    overlay.className = "provider-modal-overlay";
    overlay.style.display = "none";
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) closeModal();
    });
    var modalEl = document.createElement("div");
    modalEl.className = "provider-modal";
    overlay.appendChild(modalEl);
    document.body.appendChild(overlay);

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closeModal();
    });

    function closeModal() {
      overlay.style.display = "none";
      document.body.style.overflow = "";
    }

    function showModal(provider) {
      var html = '<button class="provider-modal-close" aria-label="Close">&times;</button>';

      html += '<h2 class="provider-modal-title">';
      html += provider.url
        ? '<a href="' + provider.url + '" target="_blank" rel="noopener">' + provider.name + " &#8599;</a>"
        : provider.name;
      html += "</h2>";

      // Features
      html += '<div class="provider-modal-features">';
      provider.features.forEach(function (f) {
        html +=
          '<div class="provider-modal-feature ' +
          (f.supported ? "supported" : "unsupported") +
          '">' +
          '<span class="provider-modal-dot ' +
          (f.supported ? "dot-yes" : "dot-no") +
          '"></span>' +
          "<span>" +
          f.label +
          "</span></div>";
      });
      html += "</div>";

      // Configuration
      html += '<h3>Configuration</h3><dl class="provider-modal-config">';
      provider.config.forEach(function (c) {
        html += "<dt>" + c.label + "</dt>";
        html += "<dd>" + (c.value || "\u2014") + "</dd>";
      });
      html += "</dl>";

      // Legend
      html += '<details class="provider-modal-legend"><summary>What do these features mean?</summary><dl>';
      provider.features.forEach(function (f) {
        if (LEGEND[f.label]) {
          html += "<dt>" + f.label + "</dt><dd>" + LEGEND[f.label] + "</dd>";
        }
      });
      html += "</dl></details>";

      modalEl.innerHTML = html;
      overlay.style.display = "flex";
      document.body.style.overflow = "hidden";

      modalEl.querySelector(".provider-modal-close").addEventListener("click", closeModal);
    }

    // Search
    input.addEventListener("input", function () {
      var query = input.value.toLowerCase().trim();
      var visibleCount = 0;

      cards.forEach(function (item) {
        if (!query) {
          item.el.style.display = "";
          visibleCount++;
          return;
        }
        var match = item.provider.searchText.includes(query);
        item.el.style.display = match ? "" : "none";
        if (match) visibleCount++;
      });

      noResults.style.display = visibleCount === 0 ? "block" : "none";
    });
  });
})();
