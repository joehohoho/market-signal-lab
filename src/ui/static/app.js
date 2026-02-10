/**
 * Market Signal Lab - Client-side helpers
 *
 * Minimal JS for Plotly chart rendering and HTMX event handling.
 */

/**
 * Render a Plotly chart from JSON data into the target element.
 *
 * @param {string} elementId - DOM element ID to render into
 * @param {object} chartJson - Plotly JSON with data and layout keys
 */
function renderChart(elementId, chartJson) {
    var el = document.getElementById(elementId);
    if (!el || !chartJson) return;

    var data = chartJson.data || [];
    var layout = chartJson.layout || {};

    // Ensure responsive sizing
    layout.autosize = true;

    var config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: [
            'sendDataToCloud',
            'lasso2d',
            'select2d',
            'autoScale2d'
        ],
        displaylogo: false,
    };

    Plotly.newPlot(el, data, layout, config);
}

/**
 * Re-render charts after HTMX content swaps.
 * Plotly charts inside swapped content need to be re-initialised.
 */
document.addEventListener('htmx:afterSwap', function(event) {
    // Look for any script tags in the swapped content and execute them
    var scripts = event.detail.target.querySelectorAll('script');
    scripts.forEach(function(script) {
        var newScript = document.createElement('script');
        newScript.textContent = script.textContent;
        document.body.appendChild(newScript);
        document.body.removeChild(newScript);
    });
});

/**
 * Add a loading state to forms during HTMX requests.
 */
document.addEventListener('htmx:beforeRequest', function(event) {
    var form = event.detail.elt;
    if (form && form.tagName === 'FORM') {
        var btn = form.querySelector('button[type="submit"]');
        if (btn) {
            btn.disabled = true;
            btn.dataset.originalText = btn.textContent;
            btn.textContent = 'Running...';
            btn.classList.add('opacity-75');
        }
    }
});

document.addEventListener('htmx:afterRequest', function(event) {
    var form = event.detail.elt;
    if (form && form.tagName === 'FORM') {
        var btn = form.querySelector('button[type="submit"]');
        if (btn && btn.dataset.originalText) {
            btn.disabled = false;
            btn.textContent = btn.dataset.originalText;
            btn.classList.remove('opacity-75');
        }
    }
});
