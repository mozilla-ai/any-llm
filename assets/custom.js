// Detect if we're on the tracing page and add a CSS class
document.addEventListener('DOMContentLoaded', function() {
    // Check if the current URL contains 'tracing'
    if (window.location.pathname.includes('/tracing/')) {
        document.body.classList.add('tracing-page');
    }
});
