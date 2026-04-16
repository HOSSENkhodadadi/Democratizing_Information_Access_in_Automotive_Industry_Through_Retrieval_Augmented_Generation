document.addEventListener('DOMContentLoaded', function() {
    // Handle click on query items
    const queryItems = document.querySelectorAll('.query-item');
    queryItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const index = this.dataset.index;
            window.location.href = `/process_query?index=${index}`;
        });
    });

    // Handle custom query form submission
    const customQueryForm = document.getElementById('customQueryForm');
    if (customQueryForm) {
        customQueryForm.addEventListener('submit', function(e) {
            const customQuery = document.getElementById('customQuery').value.trim();
            if (!customQuery) {
                e.preventDefault();
                alert('Please enter a question');
            }
        });
    }
}); 