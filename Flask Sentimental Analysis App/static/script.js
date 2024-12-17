document.getElementById('submit').addEventListener('click', function() {
    const review = document.getElementById('review').value;
    
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ review: review })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = `Sentiment: ${data.sentiment}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
