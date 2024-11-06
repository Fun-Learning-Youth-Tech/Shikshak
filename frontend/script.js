async function submitQuery() {
    const query = document.getElementById('query').value;

    if (!query) {
        alert("Please enter a question!");
        return;
    }

    // Show loading message
    document.getElementById('response').innerHTML = "Loading...";

    // Prepare data to send to backend
    const data = {
        student_id: "12345",  // Static student ID, you can change this or make it dynamic
        query: query
    };

    try {
        // Send a POST request to the Flask API
        const response = await fetch('http://127.0.0.1:5000/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Display the result or handle errors
        if (response.ok) {
            document.getElementById('response').innerHTML = result.response;
        } else {
            document.getElementById('response').innerHTML = "Sorry, I couldn't get an answer.";
        }
    } catch (error) {
        document.getElementById('response').innerHTML = "Error: " + error.message;
    }
}
