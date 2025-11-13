// static/js/task-poll.js

function pollTaskStatus(task_id, status_element_id, result_element_id) {
    const statusElement = document.getElementById(status_element_id);
    const resultElement = document.getElementById(result_element_id);

    const interval = setInterval(() => {
        fetch(`/task_status/${task_id}/`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'SUCCESS') {
                    clearInterval(interval);
                    statusElement.innerText = 'Status: ' + data.status;
                    resultElement.innerText = 'Result: ' + JSON.stringify(data.result);
                } else if (data.status === 'FAILURE') {
                    clearInterval(interval);
                    statusElement.innerText = 'Status: ' + data.status;
                    resultElement.innerText = 'Result: ' + JSON.stringify(data.result);
                } else {
                    statusElement.innerText = 'Status: ' + data.status;
                }
            })
            .catch(error => {
                clearInterval(interval);
                console.error('Error polling task status:', error);
                statusElement.innerText = 'Error polling task status.';
            });
    }, 2000);
}
