(function () {
    const logContainer = document.getElementById('attendance-log');
    const tbody = document.getElementById('attendance-log-body');

    if (!logContainer || !tbody) {
        return;
    }

    const feedUrl = logContainer.dataset.feedUrl;
    const statusBadge = (label, style) => `<span class="badge ${style}">${label}</span>`;

    const renderRows = (events) => {
        if (!events || events.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-muted">No recent recognition events.</td></tr>';
            return;
        }

        const rows = events.map((event) => {
            const timestamp = new Date(event.timestamp).toLocaleString();
            const username = event.username || 'Unknown';
            const direction = event.direction || '—';
            let status = 'Pending';
            let statusStyle = 'bg-secondary';
            let liveness = 'Not checked';
            let livenessStyle = 'bg-secondary';
            let confidence = '—';

            if (event.event_type === 'outcome') {
                status = event.accepted ? 'Accepted' : 'Rejected';
                statusStyle = event.accepted ? 'bg-success' : 'bg-danger';
                if (event.confidence !== null && event.confidence !== undefined) {
                    confidence = `${(event.confidence * 100).toFixed(1)}%`;
                } else if (event.distance !== null && event.threshold !== null) {
                    confidence = `dist ${event.distance.toFixed(3)} / ${event.threshold.toFixed(3)}`;
                }
            } else {
                status = event.successful ? 'Recognized' : 'Attempted';
                statusStyle = event.successful ? 'bg-success' : 'bg-secondary';
                if (event.liveness === 'failed') {
                    liveness = 'Failed';
                    livenessStyle = 'bg-warning text-dark';
                } else if (event.liveness === 'passed') {
                    liveness = 'Passed';
                    livenessStyle = 'bg-success';
                }
                if (event.error) {
                    status = 'Error';
                    statusStyle = 'bg-danger';
                    confidence = event.error;
                }
            }

            return `
                <tr>
                    <td class="text-nowrap">${timestamp}</td>
                    <td>${username}</td>
                    <td class="text-capitalize">${direction}</td>
                    <td>${statusBadge(status, statusStyle)}</td>
                    <td>${statusBadge(liveness, livenessStyle)}</td>
                    <td>${confidence}</td>
                </tr>
            `;
        });

        tbody.innerHTML = rows.join('');
    };

    const fetchFeed = async () => {
        try {
            const response = await fetch(feedUrl, { credentials: 'same-origin' });
            if (!response.ok) {
                throw new Error(`Request failed with status ${response.status}`);
            }
            const payload = await response.json();
            renderRows(payload.events || []);
        } catch (error) {
            tbody.innerHTML = `<tr><td colspan="6" class="text-center text-danger py-4">Unable to load live log (${error.message}).</td></tr>`;
        }
    };

    fetchFeed();
    setInterval(fetchFeed, 5000);
})();
