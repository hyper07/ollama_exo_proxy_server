// Statistics Page - Chart and Export Utilities

// --- Helper Functions for Exporting ---
function downloadURI(uri, name) {
    const link = document.createElement("a");
    link.download = name;
    link.href = uri;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function exportChartToPNG(chartId, filename) {
    const chart = Chart.getChart(chartId);
    if (chart) {
        const uri = chart.toBase64Image();
        downloadURI(uri, filename);
    }
}

function exportDataToCSV(headers, labels, data, filename) {
    let csvContent = "data:text/csv;charset=utf-8," + headers.join(",") + "\n";
    labels.forEach((label, index) => {
        csvContent += `"${label}",${data[index]}\n`;
    });
    const encodedUri = encodeURI(csvContent);
    downloadURI(encodedUri, filename);
}

function exportTableToCSV(tableId, filename) {
    const table = document.getElementById(tableId);
    let csvContent = "data:text/csv;charset=utf-8,";
    const rows = table.querySelectorAll("tr");
    
    rows.forEach(row => {
        const cols = row.querySelectorAll("th, td");
        const rowData = Array.from(cols).map(col => `"${col.innerText.trim().replace(/"/g, '""')}"`);
        csvContent += rowData.join(",") + "\n";
    });

    const encodedUri = encodeURI(csvContent);
    downloadURI(encodedUri, filename);
}

// --- Chart Initialization Function ---
function initializeStatisticsCharts(chartData) {
    const cs = getComputedStyle(document.documentElement);
    const primaryColor = cs.getPropertyValue('--color-primary-600').trim() || 'rgba(79, 70, 229, 1)';
    const primaryColorTransparent = primaryColor.replace(/[\d\.]+\)$/g, '0.2)');

    // Chart color palette
    const palette = [
        cs.getPropertyValue('--color-primary-600').trim(),
        '#059669', '#d946ef', '#f97316', '#dc2626', '#6b7280', '#3b82f6',
        cs.getPropertyValue('--color-primary-800').trim(),
        '#047857', '#a21caf', '#ea580c', '#b91c1c', '#4b5563', '#2563eb'
    ];

    // 1. Daily Usage Chart (Line)
    const dailyCtx = document.getElementById('dailyUsageChart')?.getContext('2d');
    if (dailyCtx && chartData.daily) {
        new Chart(dailyCtx, {
            type: 'line',
            data: {
                labels: chartData.daily.labels,
                datasets: [{
                    label: 'Requests',
                    data: chartData.daily.data,
                    borderColor: primaryColor,
                    backgroundColor: primaryColorTransparent,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: { 
                responsive: true, 
                scales: { 
                    y: { beginAtZero: true } 
                } 
            }
        });
    }

    // 2. Hourly Usage Chart (Bar)
    const hourlyCtx = document.getElementById('hourlyUsageChart')?.getContext('2d');
    if (hourlyCtx && chartData.hourly) {
        new Chart(hourlyCtx, {
            type: 'bar',
            data: {
                labels: chartData.hourly.labels,
                datasets: [{
                    label: 'Requests',
                    data: chartData.hourly.data,
                    backgroundColor: primaryColorTransparent,
                    borderColor: primaryColor,
                    borderWidth: 1
                }]
            },
            options: { 
                responsive: true, 
                scales: { 
                    y: { beginAtZero: true } 
                } 
            }
        });
    }

    // 3. Server Load Chart (Doughnut)
    const serverCtx = document.getElementById('serverLoadChart')?.getContext('2d');
    if (serverCtx && chartData.server) {
        new Chart(serverCtx, {
            type: 'doughnut',
            data: {
                labels: chartData.server.labels,
                datasets: [{
                    label: 'Requests',
                    data: chartData.server.data,
                    backgroundColor: palette,
                }]
            },
            options: { responsive: true }
        });
    }

    // 4. Model Usage Chart (Pie)
    const modelCtx = document.getElementById('modelUsageChart')?.getContext('2d');
    if (modelCtx && chartData.model) {
        new Chart(modelCtx, {
            type: 'pie',
            data: {
                labels: chartData.model.labels,
                datasets: [{
                    label: 'Requests',
                    data: chartData.model.data,
                    backgroundColor: palette,
                }]
            },
            options: { responsive: true }
        });
    }
}

