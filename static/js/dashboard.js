document.addEventListener("DOMContentLoaded", () => {
  const chartCanvas = document.getElementById("traffic-chart");
  if (chartCanvas && window.trafficChartData && window.Chart) {
    new Chart(chartCanvas, {
      type: "doughnut",
      data: {
        labels: window.trafficChartData.labels,
        datasets: [{
          data: window.trafficChartData.values,
          backgroundColor: ["#36d6dc", "#ff6e78"],
          borderColor: "#0d1b2b",
          borderWidth: 5,
          hoverOffset: 7
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "72%",
        plugins: {
          legend: {
            position: "right",
            labels: { color: "#a8bac9", usePointStyle: true, pointStyle: "circle", padding: 18, font: { size: 10 } }
          }
        }
      }
    });
  }

  const filter = document.getElementById("prediction-filter");
  const analysisTable = document.getElementById("analysis-table");
  if (filter && analysisTable) {
    filter.addEventListener("change", () => {
      analysisTable.querySelectorAll("tbody tr").forEach((row) => {
        row.hidden = filter.value !== "all" && row.dataset.prediction !== filter.value;
      });
    });
  }

  const search = document.getElementById("threat-search");
  const threatTable = document.getElementById("threat-table");
  if (search && threatTable) {
    search.addEventListener("input", () => {
      const query = search.value.toLowerCase();
      threatTable.querySelectorAll("tbody tr").forEach((row) => {
        row.hidden = !row.textContent.toLowerCase().includes(query);
      });
    });
  }

  const fileInput = document.getElementById("dataset");
  const fileName = document.getElementById("file-name");
  if (fileInput && fileName) {
    fileInput.addEventListener("change", () => {
      fileName.textContent = fileInput.files.length ? fileInput.files[0].name : "or select a file from your computer";
    });
  }
});
