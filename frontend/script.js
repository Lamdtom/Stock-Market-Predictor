document.getElementById("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const form = new FormData();
    const ticker = document.getElementById("ticker").value;
    const period = document.getElementById("period").value;
    const interval = document.getElementById("interval").value;
    const file = document.getElementById("file").files[0];

    if (ticker) form.append("ticker", ticker);
    form.append("period", period);
    form.append("interval", interval);
    if (file) form.append("file", file);

    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            body: form
        });

        const data = await response.json();
        document.getElementById("result").textContent = JSON.stringify(data, null, 2);
    } catch (err) {
        document.getElementById("result").textContent = "Error: " + err;
    }
});
