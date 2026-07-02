document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();


    // Package form parameters
    const data = {
        N: parseFloat(document.getElementById('N').value),
        P: parseFloat(document.getElementById('P').value),
        K: parseFloat(document.getElementById('K').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        ph: parseFloat(document.getElementById('ph').value),
        rainfall: parseFloat(document.getElementById('rainfall').value)
    };

    const resultBox = document.getElementById('result');
    resultBox.className = '';
    resultBox.innerText = 'Analyzing soil parameters...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const res = await response.json();
        
        if (response.ok && res.success) {
            resultBox.className = 'success';
            resultBox.innerText = `RECOMMENDED CROP: ${res.crop}`;
        } else {
            resultBox.className = 'error';
            resultBox.innerText = `Prediction Error: ${res.error || 'Server rejection'}`;
        }
    } catch (err) {
        resultBox.className = 'error';
        resultBox.innerText = `Connection failed: ${err.message}`;
    }
    

});



const links = document.querySelectorAll(".navbar a");
const sections = document.querySelectorAll("section");

links.forEach(link => {
    link.addEventListener("click", function(e) {
        e.preventDefault();

        const target = this.getAttribute("href").substring(1);

        sections.forEach(sec => sec.classList.remove("active"));
        document.getElementById(target).classList.add("active");
    });
});