import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

threshold_min = 84.56         # training data min
threshold_max = 11324682.73   # training data max

app = FastAPI(docs_url=None, redoc_url=None)
app.title = "🥑 Avocado Total Sales Volume Prediction"
app.description = "This API predicts the next total sales volume of avocados based on the last 30 values provided by the user."
app.mount("/static", StaticFiles(directory="static"), name="static")

model = load_model('LSTM_model.h5', compile=False, custom_objects={'Orthogonal': Orthogonal})
scaler = joblib.load("scaler.pkl")

def predict_price(last_30_values):
    input_array = np.array(last_30_values).reshape(-1, 1)
    scaled_input = scaler.transform(input_array)
    model_input = np.expand_dims(scaled_input, axis=0)
    scaled_prediction = model.predict(model_input)
    scaled_prediction = np.clip(scaled_prediction, 0, 1)
    predicted_value = scaler.inverse_transform(scaled_prediction)
    return float(predicted_value[0][0])

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>🥑 Avocado Sales Volume Prediction - Avocado Sales</title>
        <style>
          body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: linear-gradient(135deg, #e0f8d8, #b2f0a1);
            color: #000;
            border-radius: 12px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
          }
          h1 {
            text-align: center;
            color: #1a3c15;
            text-shadow: 1px 1px 3px #2c6b1a;
          }
          p.instruction {
            color: #1a3c15;
            text-align: center;
            font-weight: 500;
            margin-bottom: 10px;
          }
          textarea {
            width: 100%;
            height: 140px;
            font-size: 16px;
            padding: 10px;
            border-radius: 8px;
            border: 2px solid #3d7d0b;
            resize: vertical;
            box-sizing: border-box;
          }
          button {
            background-color: #4caf50;
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            display: block;
            margin: 20px auto 0 auto;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: background-color 0.3s ease;
          }
          button:hover {
            background-color: #3a9b00;
          }
          #result {
            margin-top: 25px;
            font-weight: 700;
            font-size: 20px;
            text-align: center;
            color: #1a3c15;
            text-shadow: 1px 1px 2px #264d00;
          }
          .error {
            color: #ffbaba;
            background-color: #69000088;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
            margin-top: 20px;
            font-weight: 600;
          }
          img {
            display: block;
            margin: 0 auto 25px auto;
            max-width: 200px;
            border-radius: 20px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
          }
          .charts-container {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 40px;
          }
          canvas {
            max-width: 600px;
            max-height: 500px;
            width: 100%;
            height: 500px;
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
          }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      </head>
      <body>
        <img src="/static/cartoon_avocado.jpeg" alt="Avocado" />
        <h1>Avocado Sales Volume Prediction</h1>
        <p class="instruction">Please enter exactly 30 past sales values separated by commas:</p>
        <textarea id="inputValues" placeholder="e.g. 12345, 12500, 12750, ..."></textarea><br/>
        <button onclick="predict()">Predict</button>
        <div id="result"></div>

        <div class="charts-container">
          <div>
            <h3 style="color: #1a3c15; text-align: center; margin-top: 40px;">Linear Scale</h3>
            <canvas id="linearChart"></canvas>
          </div>
          <div>
            <h3 style="color: #1a3c15; text-align: center; margin-top: 40px;">Logarithmic Scale</h3>
            <canvas id="logChart"></canvas>
          </div>
        </div>

        <script>
          let linearChart = null;
          let logChart = null;

          function renderChart(pastValues, predictedValue, canvasId, yScaleType = 'linear') {
            const ctx = document.getElementById(canvasId).getContext('2d');

            // For log scale, replace <= 0 values with 1 (or small positive number) to avoid errors
            let safePastValues = pastValues;
            let safePredictedValue = predictedValue;
            if (yScaleType === 'logarithmic') {
              safePastValues = pastValues.map(v => (v <= 0 ? 1 : v));
              safePredictedValue = predictedValue <= 0 ? 1 : predictedValue;
            }

            const labels = [];
            for(let i=1; i<=pastValues.length; i++) {
                labels.push('T-' + (pastValues.length - i + 1));
            }
            labels.push('T+1');

            const data = {
                labels: labels,
                datasets: [
                  {
                    label: 'Past Sales Volume',
                    data: safePastValues,
                    borderColor: 'rgba(54, 162, 235, 1)', // blue line
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                  },
                  {
                    label: 'Predicted Sales Volume',
                    data: new Array(safePastValues.length).fill(null).concat([safePredictedValue]),
                    borderColor: 'rgba(255, 99, 132, 1)', // red point
                    backgroundColor: 'rgba(255, 99, 132, 0.8)',
                    fill: false,
                    tension: 0,
                    showLine: false,
                    pointRadius: 10,
                    pointHoverRadius: 14,
                    pointStyle: 'rectRounded'
                  }
                ]
            };

            const config = {
                type: 'line',
                data: data,
                options: {
                  responsive: true,
                  plugins: {
                    legend: {
                      labels: { font: { size: 14 } }
                    },
                    tooltip: {
                      callbacks: {
                        label: ctx => ctx.parsed.y.toLocaleString()
                      }
                    }
                  },
                  scales: {
                    y: {
                      type: yScaleType,
                      min: yScaleType === 'logarithmic' ? 1 : undefined,
                      ticks: {
                        font: { size: 14 },
                        callback: value => value.toLocaleString()
                      }
                    },
                    x: {
                      ticks: { font: { size: 14 } }
                    }
                  }
                }
            };

            // Destroy previous chart instance before creating new
            if(canvasId === 'linearChart' && linearChart) {
              linearChart.destroy();
            }
            if(canvasId === 'logChart' && logChart) {
              logChart.destroy();
            }

            const newChart = new Chart(ctx, config);

            if(canvasId === 'linearChart') linearChart = newChart;
            if(canvasId === 'logChart') logChart = newChart;
          }

          async function predict() {
            const resultDiv = document.getElementById('result');
            resultDiv.textContent = '';
            let raw = document.getElementById('inputValues').value;
            let values = raw.split(',').map(v => v.trim()).filter(v => v !== '');
            if (values.length !== 30) {
              resultDiv.innerHTML = '<span class="error">Error: Please enter exactly 30 values.</span>';
              return;
            }
            let nums = [];
            for(let v of values) {
              let n = parseFloat(v);
              if (isNaN(n)) {
                resultDiv.innerHTML = '<span class="error">Error: All values must be valid numbers.</span>';
                return;
              }
              nums.push(n);
            }
            try {
              const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ last_30_values: nums })
              });
              if (!response.ok) throw new Error('Network response was not ok');
              const data = await response.json();
              resultDiv.textContent = 'Predicted next total volume: ' + data.predicted_total_volume.toFixed(2);
              if (data.warning) {
                resultDiv.innerHTML += `<br/><span class="error">${data.warning}</span>`;
              }
              renderChart(nums, data.predicted_total_volume, 'linearChart', 'linear');
              renderChart(nums, data.predicted_total_volume, 'logChart', 'logarithmic');
            } catch (err) {
              resultDiv.innerHTML = '<span class="error">Error: ' + err.message + '</span>';
            }
          }
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)


class InputData(BaseModel):
    last_30_values: list[float]

@app.post("/predict")
async def predict(data: InputData):
    input_values = np.array(data.last_30_values)
    
    warning = None
    if np.min(input_values) < threshold_min or np.max(input_values) > threshold_max:
        warning = "Warning: These values may be out of training range."
    
    prediction = predict_price(data.last_30_values)
    
    response = {"predicted_total_volume": prediction}
    if warning:
        response["warning"] = warning
    
    return response