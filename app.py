from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import joblib
import io
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the model
model = joblib.load("my_model.pkl", mmap_mode='r')

# def plot_predictions(self, df, predictions):
#     plt.figure(figsize=(10, 5))
#     plt.plot(df['Day'], predictions, label='Predicted Sales')
#     plt.xlabel('Day')
#     plt.ylabel('Sales')
#     plt.title('Predicted Sales Over Time')
#     plt.legend()
#     plt.savefig('static/predictions_plot.png')
#     plt.close()

# Define a route for prediction
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get the input data from requests
    input_data = request.get_json(force=True)
    print("Input data:", input_data)

    # Convert "Yes" or "No" to 1 or 0
    input_data["IsHoliday"] = 1 if input_data["IsHoliday"] == "Yes" else 0
    input_data["IsWeekend"] = 1 if input_data["IsWeekend"] == "Yes" else 0
    input_data["IsPromo"] = 1 if input_data["IsPromo"] == "Yes" else 0
    input_data["Open"] = 1 if input_data["Open"] == "Yes" else 0

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data], index=[0])
    print("Input DataFrame:", input_df)

    # Make prediction
    prediction = model.predict(input_df)
    # Plot prediction
    #plot_predictions(input_df,prediction)
    
    
    # # Plotting predicted values
    # fig, ax = plt.subplots()
    # ax.plot(prediction, label='Predicted Sales Amount')
    # ax.plot(prediction * 0.8, label='Predicted Number of Customers') 
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Amount')
    # ax.legend()

    # # Save plot to bytes buffer
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # plt.close()

    return render_template('index.html', prediction=prediction[0])

@app.route('/upload', methods=['POST'])
def upload():
    # Get input data from the request
    input_data = request.get_json(force=True)
    print("Input data:", input_data)

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data], index=[0])
    print("Input DataFrame:", input_df)

    # Make prediction
    predictions = model.predict(input_df)
    
    # Create a CSV buffer
    csv_buffer = io.StringIO()
    input_df['Prediction'] = predictions
    input_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        attachment_filename='predictions.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)
