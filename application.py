from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Creating object for CustomData class
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_prep'),
            math_score = request.form.get('math_score'),
            reading_score = float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))
        )
        
        #Transforming the data got from website to DataFrame 
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        #Making prediction on new input
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=result[0])
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)






