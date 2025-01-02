import joblib
import numpy as np
import gradio as gr

# Load the trained regression model
model = joblib.load("rf_model_selected.pkl")

# Define the prediction function
def predict_salary(US, MLinBUS, MLCloudSpend, yearsCoding, yearsMachineLearning):
    # Handle missing or None values in inputs
    if yearsCoding is None:
        yearsCoding = 0  # Default to 0 years of coding
    if MLCloudSpend is None:
        MLCloudSpend = 0  # Default to HS if not provided
    if US is False:
        USnew = 0  # Default to Gaming if not provided
    else:
        USnew = 1
    if MLinBUS is None:
        BusMLmapped = 0
    if yearsMachineLearning is None:
        yearsMachineLearning = 0;
        
        
    #ML_order = {'I do not use machine learning methods': 1, 'Under 1 year': 2, '1-2 years': 3, '2-3 years': 4, '3-4 years': 5, '4-5 years': 6, '5-10 years': 7, '10-20 years': 8, '20 or more years': 9}
    if yearsCoding == 0:
        yearsCodingMapped = 1
    elif yearsCoding < 1:
        yearsCodingMapped = 2
    elif yearsCoding <= 2:
        yearsCodingMapped = 3
    elif yearsCoding <= 5:
        yearsCodingMapped = 4
    elif yearsCoding <= 10:
        yearsCodingMapped = 5
    elif yearsCoding <= 20:
        yearsCodingMapped = 6
    else:
        yearsCodingMapped = 7
    
    
    if yearsMachineLearning == 0:
        yearsMLMapped = 1
    elif yearsMachineLearning < 1:
        yearsMLMapped = 2
    elif yearsMachineLearning <= 2:
        yearsMLMapped = 3
    elif yearsMachineLearning <= 3:
        yearsMLMapped = 4
    elif yearsMachineLearning <= 4:
        yearsMLMapped = 5
    elif yearsMachineLearning <= 5:
        yearsMLMapped = 6
    elif yearsMachineLearning <= 10:
        yearsMLMapped = 7
    elif yearsMachineLearning <= 20:
        yearsMLMapped = 8
    else:
        yearsMLMapped = 9
        
    if MLCloudSpend == 0:
        MLCloudSpendEncoded = 0
    elif MLCloudSpend <= 99:
        MLCloudSpendEncoded = 50
    elif MLCloudSpend <= 999:
        MLCloudSpendEncoded = 500
    elif MLCloudSpend <= 9999:
        MLCloudSpendEncoded = 5000
    elif MLCloudSpend <= 99999:
        MLCloudSpendEncoded = 50000
    else:
        MLCloudSpendEncoded = 100000
        
    
    
    
        
    # Encode education using one-hot encoding
    #education_mapping = {"HS": [1, 0, 0], "MS": [0, 1, 0], "PHD": [0, 0, 1]}
    #education_encoded = education_mapping.get(education, [0, 0, 0])  # Default to BS baseline

    # Encode hobby using one-hot encoding
    #hobby_mapping = {"Reading": [1, 0], "Sports": [0, 1], "Gaming": [0, 0]}
    #hobby_encoded = hobby_mapping.get(hobby, [0, 0])  # Default to Gaming baseline

    # Create input array for the model
    inputs = [USnew, MLinBUS, MLCloudSpendEncoded, yearsCodingMapped, yearsMLMapped]  # Align with model feature order
    inputs = np.array(inputs).reshape(1, -1)

    # Predict salary
    try:
        predicted_salary = model.predict(inputs)
        return f"Predicted Salary: ${predicted_salary[0]:,.2f}"
    except ValueError as e:
        return f"Error: {str(e)}"

# Title for the app
title = "<h1 style='color:red; font-weight:bold;'>Data Scientist Salary Estimator</h1>"

# Footer with the last modified date
footer = "<div style='text-align:right; font-size:small; color:gray;'>Last modified 11/20/24</div>"

# Load and display a cartoon image of a data scientist
image_url = "https://content.sportslogos.net/logos/33/777/full/north_carolina_state_wolfpack_logo_secondary_19677419.png"
image_html = f"<div style='text-align:center;'><img src='{image_url}' alt='Data Scientist' width='400'></div>"

## ml_in_business Ordinal

#degree_order = {'We are exploring ML methods (and may one day put a model into production)': 1, 
#'We use ML methods for generating insights (but do not put working models into production)': 2, 
#'We recently started using ML methods (i.e., models in production for less than 2 years)': 3, 
#'We have well established ML methods (i.e., models in production for more than 2 years)': 4, 
#'No (we do not use ML methods)': 0, 'I do not know': 0}

# Define the Gradio interface

years_coding_input = gr.Number(label="Years of Coding Experience")
years_ML_input = gr.Number(label="Years of Machine Learning Experience")
US = gr.Checkbox(label="Do you reside in the United States?")
MachLrnInBusiness = gr.Dropdown(label="What best describes the level your organization uses Machine Learning?", 
                                choices=[("We are exploring ML methods (and may one day put a model into production)", 1), 
                                         ("We use ML methods for generating insights (but do not put working models into production)", 2),
                                         ("We recently started using ML methods (i.e., models in production for less than 2 years)", 3), 
                                         ("We have well established ML methods (i.e., models in production for more than 2 years)", 4),
                                         ("No (we do not use ML methods)", 0),
                                         ("I do not know", 0)])
Money_ML_input = gr.Number(label="how much money have you spent on machine learning and/or cloud computing services at home or at work in the past 5 years")
output = gr.Textbox(label="Predicted Salary")

# Combine all components into a polished interface
interface = gr.Blocks()

with interface:
    gr.Markdown(title)
    gr.HTML(image_html)  # Add cartoon image
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Enter the details below to estimate the salary:")
            # US, MLinBUS, MLCloudSpend, yearsCoding, yearsMachineLearning
            inputs = [US, MachLrnInBusiness, Money_ML_input, years_coding_input, years_ML_input]
            gr.Interface(
                fn=predict_salary,
                inputs=inputs,
                outputs=output,
                live=True
            )
    gr.Markdown(footer)  # Add footer at the bottom

interface.launch()