# wineML
A simple ML model using a dataset to predict the quality of wine based on quantitative features like the wine’s “fixed acidity”, “pH”, “residual sugar”, and so on
[https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html]

### Experiments
- Model experiments being trained (configured as CI via Github Actions)
- Model metrics are tracked on a cloud MLFlow server: http://ec2-65-2-9-37.ap-south-1.compute.amazonaws.com:5000/ hosted on AWS EC2
- The model artifacts are pushed to a S3 bucket
