FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000

# Train model first (creates model_params.pkl
RUN python train_model.py

# Then Run Flask app
CMD ["python", "app.py"]
