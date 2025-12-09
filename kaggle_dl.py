import kagglehub

# Download latest version
path = kagglehub.dataset_download("cindybtari/id-card-classification")

print("Path to dataset files:", path)