# Download dataset
wget https://isic-archive.s3.amazonaws.com/challenges/2024/ISIC_2024_Training_Input.zip -O dataset.zip

# Extract dataset
## Linux/MacOS
unzip -q dataset.zip -d dataset
## Window
Expand-Archive -Path "dataset.zip" -DestinationPath "dataset"