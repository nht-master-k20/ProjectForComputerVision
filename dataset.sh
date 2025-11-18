# Download dataset
wget -P dataset/ https://isic-archive.s3.amazonaws.com/challenges/2024/ISIC_2024_Training_Input.zip
wget -P dataset/ https://isic-archive.s3.amazonaws.com/challenges/2024/ISIC_2024_Training_GroundTruth.csv

# Extract dataset
unzip -q dataset/ISIC_2024_Training_Input.zip -d dataset/
rm dataset/ISIC_2024_Training_Input.zip