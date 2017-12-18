mkdir -p raw_data
cd raw_data

echo "Downloading train.zip..."
wget https://ams3.digitaloceanspaces.com/datasets/train.zip
echo "Unzipping train.zip..."
unzip train.zip
echo "Deleting train.zip..."
rm -rf ./train.zip

echo "Downloading test.zip..."
wget https://ams3.digitaloceanspaces.com/datasets/test.zip
echo "Unzipping test.zip..."
unzip test.zip
echo "Deleting test.zip..."
rm -rf ./test.zip
