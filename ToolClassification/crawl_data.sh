#!/bin/sh

# SETTINGS
folder_path=data
folder_training=$folder_path/training
sizepx=224

# DELETE CURRENT DATASET
echo "Deleting current dataset"
rm -rf $folder_training/*

# CRAWL IMAGES
# http://image-net.org/synset?wnid=n00000000
#                                 ^^^^^^^^^
#                                 SYNSET_ID

crawl="pipenv run imagenetscraper --size $sizepx,$sizepx --quiet"

# hammer
echo "Downloading hammer ..."
$crawl n03481172 $folder_training/hammer

# wrench
echo "Downloading wrench ..."
$crawl n02680754 $folder_training/wrench

# plane
echo "Downloading plane ..."
$crawl n03954731 $folder_training/plane

# background - http://slazebni.cs.illinois.edu/research/uiuc_texture_dataset.zip
echo "Resizing UIUC textures ..."
mkdir $folder_training/background
for file in $folder_path/uiuc_texture/*.jpg; do
    fileName=$(basename "$file")
    convert $file -resize $sizepx\x$sizepx! $folder_training/background/$fileName
done