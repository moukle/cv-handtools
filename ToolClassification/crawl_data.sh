#!/bin/sh

folder_path=dataset
folder_unsorted=$folder_path/unsorted
folder_split=$folder_path/split

# delete current dataset
echo "Deleting current dataset"
rm -rf $folder_unsorted/* $folder_split/*

sizepx=128
# SETTINGS
scraper="pipenv run imagenetscraper"
size="--size $sizepx,$sizepx"
quiet=""
quiet="--quiet"

# sysnet_id="n0000000"
# http://image-net.org/synset?wnid=n00000000
#                                 ^^^^^^^^^
#                                 SYNSET_ID

#### DOWNLOAD IMAGES
# hammer
echo "Downloading hammer ..."
sysnet_id="n03481172"
$scraper $sysnet_id $folder_unsorted/hammer $size $quiet

# wrench
echo "Downloading wrench ..."
sysnet_id="n02680754"
$scraper $sysnet_id $folder_unsorted/wrench $size $quiet

# plane
# echo "Downloading plane ..."
# sysnet_id="n03954731"
# $scraper $sysnet_id $folder_unsorted/plane $size $quiet

# background
# http://slazebni.cs.illinois.edu/research/uiuc_texture_dataset.zip
echo "Resizing UIUC textures ..."
mkdir $folder_unsorted/background
for file in $folder_path/uiuc_texture/*.jpg; do
    fileName=$(basename "$file")
    convert $file -resize $sizepx\x$sizepx! $folder_unsorted/background/$fileName
done

#### SPLIT IN TRAIN / VAL / TEST
echo "Splitting data in train/val/test dataset ..."
pipenv run split_folders $folder_unsorted --output $folder_split --ratio .8 .1 .1
