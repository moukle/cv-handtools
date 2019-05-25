#!/bin/sh

folder_path="dataset/unsorted"

# delete current dataset
echo "Deleting current dataset"
rm -rf $folder_path/*_images

scraper="pipenv run imagenetscraper"
size="--size 128,128"
quiet=""
# quiet="--quiet"

sysnet_id="n0000000"
# http://image-net.org/synset?wnid=n00000000
#                                 ^^^^^^^^^
#                                 SYNSET_ID

# hammer
echo "Downloading hammer ..."
sysnet_id="n03481172"
$scraper $sysnet_id $folder_path/hammer_images $size $quiet

# wrench
echo "Downloading wrench ..."
sysnet_id="n02680754"
$scraper $sysnet_id $folder_path/wrench_images $size $quiet

# plane
echo "Downloading plane ..."
sysnet_id="n03954731"
$scraper $sysnet_id $folder_path/plane_images $size $quiet
