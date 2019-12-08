echo jpg 512 train nocrop
python cleanup.py --resize 0 --test 0 --tiff 0 --crop 0 --n_workers 8
echo jpg 256 train nocrop
python cleanup.py --resize 1 --test 0 --tiff 0 --crop 0 --n_workers 8
echo jpg 512 test nocrop
python cleanup.py --resize 0 --test 1 --tiff 0 --crop 0 --n_workers 8

echo tiff 512 train nocrop
python cleanup.py --resize 0 --test 0 --tiff 1 --crop 0 --n_workers 8
echo tiff 256 train nocrop
python cleanup.py --resize 1 --test 0 --tiff 1 --crop 0 --n_workers 8

