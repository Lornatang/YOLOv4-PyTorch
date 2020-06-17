# Setup Builtin Datasets

YOLOv4-PyTorch can support most datasets.
The default is `../data` relative to your current working directory.
Under this directory, YOLOv4-PyTorch expects to find datasets in the structure described below.

## Expected dataset structure for COCO instance detection:
```
COCO/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

You can use the 2014 version of the dataset as well.
Then, run `python coco2yolo.py --help`, to extract semantic annotations from panoptic annotations.


## Expected dataset structure for Pascal VOC:
```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
      # train.txt or val.txt, if you use these splits
  JPEGImages/
```