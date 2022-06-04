# Flower classification with transfer learning

## CLI application usage :

usage: main.py [-h] [-i] [-u] [-p]

This application helps with detecting whether a flower is a daisy, dandelion, rose, sunflower or tulip

optional arguments:
-  -h, --help            : Show this help message and exit.
-  -i , --input_folder   : A folder with images to analyze.
-  -u , --url            : A link to an image to analyze.
-  -p , --match_pattern  : Get files similar to this pattern
## Docker execution command

The expected command line that should work:
```bash
# build image
docker build -t classification_app .
# define the image folder on host
export IMAGE_FOLDER=<YOUR IMG FOLDER>
# run the app with gpu
docker run --gpus all -it --rm -v ${IMAGE_FOLDER}:/images/ classification_app -i /images/ -m enlaps 2>/dev/null
# run the app without gpu
docker run -it --rm -v ${IMAGE_FOLDER}:/images/ classification_app -i /images/ -m custom 2>/dev/null
```

## Output format
For each image in the input folder the app should print a line in the form:
```json
{"img_name.jpg": {"score":0.25, "class":"sunflower"}}
```

