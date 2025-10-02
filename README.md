# SAMnotate
Segment Anything Model (SAM) based image annotation tool, just give the path to your images folder for a certain class and click on the objects you want to annotate once and voila - SAMnotate will do the rest for you!.

Thus, you can annotate custom objects for which there is no pre-trained model available and also no dataset available to train a model from scratch. If you wanna run a lightweight yolo model on your raspberry pi, for detecting some custom objects like a watch strap so that your robot will be able to pick it up, you can use SAMnotate to annotate a few images of watch straps and train a lightweight yolo model on it.
### Setup
1. Clone the repository
2. Install the required packages by running the following command:
```pip install -r requirements.txt```
3. Then run the Detector.py file by providing the path to your images folder in the code.
```bash
python Detector.py
```

`"Et voilà!, plus besoin de chercher des datasets sur internet, plus besoin de passer des heures à annoter des images, SAMnotate s'occupe de tout!"`
- Me
