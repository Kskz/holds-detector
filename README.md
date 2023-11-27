# holds-detector

CLIMBING HOLDS DETECTOR

The program is designed to recognize climbing holds on pictures.

A faster R-CNN model with an FPN backbone is used to guarantee better accuracy, especially on small objects

RESULT

Sample results:

![image](https://github.com/Kskz/holds-detector/assets/63345177/3a4fe29e-b884-4e22-97a4-f61bc71bf74a)

![image](https://github.com/Kskz/holds-detector/assets/63345177/17a153f2-ace9-4522-b078-203f92fbe161)

Train loss diagram 
python detect_image.py "C:\Users\itachi\UOL\lily\social_climbing_images\first.png"

Usage
- Clone the repo using git clone --https://github.com/Kskz/holds-detector.git
- Use pip install to install the dependencies
- Upload data set from https://universe.roboflow.com/uol-sguju/holds-hs0d5/dataset/1 and add it to "data" directory
- You can change the batch size, number of epochs and number of workers in "config.py"
- Train model by running "train.py"
- Test the model by running "test.py". You can see testing results in the "test_output" directory
- To detect holds on a single image run python detect_image.py [image_path]
