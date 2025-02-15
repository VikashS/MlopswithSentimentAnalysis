import requests
import json

image_url = "https://raw.githubusercontent.com/VikashS/images/refs/heads/main/street.jpg"
text = "Donald Trump"


def parse_json(data):
    json_string = data['result']
    parsed_data = json.loads(json_string)
    predictions = parsed_data['predictions']
    print("==================================>")
    print("Image Prediction:")
    for prediction in predictions:
        label = prediction['label']
        probability = prediction['probability']
        print(f"- Label: {label}")
        print(f"  Probability: {probability}%")
    print("==================================>")

def check_test():
    # Test image prediction
    response = requests.post(f'http://localhost:8001/predict/image?image_url={image_url}')
    parse_json(response.json())


    # Test text prediction

    response = requests.post(f'http://localhost:8001/predict/text?text={text}')
    print("==================================>")
    print("Text Prediction:", response.json())
    print("==================================>")


if __name__=="__main__":
    url = "https://raw.githubusercontent.com/VikashS/images/refs/heads/main/street.jpg"
    print("started")
    check_test()
    print("ended")