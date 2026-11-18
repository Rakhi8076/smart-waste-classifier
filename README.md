# Smart Waste Classifier

An AI-powered waste classification and disposal recommendation system. Upload a photo of a waste item and the app tells you its **category**, whether it is **biodegradable**, how **confident** the model is, and **how to dispose of it**.

Built with a MobileNetV2 deep learning model (PyTorch), a FastAPI backend and a React frontend.

<!-- Add a screenshot of the app here: ![Demo](docs/demo.png) -->

## Features

- Image upload with preview
- Classifies waste into 7 categories: cardboard, glass, metal, paper, plastic, trash, organic
- Shows the model's confidence for the prediction
- Biodegradable / non-biodegradable flag
- Disposal recommendation (which bin, plus practical tips)
- Low-confidence warning that asks the user to retake the photo

## How it works

```
React app  ->  POST /predict (image)  ->  FastAPI  ->  MobileNetV2  ->  category + confidence
                                                                 |
                                       rule-based lookup  <------+
                                  (biodegradable flag + disposal tips)
```

- **Deep learning part:** the CNN predicts the waste category and a confidence score (softmax probability).
- **Rule-based part:** the biodegradable flag and the disposal tips are a fixed lookup based on the predicted category. They are not predicted by a separate model.

## Model and training

- **Dataset:** [TrashBusters/combined](https://huggingface.co/datasets/TrashBusters/combined) on Hugging Face (MIT license), already split into train / validation / test.
- **Relabelling:** the original dataset has 6 classes, but its `trash` class mostly contained food waste. Using the dataset's `original_label` column, I created a 7th class:
  - `organic` = A_Foods, food_waste, eggshells, tea_bags, biological
  - `trash` = everything else that was in the original trash class (diapers, masks, gloves, porcelain, wrappers, construction scrap and similar)
  - `B_Animal Dead Body` images were dropped
- **Final split sizes:** 20,827 train, 4,464 validation, 4,460 test images.
- **Model:** MobileNetV2 pretrained on ImageNet, with a new 7-class head.
- **Training setup:**
  - Input size 224x224, ImageNet normalization
  - Augmentation on the training set: random resized crop, horizontal flip, rotation, color jitter
  - Adam optimizer, cross-entropy loss with class weights (the `trash` and `organic` classes have fewer images)
  - Transfer learning in two stages: first only the new classifier head, then the last 4 feature blocks were unfrozen and fine-tuned with a small learning rate
  - Final 7-class model trained for 6 epochs; the checkpoint with the best validation accuracy (89.58%) was kept
- Trained on Google Colab (T4 GPU).

## Results

Evaluated once on the held-out test set (4,460 images, 7 classes) with the final model:

**Test accuracy: 90.38%**

| Class | Precision | Recall | F1-score | Test images |
|---|---|---|---|---|
| cardboard | 0.9226 | 0.9492 | 0.9357 | 728 |
| glass | 0.9366 | 0.9653 | 0.9508 | 750 |
| metal | 0.9168 | 0.9253 | 0.9210 | 750 |
| paper | 0.8779 | 0.8627 | 0.8702 | 750 |
| plastic | 0.9088 | 0.8240 | 0.8643 | 750 |
| trash | 0.7839 | 0.8629 | 0.8215 | 248 |
| organic | 0.9022 | 0.9153 | 0.9087 | 484 |

Macro-average F1: 0.8960.

Progression on the way here (different setups, so not directly comparable):

| Setup | Test accuracy |
|---|---|
| 6 classes, frozen backbone | 78.79% |
| 6 classes, last 4 blocks fine-tuned | 88.68% |
| 7 classes (with `organic`), fine-tuned | 90.38% |

## Tech stack

| Part | Tools |
|---|---|
| Model | PyTorch, torchvision (MobileNetV2) |
| Backend | FastAPI, Uvicorn, Pillow |
| Frontend | React (Vite) |
| Training | Google Colab, Hugging Face `datasets`, scikit-learn |

## Project structure

```
waste-classifier/
├── backend/
│   ├── main.py              # FastAPI app: /predict endpoint, biodegradable map, disposal tips
│   ├── requirements.txt
│   └── model/
│       └── best_model_v2.pth  # trained 7-class weights
└── frontend/                # React (Vite) app
    └── src/
        ├── App.jsx
        └── App.css
```

## Run locally

You need Python 3.8+ and Node.js. Run the backend and the frontend in two separate terminals.

**1. Backend**

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows (on macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`. Interactive docs are at `http://127.0.0.1:8000/docs`.

**2. Frontend**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, choose an image and click **Classify**.

## API

`POST /predict` with a multipart form field named `file` (an image).

Example response:

```json
{
  "category": "organic",
  "confidence": 0.9993,
  "biodegradable": true,
  "disposal": {
    "bin": "Wet / biodegradable waste",
    "tips": [
      "Compost at home if you can (kitchen scraps, peels, tea bags, eggshells).",
      "Keep wet waste separate from plastic and dry waste.",
      "Drain excess liquid before putting it in the wet waste bin."
    ]
  },
  "low_confidence": false,
  "message": null,
  "all_probabilities": {
    "cardboard": 0.0001, "glass": 0, "metal": 0, "paper": 0,
    "plastic": 0, "trash": 0.0005, "organic": 0.9993
  }
}
```

## Limitations

- **Only 7 classes.** The model has never seen anything outside them. An unrelated image (a person, furniture) will still be forced into one of the 7 classes, sometimes with high confidence. Softmax confidence is not the same as the probability of being correct, so the 0.60 low-confidence threshold is a starting value that I have not tuned.
- **Lesson from the first version:** the original 6-class model labelled a banana peel as `trash` with 99.9% confidence, because food waste was hidden inside the dataset's `trash` class. Relabelling the data and adding an `organic` class fixed this case.
- **Dataset noise.** The dataset combines several sources and some labels are noisy (for example, non-food items appear among food images).
- **The `trash` class is a mixed bag** (diapers, masks, gloves, wrappers and more) and has the lowest F1 (0.82), with only 248 test images.
- **Plastic and paper are the weakest of the main classes** and are sometimes confused with glass, metal or cardboard.
- **Disposal tips are general guidelines** written for the common wet / dry waste segregation in India. Rules differ between cities, so users should check their local municipal guidelines.
- Photos with a clean background and good lighting work best, since many dataset images are of that kind.

## Possible future work

- Add more classes (e.g. e-waste, batteries, hazardous waste)
- Out-of-distribution detection so unrelated images are rejected
- Camera capture for mobile users
- Deployment of the backend and frontend

## Acknowledgements

Dataset: [TrashBusters/combined](https://huggingface.co/datasets/TrashBusters/combined) (MIT license). Pretrained MobileNetV2 weights from torchvision (ImageNet).
