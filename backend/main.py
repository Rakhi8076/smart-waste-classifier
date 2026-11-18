import io
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import models, transforms
from fastapi.middleware.cors import CORSMiddleware

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'organic']

# True = biodegradable, False = non-biodegradable, None = mixed/unclear
BIODEGRADABLE = {
    'cardboard': True,
    'paper': True,
    'organic': True,
    'glass': False,
    'metal': False,
    'plastic': False,
    'trash': False,
}
DISPOSAL = {
    'cardboard': {
        'bin': 'Dry / recyclable waste',
        'tips': [
            'Flatten the box and keep it dry.',
            'Remove tape and plastic wrapping if possible.',
            'Greasy or food-soaked cardboard is better composted than recycled.',
        ],
    },
    'paper': {
        'bin': 'Dry / recyclable waste',
        'tips': [
            'Keep paper clean and dry before giving it for recycling.',
            'Used tissues and heavily soiled paper are not recyclable.',
            'Old newspapers and notebooks can go to a local scrap dealer.',
        ],
    },
    'glass': {
        'bin': 'Dry / recyclable waste',
        'tips': [
            'Rinse bottles and jars and keep them whole.',
            'Wrap broken glass in newspaper and label it before disposal.',
            'Glass can be recycled again and again, so avoid mixing it with wet waste.',
        ],
    },
    'metal': {
        'bin': 'Dry / recyclable waste',
        'tips': [
            'Rinse food cans and let them dry.',
            'Give metal scrap to a local scrap dealer or recycling centre.',
            'Do not throw batteries or aerosol cans in normal waste.',
        ],
    },
    'plastic': {
        'bin': 'Dry / recyclable waste',
        'tips': [
            'Empty, rinse and dry the item before disposal.',
            'Keep plastic separate from wet and food waste.',
            'Never burn plastic, as it releases harmful fumes.',
        ],
    },
    'organic': {
        'bin': 'Wet / biodegradable waste',
        'tips': [
            'Compost at home if you can (kitchen scraps, peels, tea bags, eggshells).',
            'Keep wet waste separate from plastic and dry waste.',
            'Drain excess liquid before putting it in the wet waste bin.',
        ],
    },
    'trash': {
        'bin': 'General / non-recyclable waste',
        'tips': [
            'This item is generally not recyclable.',
            'Wrap diapers, masks, gloves and bandages in paper before disposal.',
            'Do not mix it with wet or recyclable waste.',
        ],
    },
}

CONFIDENCE_THRESHOLD = 0.60  # isse kam par "uncertain" bolenge

# Model load (server start hote hi ek baar)
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
model.load_state_dict(torch.load("model/best_model_v2.pth", map_location="cpu"))
model.eval()

# Training ke eval transform jaisa hi hona chahiye
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI(title="Waste Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image.")

    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    conf, idx = probs.max(0)
    category = CLASS_NAMES[idx.item()]
    confidence = round(conf.item(), 4)
    low_conf = confidence < CONFIDENCE_THRESHOLD
    message = (
        "I'm not sure about this item. Try a clearer photo in good light with a plain background."
        if low_conf else None
    )

    return {
        "category": category,
        "confidence": confidence,
        "biodegradable": BIODEGRADABLE[category],
        "disposal": DISPOSAL[category],
        "low_confidence": low_conf,
        "message": message,
        "all_probabilities": {c: round(p.item(), 4) for c, p in zip(CLASS_NAMES, probs)},
    }