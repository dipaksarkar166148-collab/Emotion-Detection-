# ============================================================
# Cross-Lingual Assamese Emotion Classification using MuRIL
# Contrastive Alignment + Transfer Learning
# + Class Weight Balancing for Imbalanced Dataset
# ============================================================

!pip install transformers datasets accelerate sentencepiece seaborn -q

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments
)

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME      = "google/muril-base-cased"
MAX_LEN         = 128
BATCH_SIZE      = 16
EPOCHS          = 5
LR              = 2e-5
ALIGNMENT_WEIGHT = 0.3

# ============================================================
# LOAD DATA
# ============================================================

eng_df = pd.read_excel("/kaggle/input/datasets/dipaksarkar1/english-dataset/tweet_emotions.xlsx")
asm_df = pd.read_excel("/kaggle/input/datasets/dipaksarkar1/dataassamese/tweet_emotions_assamese.xlsx")

eng_df = eng_df[['content', 'emotion']]
asm_df = asm_df[['content', 'emotion']]

print("English samples :", len(eng_df))
print("Assamese samples:", len(asm_df))

assert len(eng_df) == len(asm_df)

# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()
labels        = label_encoder.fit_transform(eng_df["emotion"])
num_labels    = len(label_encoder.classes_)

print("\nClasses:", label_encoder.classes_)

# ============================================================
# CLASS DISTRIBUTION PLOT (before balancing)
# ============================================================

class_counts = pd.Series(labels).value_counts().sort_index()

plt.figure(figsize=(10, 5))
sns.barplot(
    x=[label_encoder.classes_[i] for i in class_counts.index],
    y=class_counts.values,
    palette="viridis"
)
plt.title("Class Distribution (Before Balancing)")
plt.xlabel("Emotion")
plt.ylabel("Sample Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.show()
print("Saved: class_distribution.png")

# ============================================================
# COMPUTE CLASS WEIGHTS
# ============================================================

# sklearn computes  weight[c] = n_samples / (n_classes * count[c])
# so minority classes get higher weight automatically.

class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_labels),
    y=labels
)

print("\nClass weights:")
for cls, w in zip(label_encoder.classes_, class_weights_np):
    print(f"  {cls:20s} -> {w:.4f}")

# Keep as a float32 tensor; will be moved to device inside Trainer
class_weights_tensor = torch.tensor(
    class_weights_np,
    dtype=torch.float32
)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

idx_train, idx_test = train_test_split(
    np.arange(len(eng_df)),
    test_size=0.20,
    stratify=labels,
    random_state=42
)

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# DATASET
# ============================================================

class EmotionDataset(torch.utils.data.Dataset):

    def __init__(self, en_texts, as_texts, labels):
        self.en_texts = en_texts
        self.as_texts = as_texts
        self.labels   = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        en = tokenizer(
            str(self.en_texts[idx]),
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length"
        )

        assamese = tokenizer(
            str(self.as_texts[idx]),
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length"
        )

        return {
            "en_input_ids"      : torch.tensor(en["input_ids"]),
            "en_attention_mask" : torch.tensor(en["attention_mask"]),
            "as_input_ids"      : torch.tensor(assamese["input_ids"]),
            "as_attention_mask" : torch.tensor(assamese["attention_mask"]),
            "labels"            : torch.tensor(self.labels[idx])
        }


train_dataset = EmotionDataset(
    eng_df.iloc[idx_train]["content"].tolist(),
    asm_df.iloc[idx_train]["content"].tolist(),
    labels[idx_train]
)

test_dataset = EmotionDataset(
    eng_df.iloc[idx_test]["content"].tolist(),
    asm_df.iloc[idx_test]["content"].tolist(),
    labels[idx_test]
)

# ============================================================
# MODEL
# ============================================================

class CrossLingualMuRIL(nn.Module):

    def __init__(self, num_labels):
        super().__init__()

        self.encoder    = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size     = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def mean_pool(self, last_hidden, attention_mask):
        mask       = attention_mask.unsqueeze(-1).float()
        embeddings = (last_hidden * mask).sum(dim=1)
        embeddings = embeddings / mask.sum(dim=1)
        return embeddings

    def contrastive_loss(self, emb_en, emb_as, temperature=0.07):
        emb_en  = F.normalize(emb_en, dim=1)
        emb_as  = F.normalize(emb_as, dim=1)
        logits  = (emb_en @ emb_as.T) / temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)

    def forward(
        self,
        en_input_ids,
        en_attention_mask,
        as_input_ids,
        as_attention_mask,
        labels=None
    ):
        en_output = self.encoder(
            input_ids=en_input_ids,
            attention_mask=en_attention_mask
        )
        as_output = self.encoder(
            input_ids=as_input_ids,
            attention_mask=as_attention_mask
        )

        emb_en = self.mean_pool(en_output.last_hidden_state, en_attention_mask)
        emb_as = self.mean_pool(as_output.last_hidden_state, as_attention_mask)

        logits = self.classifier(self.dropout(emb_as))

        # NOTE: loss is computed in WeightedTrainer using class weights;
        # returning logits only here is fine. We still compute a basic loss
        # so the standard Trainer path also works if needed.
        loss = None
        if labels is not None:
            alignment_loss = self.contrastive_loss(emb_en, emb_as)
            # Classification loss is overridden in WeightedTrainer below
            classification_loss = F.cross_entropy(logits, labels)
            loss = classification_loss + ALIGNMENT_WEIGHT * alignment_loss

        return {"loss": loss, "logits": logits}


model = CrossLingualMuRIL(num_labels)

# ============================================================
# WEIGHTED TRAINER
# Uses class-weighted cross-entropy for the classification head
# so minority emotions are penalised more on mis-classification.
# ============================================================

class WeightedTrainer(Trainer):
    """
    Overrides compute_loss to inject class weights into
    the cross-entropy for the classification part.
    The contrastive alignment loss remains unweighted
    (it is a self-supervised objective and doesn't use labels).
    """

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register weights; will be moved to the correct device automatically
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        labels = inputs.pop("labels")

        # ------------------------------------------------------------------
        # Unwrap DataParallel / DistributedDataParallel so we can call
        # model-specific methods (mean_pool, contrastive_loss, encoder).
        # On a single GPU / CPU, hasattr(model, 'module') is False and
        # raw_model == model, so this is always safe.
        # ------------------------------------------------------------------
        raw_model = model.module if hasattr(model, "module") else model

        en_input_ids      = inputs["en_input_ids"]
        en_attention_mask = inputs["en_attention_mask"]
        as_input_ids      = inputs["as_input_ids"]
        as_attention_mask = inputs["as_attention_mask"]

        # --- Single forward pass through the shared encoder ---
        en_out = raw_model.encoder(
            input_ids=en_input_ids,
            attention_mask=en_attention_mask
        )
        as_out = raw_model.encoder(
            input_ids=as_input_ids,
            attention_mask=as_attention_mask
        )

        emb_en = raw_model.mean_pool(en_out.last_hidden_state, en_attention_mask)
        emb_as = raw_model.mean_pool(as_out.last_hidden_state, as_attention_mask)

        logits = raw_model.classifier(raw_model.dropout(emb_as))

        outputs = {"logits": logits}

        # ---- Weighted classification loss ----
        weights = self.class_weights.to(logits.device)

        classification_loss = F.cross_entropy(
            logits,
            labels,
            weight=weights          # <-- class weight balancing applied here
        )

        # ---- Contrastive alignment loss (unweighted) ----
        alignment_loss = raw_model.contrastive_loss(emb_en, emb_as)

        loss = classification_loss + ALIGNMENT_WEIGHT * alignment_loss

        return (loss, outputs) if return_outputs else loss


# ============================================================
# METRICS
# ============================================================

def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(axis=1)

    return {
        "accuracy"        : accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted" : recall_score(y_true, y_pred, average="weighted"),
        "f1_macro"        : f1_score(y_true, y_pred, average="macro")
    }

# ============================================================
# TRAINING ARGS
# ============================================================

training_args = TrainingArguments(
    output_dir                  = "./muril_emotion",
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate               = LR,
    weight_decay                = 0.01,
    eval_strategy               = "epoch",
    save_strategy               = "no",
    logging_steps               = 50,
    report_to                   = "none"
)

# ============================================================
# TRAINER
# ============================================================

trainer = WeightedTrainer(
    class_weights   = class_weights_tensor,   # <-- injected here
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = test_dataset,
    compute_metrics = compute_metrics
)

# ============================================================
# TRAIN
# ============================================================

trainer.train()

torch.save(model.state_dict(), "muril_emotion_model.pt")
print("Model saved: muril_emotion_model.pt")

# ============================================================
# EVALUATION
# ============================================================

predictions = trainer.predict(test_dataset)

y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)
y_prob = torch.softmax(
    torch.tensor(predictions.predictions), dim=1
).numpy()

print("\nClassification Report\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4
    )
)

# ============================================================
# MACRO ROC AUC
# ============================================================

y_true_bin = label_binarize(y_true, classes=np.arange(num_labels))

macro_auc = roc_auc_score(
    y_true_bin,
    y_prob,
    average="macro",
    multi_class="ovr"
)
print("\nMacro ROC-AUC =", round(macro_auc, 4))

# ============================================================
# CLASS WEIGHT BAR CHART
# ============================================================

plt.figure(figsize=(10, 5))
sns.barplot(
    x=list(label_encoder.classes_),
    y=class_weights_np,
    palette="magma"
)
plt.title("Computed Class Weights (Inverse Frequency)")
plt.xlabel("Emotion")
plt.ylabel("Weight")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("class_weights.png", dpi=300)
plt.show()
print("Saved: class_weights.png")

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

cm_norm = confusion_matrix(y_true, y_pred, normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig("normalized_confusion_matrix.png", dpi=300)
plt.show()

# ============================================================
# ROC CURVES
# ============================================================

plt.figure(figsize=(10, 8))

for i in range(num_labels):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc     = auc(fpr, tpr)
    plt.plot(
        fpr, tpr, lw=2,
        label=f"{label_encoder.classes_[i]} (AUC={roc_auc:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curves")
plt.legend(loc="lower right", fontsize=8)
plt.grid()
plt.tight_layout()
plt.savefig("roc_auc_curve.png", dpi=300)
plt.show()

# ============================================================
# t-SNE VISUALIZATION
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

english_embeddings  = []
assamese_embeddings = []

with torch.no_grad():

    for i in range(min(500, len(test_dataset))):

        sample  = test_dataset[i]

        en_ids  = sample["en_input_ids"].unsqueeze(0).to(device)
        en_mask = sample["en_attention_mask"].unsqueeze(0).to(device)
        as_ids  = sample["as_input_ids"].unsqueeze(0).to(device)
        as_mask = sample["as_attention_mask"].unsqueeze(0).to(device)

        en_out  = model.encoder(en_ids, attention_mask=en_mask)
        as_out  = model.encoder(as_ids, attention_mask=as_mask)

        en_emb  = model.mean_pool(en_out.last_hidden_state, en_mask)
        as_emb  = model.mean_pool(as_out.last_hidden_state, as_mask)

        english_embeddings.append(en_emb.cpu().numpy()[0])
        assamese_embeddings.append(as_emb.cpu().numpy()[0])

english_embeddings  = np.array(english_embeddings)
assamese_embeddings = np.array(assamese_embeddings)

combined = np.vstack([english_embeddings, assamese_embeddings])

tsne    = TSNE(n_components=2, perplexity=30, random_state=42)
reduced = tsne.fit_transform(combined)

n = len(english_embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(reduced[:n, 0], reduced[:n, 1], alpha=0.6, label="English")
plt.scatter(reduced[n:, 0], reduced[n:, 1], alpha=0.6, label="Assamese")
plt.legend()
plt.title("Cross-Lingual Alignment t-SNE")
plt.tight_layout()
plt.savefig("tsne_alignment.png", dpi=300)
plt.show()

print("\nAll outputs saved:")
print("  class_distribution.png")
print("  class_weights.png")
print("  confusion_matrix.png")
print("  normalized_confusion_matrix.png")
print("  roc_auc_curve.png")
print("  tsne_alignment.png")
