# BrainScan — Classification d'IRM cérébrales par CNN

## 1. Présentation du projet

BrainScan est un projet de Deep Learning qui vise à développer un modèle de classification d'images d'IRM cérébrales à l'aide d'un réseau de neurones convolutif (CNN).

Le modèle apprend à classer les images dans quatre catégories : glioma, meningioma, notumor et pituitary.

Ce projet a un objectif pédagogique. Il ne constitue pas un outil médical validé et ne doit pas être utilisé pour établir un diagnostic.

## 2. Objectifs

- Explorer et préparer un dataset d'IRM cérébrales.
- Détecter et supprimer les images strictement dupliquées.
- Redimensionner et normaliser les images.
- Construire et entraîner un CNN avec TensorFlow/Keras.
- Évaluer les performances à l'aide de plusieurs métriques.
- Analyser les erreurs avec une matrice de confusion.

## 3. Technologies utilisées

- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- OpenCV

## 4. Classes à prédire

| Classe | Description |
|---|---|
| `glioma` | Catégorie d'images associées au gliome |
| `meningioma` | Catégorie d'images associées au méningiome |
| `notumor` | Images sans tumeur identifiée dans le dataset |
| `pituitary` | Catégorie d'images associées à une tumeur hypophysaire |

Les descriptions correspondent aux étiquettes du dataset et ne représentent pas une interprétation médicale des images.

## 5. Préparation des données

Les principales étapes de préparation sont :

1. Exploration du dataset et vérification des classes.
2. Détection des images dupliquées à l'aide de leurs empreintes SHA-256.
3. Conservation d'une seule copie des images strictement identiques.
4. Mélange des données.
5. Redimensionnement des images en 224 × 224 pixels.
6. Conversion des images en RGB.
7. Normalisation des pixels entre 0 et 1.
8. Encodage des labels en valeurs numériques.
9. Séparation des données en ensembles d'entraînement, de validation et de test.

Après la suppression des doublons, le dataset contient 6 726 images uniques.

| Ensemble | Nombre d'images |
|---|---:|
| Entraînement | 4 708 |
| Validation | 1 009 |
| Test | 1 009 |
| Total | 6 726 |

Les images sont représentées sous la forme `(224, 224, 3)`.

## 6. Architecture du CNN

Le modèle utilise les couches suivantes :

- `Conv2D` : extraction progressive des caractéristiques visuelles.
- `ReLU` : introduction de la non-linéarité.
- `MaxPooling2D` : réduction des dimensions spatiales.
- `Dropout` : réduction du risque de surapprentissage.
- `Flatten` : transformation des cartes de caractéristiques en vecteur.
- `Dense` : classification à partir des caractéristiques extraites.
- `Softmax` : calcul des probabilités associées aux quatre classes.

L'entraînement utilise l'optimiseur Adam et la fonction de perte `sparse_categorical_crossentropy`.

## 7. Résultats de l'évaluation

Sur l'ensemble de test, le modèle a obtenu les résultats suivants :

| Métrique | Résultat |
|---|---:|
| Accuracy | 94,45 % |
| Test loss | 0,1903 |

### Résultats par classe

| Classe | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Glioma | 0,94 | 0,93 | 0,94 |
| Meningioma | 0,90 | 0,89 | 0,90 |
| Notumor | 0,98 | 0,97 | 0,97 |
| Pituitary | 0,96 | 0,98 | 0,97 |

La classe `meningioma` présente les performances les plus faibles parmi les quatre catégories. La matrice de confusion permet d'identifier les principales erreurs de classification.

Ces résultats concernent uniquement le dataset de test utilisé. Ils ne garantissent pas des performances équivalentes sur de nouvelles données.

## 8. Structure du projet

```text
BrainScan/
├── Data/
├── data/
│   └── processed/
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       └── y_test.npy
├── models/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cnn_training.ipynb
│   └── 03_model_evaluation.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. Installation

Cloner le dépôt :

```bash
git clone https://github.com/MariaE003/BrainScan.git
cd BrainScan
```

Créer un environnement virtuel :

```bash
python -m venv venv
```

Activer l'environnement sous Windows PowerShell :

```powershell
.\venv\Scripts\Activate.ps1
```

Installer les dépendances :

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Lancer Jupyter Notebook :

```bash
jupyter notebook
```


## 10. Auteur

**Maria Elgotby**

GitHub : https://github.com/MariaE003

Projet réalisé dans le cadre de l'apprentissage du Deep Learning et de la classification d'images par CNN.
