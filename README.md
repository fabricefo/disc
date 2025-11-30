Une application moderne d'évaluation de la personnalité DISC, conçue avec Streamlit et Python, pour générer des profils DISC personnalisés.

Ce projet est une version traduite en français.
Projet source : https://github.com/dzyla/disc-personality-assessment

Modifications : 
- Traduction des questions dans questions.json
- Traduction des profils dans disc_description.json
- Traduction des points forts dans strengths.json
- Import des json avec encoding="utf-8" 
- Résultat : la barre de progrès est en couleur.

# Requirements

streamlit
numpy==1.26.4
matplotlib
reportlab
Pillow

# Installation

```uv venv```
```uv pip install -r requirements.txt```

# Run the app

```uv run streamlit run disc_style.py```

