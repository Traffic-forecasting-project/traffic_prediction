
# Projet MLOps – Prévision et Analyse du Trafic Routier

## Objectif

Ce projet a pour objectif de construire une pipeline MLOps complète autour d’un cas d’usage réel : la **prévision du trafic routier** et l’**analyse d’incidents** à l’aide de données en temps réel issues de l’API TomTom et d’une API météo.

## Fonctionnalités principales

- Données mises à jour quotidiennement via une ingestion automatisée
- Chaîne de traitement (pipeline) entièrement automatisée (ETL, entraînement, déploiement)
- Suivi continu des performances et du drift du modèle
- API de prédiction sécurisée et scalable
- (Optionnel) Génération automatique de résumés d’incidents via LLM


## Arborescence


## Données utilisées

- **Incidents et trafic** : [TomTom Traffic API](https://developer.tomtom.com/traffic-api/api-explorer)
- **Météo (à venir)** : API externe (ex. OpenWeatherMap)

## Lancement du script

Avant exécution, définir la clé API TomTom comme variable d’environnement :

```bash
export API_TOMTOM=your_key_here
```

Puis exécuter :

```bash
python src/data/import_data.py
```
