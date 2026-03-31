# LiDAR Scanner GUI

Interface graphique pour le scanner 3D Pandar 40P + moteur SimpleFOC.

## Lancement

```bash
python3 /home/jason/lidar/app/main.py
```

## Dependances

```bash
pip install PySide6 numpy pyserial pyqtgraph PyOpenGL --break-system-packages
```

## Fenetre principale — Scan

- Configurer IP LiDAR, port UDP, port serie du moteur
- Regler duree, RPM, decimation, retour moteur (activable)
- L'angle total est calcule automatiquement (RPM x duree)
- Option "Bandes centrales (10-30) + avant" pour filtrer le nuage
- Option calibration de montage (fichier JSON)
- Barre de progression + log temps reel
- Bouton Stop interrompt le scan a tout moment (le moteur est toujours arrete)

## Fenetre Calibration

Accessible via le bouton orange "Calibration...".

- **Enregistrer PCAP** : scan lent 360 deg en ~6 min, sauvegarde les paquets bruts
- **5 methodes** d'optimisation (v1 a v5), lancees en subprocess
- **Workflows** : "Complet (v1+v2)" ou "Rapide (v1+v5)" enchainent automatiquement
- Les resultats (roll/pitch/yaw, tx/ty/tz, amelioration) s'affichent en temps reel
- Les valeurs optimales sont reportees dans les champs "warm start" pour le chainage

## Fenetre Live View 3D

Accessible via le bouton violet "Live View 3D".

- Affichage 3D OpenGL du flux LiDAR en temps reel
- Points colores par canal (bleu=haut, rouge=bas)
- Controles : temps d'accumulation, taille des points, distance max, decimation
- Camera : clic gauche=rotation, molette=zoom, clic milieu=pan

## Fichiers

| Fichier | Role |
|---------|------|
| `main.py` | Point d'entree, fenetre principale scan |
| `scan_gui.ui` | Interface Qt du scan |
| `calibration.py` | Fenetre calibration + enregistrement PCAP |
| `calibration_gui.ui` | Interface Qt de la calibration |
| `liveview.py` | Vue 3D temps reel |
| `record_lidar.py` | Capture UDP, parsing paquets, ecriture PCD |
| `scan_lidar.py` | Scan avec rotation moteur |
| `replay_scan.py` | Lecture PCAP (utilise par les scripts optimize) |
