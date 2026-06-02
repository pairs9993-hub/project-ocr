"""Generate washer/dryer UI synthetic screenshots for OCR training.

This generator is intentionally narrower than the root synth_generator.py:
it targets LG-style washer/dryer screens across selected UI locales and emits
the same labels.jsonl schema used by the existing det/rec prep scripts.

Example:
  python scripts/generate_real_ui_synth.py --output-dir artifacts/real_ui_smoke --count 30

Chunked 1M generation:
  python scripts/generate_real_ui_synth.py --output-dir generated_1000000_real_ui_en_fr_es/chunks/chunk_000000_124999 --count 1000000 --start-index 0 --chunk-count 125000
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from real_ui_zh_lang import ZH_LANG


CANVAS_SMALL = (320, 240)
CANVAS_LARGE = (1280, 480)

WHITE = (255, 255, 255)
DIM = (145, 145, 145)
MUTED = (95, 95, 95)
BLUE = (0, 174, 239)
PANEL = (38, 38, 38)

FONT_CANDIDATES = {
    "regular": [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "bold": [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/tahomabd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
}

CJK_FONT_CANDIDATES = {
    "regular": [
        "C:/Windows/Fonts/msjh.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/mingliu.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ],
    "bold": [
        "C:/Windows/Fonts/msjhbd.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/mingliub.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    ],
}


@dataclass
class Element:
    type: str
    bbox: list[int]
    text: str | None = None
    selected: bool | None = None
    color_class: str | None = None
    size_class: str | None = None


@dataclass
class Label:
    image_path: str
    pattern: str
    language: str
    background: str
    canvas_size: list[int]
    elements: list[Element]
    raw_text: str


LANG = {
    "en": {
        "cycles": [
            "Normal", "AI Wash & Dry", "Delicates", "Bedding", "Heavy Duty", "Quick",
            "Towels", "Tub Clean", "Timed Dry", "Activewear", "Rinse & Spin",
            "Spin Only", "Dry Only", "Quick Tub Rinse", "Tub Dry", "Small Load",
            "Large Load", "XL Load", "Sanitary", "Kids Wear", "Color Care",
            "Sweat Stains", "Hand/Wool", "Jeans", "Blanket Refresh", "Perm. Press",
            "ColdWash™", "Allergiene™", "BrightWhites™", "ezDispense™ Nozzle Clean",
        ],
        "settings": [
            "Wash Settings", "Dry Settings", "Drum Light", "Wrinkle Care", "Control Lock",
            '"More Cycles" Edit', "Additional Settings", "Wi-Fi", "Language", "Signal",
            "Smart Diagnosis", "Factory Reset", "Open Source Licenses", "Steam",
            "TurboWash™", "Pre-wash", "Soak", "Use of Dispenser 2",
            "Default Det. Dispense", "Default Soft. Dispense", "Energy Saver",
            "Dry Start Signal", "Damp Signal", "Pre-dry Spin", "Remote Start",
        ],
        "status": [
            "Washing", "Rinsing", "Spinning", "Drying", "Cooling", "Cleaning", "Refreshing",
            "Paused", "Wash finished", "Dry finished", "Wash & Dry finished",
            "Wrinkle Care finished", "Waiting to be completed", "Detecting Load Level",
            "Detecting Soil Level", "Auto Detergent Dispensing", "Auto Softener Dispensing",
        ],
        "labels": {
            "temp": "Temp.", "spin": "Spin", "soil": "Soil", "dry_level": "Dry Level",
            "dry_temp": "Dry Temp.", "dry_time": "Dry Time", "delay": "Delay Start",
        },
        "temps": ["Extra Hot", "Hot", "Warm", "Cold", "Tap Cold", "Med. High", "High", "-"],
        "spins": ["Extra High", "High", "Medium", "Low", "No Spin", "-"],
        "soils": ["Heavy", "Med. Heavy", "Medium", "Med. Light", "Light", "-"],
        "dry_levels": ["Very", "Normal", "Damp", "-"],
        "dry_temps": ["High", "Med. High", "Medium", "Med. Low", "Low", "-"],
        "detergent": ["Maximum", "More", "Normal", "Less", "Minimum", "Off"],
        "signal": ["Very High", "High", "Medium", "Low", "Off"],
        "on_off": ["On", "Off"],
        "saved": ["Saved.", "Cycle is canceled.", "Delay Start is canceled.", "Turning off Remote Start."],
        "hint_add": "Press to add garments.",
        "hint_start": "Press to start the cycle.",
        "hr": "hr",
        "min": "min",
        "messages": [
            ["Door open", "Open and close", "the door again."],
            ["Door unlocked", "Check for objects", "around the door", "and door seal."],
            ["Water supply", "Turn on water.", "Check water pressure", "and inlet hoses."],
            ["Drainage", "Check for a blocked", "drain filter or", "bent drain hose."],
            ["Unbalance", "Rearrange the laundry", "and resume the cycle."],
            ["Oversudsing", "Do not add more than", "the recommended", "amount of det."],
            ["Communication error", "The appliance does not", "work properly. If error", "recurs, call for service."],
            ["Remote Start still on.", "Press Power button", "to disable Remote Start."],
            ["Press and hold OK", "for 3 seconds", "to turn off Control Lock."],
            ["A maximum of 10 cycles", "can be displayed.", "Uncheck unused cycles."],
        ],
        "codes": ["UNKNOWN ERR", "B3", "B4", "B5", "FE1", "EMPERATURE_SENSOR_I", "<PROC_W_SOAKING_PRC", "<CYCLE_D_AIR_DRY_1I"],
    },
    "fr": {
        "cycles": [
            "Normal", "Lavage AI", "Délicat", "Literie", "Intensif", "Rapide",
            "Serviettes", "Nettoyage cuve", "Séchage minuté", "Vêtements sport",
            "Rinçage+Essorage", "Essorage seul", "Séchage seul", "Rinçage rapide cuve",
            "Séchage cuve", "Petite charge", "Grande charge", "Très grande charge",
            "Sanitaire", "Vêtements enfants", "Soin couleurs", "Taches de sueur",
            "Main/Laine", "Jeans", "Rafraîchir couverture", "Pressage perm.",
            "ColdWash™", "Allergiene™", "BrightWhites™", "Nettoyage buse ezDispense™",
        ],
        "settings": [
            "Réglages lavage", "Réglages séchage", "Éclairage tambour", "Antifroissage",
            "Verrouillage", 'Modifier "Plus de cycles"', "Paramètres supplémentaires",
            "Wi-Fi", "Langue", "Signal", "Diagnostic intelligent", "Réinitialisation",
            "Licences open source", "Vapeur", "TurboWash™", "Prélavage", "Trempage",
            "Utilisation distributeur 2", "Dose dét. par défaut", "Dose assoupl. par défaut",
            "Économie d'énergie", "Signal début séchage", "Signal humide", "Essorage avant séchage",
            "Démarrage à distance",
        ],
        "status": [
            "Lavage", "Rinçage", "Essorage", "Séchage", "Refroidissement", "Nettoyage",
            "Rafraîchissement", "En pause", "Lavage terminé", "Séchage terminé",
            "Lavage & séchage terminé", "Antifroissage terminé", "En attente de fin",
            "Détection charge", "Détection salissure", "Distribution détergent auto",
            "Distribution assouplissant auto",
        ],
        "labels": {
            "temp": "Temp.", "spin": "Essorage", "soil": "Salissure", "dry_level": "Niveau séchage",
            "dry_temp": "Temp. séchage", "dry_time": "Temps séchage", "delay": "Départ différé",
        },
        "temps": ["Très chaud", "Chaud", "Tiède", "Froid", "Eau froide", "Moy. haut", "Haut", "-"],
        "spins": ["Très haut", "Haut", "Moyen", "Bas", "Sans essorage", "-"],
        "soils": ["Très sale", "Moy. sale", "Moyen", "Peu sale", "Léger", "-"],
        "dry_levels": ["Très", "Normal", "Humide", "-"],
        "dry_temps": ["Haut", "Moy. haut", "Moyen", "Moy. bas", "Bas", "-"],
        "detergent": ["Maximum", "Plus", "Normal", "Moins", "Minimum", "Arrêt"],
        "signal": ["Très haut", "Haut", "Moyen", "Bas", "Arrêt"],
        "on_off": ["Marche", "Arrêt"],
        "saved": ["Enregistré.", "Cycle annulé.", "Départ différé annulé.", "Démarrage à distance désactivé."],
        "hint_add": "Appuyez pour ajouter du linge.",
        "hint_start": "Appuyez pour démarrer le cycle.",
        "hr": "h",
        "min": "min",
        "messages": [
            ["Porte ouverte", "Ouvrez et fermez", "la porte à nouveau."],
            ["Porte déverrouillée", "Vérifiez les objets", "près de la porte", "et du joint."],
            ["Alimentation en eau", "Ouvrez l'eau.", "Vérifiez la pression", "et les tuyaux d'entrée."],
            ["Vidange", "Vérifiez le filtre", "de vidange ou", "le tuyau plié."],
            ["Déséquilibre", "Réorganisez le linge", "et reprenez le cycle."],
            ["Trop de mousse", "N'ajoutez pas plus", "que la quantité", "recommandée."],
            ["Erreur communication", "L'appareil ne fonctionne", "pas correctement.", "Appelez le service."],
            ["Démarrage à distance actif.", "Appuyez sur Power", "pour le désactiver."],
            ["Maintenez OK", "pendant 3 secondes", "pour désactiver le verrou."],
            ["Un maximum de 10 cycles", "peut être affiché.", "Décochez les cycles inutilisés."],
        ],
        "codes": ["ERREUR INCONNUE", "B3", "B4", "B5", "FE1", "CAPTEUR_TEMP_I", "<PROC_W_TREMPAGE", "<CYCLE_D_AIR_DRY_1I"],
    },
    "es": {
        "cycles": [
            "Normal", "Lavado AI", "Delicadas", "Ropa de cama", "Carga pesada", "Lavado rápido",
            "Toallas", "Limpieza de tina", "Secado programado", "Ropa deportiva",
            "Enjuague+Centrifugado", "Solo centrifugado", "Solo secado", "Enjuague rápido tina",
            "Secado de tina", "Carga pequeña", "Carga grande", "Carga XL", "Higiénico",
            "Ropa de niños", "Cuidado de color", "Manchas de sudor", "A mano/lana",
            "Jeans", "Refrescar manta", "Planchado perm.", "ColdWash™", "Allergiene™",
            "BrightWhites™", "Limpieza boquilla ezDispense™",
        ],
        "settings": [
            "Ajustes lavado", "Ajustes secado", "Luz del tambor", "Cuidado antiarrugas",
            "Bloqueo de Control", "Editar Más ciclos", "Ajustes adicionales", "Wi-Fi",
            "Idioma", "Señal", "Smart Diagnosis", "Restablecer fábrica",
            "Licencias de código abierto", "Vapor", "TurboWash™", "Prelavado", "Remojo",
            "Uso del Dispensador 2", "Disp. det. predeterm.", "Disp. suav. predeterm.",
            "Ahorro de energía", "Señal inicio secado", "Señal de humedad", "Centrifugado presecado",
            "Inicio remoto",
        ],
        "status": [
            "Lavando", "Enjuagando", "Centrifugando", "Secando", "Enfriando", "Limpiando",
            "Refrescando", "Pausado", "Lavado finalizado", "Secado finalizado",
            "Lavado y secado finalizado", "Cuidado antiarrugas finalizado", "Esperando finalizar",
            "Detectando nivel de carga", "Detectando nivel de suciedad", "Dispensando detergente auto",
            "Dispensando suavizante auto",
        ],
        "labels": {
            "temp": "Temp.", "spin": "Centrifugado", "soil": "Suciedad", "dry_level": "Nivel secado",
            "dry_temp": "Temp. secado", "dry_time": "Tiempo secado", "delay": "Inicio Diferido",
        },
        "temps": ["Extra caliente", "Caliente", "Tibio", "Fría", "Agua fría", "Med. alta", "Alta", "-"],
        "spins": ["Extra Alta", "Alta", "Media", "Baja", "Sin centrif.", "-"],
        "soils": ["Pesado", "Med. pesado", "Media", "Med. ligero", "Ligero", "-"],
        "dry_levels": ["Muy", "Normal", "Húmedo", "-"],
        "dry_temps": ["Alta", "Med. alta", "Media", "Med. baja", "Baja", "-"],
        "detergent": ["Máximo", "Más", "Normal", "Menos", "Mínimo", "Off"],
        "signal": ["Muy alto", "Alto", "Medio", "Bajo", "Off"],
        "on_off": ["On", "Off"],
        "saved": ["Guardado.", "Ciclo cancelado.", "Inicio Diferido se canceló.", "Desactivó el Inicio Remoto."],
        "hint_add": "Presione para agregar prendas.",
        "hint_start": "Presione para iniciar el ciclo.",
        "hr": "h",
        "min": "min",
        "messages": [
            ["Puerta abierta", "Abra y cierre", "la puerta de nuevo."],
            ["Puerta desbloqueada", "Revise objetos", "alrededor de la puerta", "y la junta."],
            ["Suministro de agua", "Abra el suministro.", "Revise la presión", "y las mangueras."],
            ["Drenaje", "Revise el filtro", "de drenaje o", "la manguera doblada."],
            ["Desbalanceo", "Reorganice la ropa", "y reanude el ciclo."],
            ["Exceso de espuma", "No agregue más", "que la cantidad", "recomendada."],
            ["Error de comunicación", "El aparato no funciona", "correctamente.", "Llame al servicio."],
            ["Inicio remoto activo.", "Presione Power", "para desactivarlo."],
            ["Mantenga presionado OK", "por 3 segundos", "para desactivar bloqueo."],
            ["Un máximo de 10 ciclos", "pueden mostrarse.", "Desmarque ciclos no usados."],
        ],
        "codes": ["ERROR DESCONOCIDO", "B3", "B4", "B5", "FE1", "SENSOR_TEMPERATURA_I", "<PROC_W_REMOJO", "<CYCLE_D_AIR_DRY_1I"],
    },
    "de": {
        "cycles": [
            "Normal", "AI Waschen & Trocknen", "Feinwäsche", "Bettwäsche", "Intensiv", "Schnell",
            "Handtücher", "Trommelreinigung", "Zeittrocknen", "Sportkleidung", "Spülen & Schleudern",
            "Nur Schleudern", "Nur Trocknen", "Trommelspülung", "Trommel trocknen", "Kleine Beladung",
            "Große Beladung", "XL Beladung", "Hygiene", "Kinderkleidung", "Farbschonend",
            "Schweißflecken", "Hand/Wolle", "Jeans", "Decken auffrischen", "Pflegeleicht",
            "ColdWash™", "Allergiene™", "BrightWhites™", "ezDispense™ Düsenreinigung",
        ],
        "settings": [
            "Wascheinstellungen", "Trockeneinstellungen", "Trommellicht", "Knitterschutz", "Tastensperre",
            '"Mehr Programme" bearb.', "Weitere Einstellungen", "Wi-Fi", "Sprache", "Signal",
            "Smart Diagnosis", "Werkseinstellungen", "Open-Source-Lizenzen", "Dampf",
            "TurboWash™", "Vorwäsche", "Einweichen", "Fach 2 verwenden",
            "Standard Waschmittel", "Standard Weichspüler", "Energiesparen",
            "Trockenstart-Signal", "Feucht-Signal", "Vorschleudern", "Fernstart",
        ],
        "status": [
            "Wäscht", "Spült", "Schleudert", "Trocknet", "Kühlt ab", "Reinigt", "Frischt auf",
            "Pausiert", "Waschen beendet", "Trocknen beendet", "Waschen & Trocknen fertig",
            "Knitterschutz beendet", "Warten auf Abschluss", "Beladung wird erkannt",
            "Verschmutzung wird erkannt", "Autom. Waschmittelzugabe", "Autom. Weichspülerzugabe",
        ],
        "labels": {
            "temp": "Temp.", "spin": "Schleudern", "soil": "Schmutz", "dry_level": "Trockengrad",
            "dry_temp": "Trockentemp.", "dry_time": "Trockenzeit", "delay": "Startzeitvorwahl",
        },
        "temps": ["Extra heiß", "Heiß", "Warm", "Kalt", "Leitungskalt", "Mittelhoch", "Hoch", "-"],
        "spins": ["Extra hoch", "Hoch", "Mittel", "Niedrig", "Kein Schleudern", "-"],
        "soils": ["Stark", "Mittelstark", "Mittel", "Mittelleicht", "Leicht", "-"],
        "dry_levels": ["Sehr", "Normal", "Feucht", "-"],
        "dry_temps": ["Hoch", "Mittelhoch", "Mittel", "Mittelniedrig", "Niedrig", "-"],
        "detergent": ["Maximum", "Mehr", "Normal", "Weniger", "Minimum", "Aus"],
        "signal": ["Sehr hoch", "Hoch", "Mittel", "Niedrig", "Aus"],
        "on_off": ["Ein", "Aus"],
        "saved": ["Gespeichert.", "Programm abgebrochen.", "Startzeitvorwahl abgebrochen.", "Fernstart wird deaktiviert."],
        "hint_add": "Drücken, um Wäsche hinzuzufügen.",
        "hint_start": "Drücken, um Programm zu starten.",
        "hr": "Std.",
        "min": "Min.",
        "messages": [
            ["Tür offen", "Tür erneut öffnen", "und schließen."],
            ["Tür entriegelt", "Auf Gegenstände prüfen", "an Tür und", "Türdichtung."],
            ["Wasserzufuhr", "Wasser aufdrehen.", "Wasserdruck und", "Zulaufschläuche prüfen."],
            ["Ablauf", "Ablauffilter auf", "Verstopfung oder", "Knick im Schlauch prüfen."],
            ["Unwucht", "Wäsche neu verteilen", "und Programm fortsetzen."],
            ["Zu viel Schaum", "Nicht mehr als die", "empfohlene Menge", "Waschmittel zugeben."],
            ["Kommunikationsfehler", "Gerät funktioniert nicht", "richtig. Bei erneutem", "Fehler Service rufen."],
            ["Fernstart noch aktiv.", "Ein/Aus-Taste drücken,", "um Fernstart zu beenden."],
            ["OK 3 Sekunden", "gedrückt halten, um die", "Tastensperre zu lösen."],
            ["Maximal 10 Programme", "können angezeigt werden.", "Unbenutzte abwählen."],
        ],
        "codes": ["UNKNOWN ERR", "B3", "B4", "B5", "FE1", "EMPERATURE_SENSOR_I", "<PROC_W_SOAKING_PRC", "<CYCLE_D_AIR_DRY_1I"],
    },
    "nl": {
        "cycles": [
            "Normaal", "AI Wassen & Drogen", "Fijne was", "Beddengoed", "Intensief", "Snel",
            "Handdoeken", "Trommelreiniging", "Tijddrogen", "Sportkleding", "Spoelen & Centrif.",
            "Alleen centrif.", "Alleen drogen", "Snel trommel spoelen", "Trommel drogen", "Kleine lading",
            "Grote lading", "XL lading", "Hygiëne", "Kinderkleding", "Kleurbescherming",
            "Zweetvlekken", "Hand/Wol", "Jeans", "Dekens opfrissen", "Kreukvrij",
            "ColdWash™", "Allergiene™", "BrightWhites™", "ezDispense™ Spuitmond reinig.",
        ],
        "settings": [
            "Wasinstellingen", "Drooginstellingen", "Trommelverlichting", "Antikreuk", "Toetsvergrendeling",
            '"Meer programma\'s" bew.', "Extra instellingen", "Wi-Fi", "Taal", "Signaal",
            "Smart Diagnosis", "Fabrieksinstellingen", "Open-source licenties", "Stoom",
            "TurboWash™", "Voorwas", "Weken", "Gebruik bakje 2",
            "Standaard wasmiddel", "Standaard wasverz.", "Energiebesparing",
            "Droogstart-signaal", "Vochtig-signaal", "Voorcentrifug.", "Start op afstand",
        ],
        "status": [
            "Wassen", "Spoelen", "Centrifugeren", "Drogen", "Afkoelen", "Reinigen", "Opfrissen",
            "Gepauzeerd", "Wassen voltooid", "Drogen voltooid", "Wassen & drogen klaar",
            "Antikreuk voltooid", "Wacht op voltooiing", "Lading detecteren",
            "Vervuiling detecteren", "Autom. wasmiddel", "Autom. wasverzachter",
        ],
        "labels": {
            "temp": "Temp.", "spin": "Centrif.", "soil": "Vuil", "dry_level": "Droogniveau",
            "dry_temp": "Droogtemp.", "dry_time": "Droogtijd", "delay": "Uitgestelde start",
        },
        "temps": ["Extra heet", "Heet", "Warm", "Koud", "Leidingkoud", "Middelhoog", "Hoog", "-"],
        "spins": ["Extra hoog", "Hoog", "Gemiddeld", "Laag", "Niet centrif.", "-"],
        "soils": ["Zwaar", "Middelzwaar", "Gemiddeld", "Middellicht", "Licht", "-"],
        "dry_levels": ["Zeer", "Normaal", "Vochtig", "-"],
        "dry_temps": ["Hoog", "Middelhoog", "Gemiddeld", "Middellaag", "Laag", "-"],
        "detergent": ["Maximum", "Meer", "Normaal", "Minder", "Minimum", "Uit"],
        "signal": ["Zeer hoog", "Hoog", "Gemiddeld", "Laag", "Uit"],
        "on_off": ["Aan", "Uit"],
        "saved": ["Opgeslagen.", "Programma geannuleerd.", "Uitgestelde start geannul.", "Start op afstand uit."],
        "hint_add": "Druk om was toe te voegen.",
        "hint_start": "Druk om programma te starten.",
        "hr": "u",
        "min": "min",
        "messages": [
            ["Deur open", "Open en sluit", "de deur opnieuw."],
            ["Deur ontgrendeld", "Controleer op objecten", "rond de deur", "en deurrubber."],
            ["Watertoevoer", "Draai de kraan open.", "Controleer waterdruk", "en toevoerslangen."],
            ["Afvoer", "Controleer op een", "verstopt filter of", "geknikte afvoerslang."],
            ["Onbalans", "Herschik de was", "en hervat het programma."],
            ["Te veel schuim", "Voeg niet meer dan de", "aanbevolen hoeveelheid", "wasmiddel toe."],
            ["Communicatiefout", "Het apparaat werkt niet", "goed. Bel de service", "als de fout terugkeert."],
            ["Start op afstand actief.", "Druk op de Aan/Uit-knop", "om dit uit te schakelen."],
            ["Houd OK 3 seconden", "ingedrukt om de toets-", "vergrendeling uit te zetten."],
            ["Er kunnen max. 10", "programma's getoond worden.", "Vink ongebruikte uit."],
        ],
        "codes": ["UNKNOWN ERR", "B3", "B4", "B5", "FE1", "EMPERATURE_SENSOR_I", "<PROC_W_SOAKING_PRC", "<CYCLE_D_AIR_DRY_1I"],
    },
    "it": {
        "cycles": [
            "Normale", "AI Lava & Asciuga", "Delicati", "Biancheria letto", "Intensivo", "Rapido",
            "Asciugamani", "Pulizia cestello", "Asciug. a tempo", "Abbigl. sportivo", "Risciacquo & Centr.",
            "Solo centrifuga", "Solo asciugatura", "Risciacquo cestello", "Asciug. cestello", "Carico piccolo",
            "Carico grande", "Carico XL", "Igienizzante", "Abbigl. bambini", "Cura colori",
            "Macchie sudore", "Mano/Lana", "Jeans", "Rinfresca coperte", "Stiro facile",
            "ColdWash™", "Allergiene™", "BrightWhites™", "ezDispense™ Pulizia ugello",
        ],
        "settings": [
            "Impostaz. lavaggio", "Impostaz. asciug.", "Luce cestello", "Antipiega", "Blocco comandi",
            'Modifica "Altri cicli"', "Altre impostazioni", "Wi-Fi", "Lingua", "Segnale",
            "Smart Diagnosis", "Ripristino fabbrica", "Licenze open source", "Vapore",
            "TurboWash™", "Prelavaggio", "Ammollo", "Uso vaschetta 2",
            "Detersivo predefinito", "Ammorbid. predefinito", "Risparmio energia",
            "Segnale avvio asciug.", "Segnale umido", "Centrifuga iniziale", "Avvio remoto",
        ],
        "status": [
            "Lavaggio", "Risciacquo", "Centrifuga", "Asciugatura", "Raffreddamento", "Pulizia", "Rinfresco",
            "In pausa", "Lavaggio terminato", "Asciugatura terminata", "Lava & Asciuga finito",
            "Antipiega terminato", "In attesa di completam.", "Rilevam. carico",
            "Rilevam. sporco", "Dosaggio auto detersivo", "Dosaggio auto ammorbid.",
        ],
        "labels": {
            "temp": "Temp.", "spin": "Centrifuga", "soil": "Sporco", "dry_level": "Livello asciug.",
            "dry_temp": "Temp. asciug.", "dry_time": "Tempo asciug.", "delay": "Avvio ritardato",
        },
        "temps": ["Extra calda", "Calda", "Tiepida", "Fredda", "Fredda rete", "Medio-alta", "Alta", "-"],
        "spins": ["Extra alta", "Alta", "Media", "Bassa", "No centrifuga", "-"],
        "soils": ["Molto sporco", "Medio-alto", "Medio", "Medio-basso", "Leggero", "-"],
        "dry_levels": ["Molto", "Normale", "Umido", "-"],
        "dry_temps": ["Alta", "Medio-alta", "Media", "Medio-bassa", "Bassa", "-"],
        "detergent": ["Massimo", "Di più", "Normale", "Di meno", "Minimo", "Spento"],
        "signal": ["Molto alto", "Alto", "Medio", "Basso", "Spento"],
        "on_off": ["Acceso", "Spento"],
        "saved": ["Salvato.", "Ciclo annullato.", "Avvio ritardato annullato.", "Disattivazione avvio remoto."],
        "hint_add": "Premere per aggiungere capi.",
        "hint_start": "Premere per avviare il ciclo.",
        "hr": "h",
        "min": "min",
        "messages": [
            ["Porta aperta", "Aprire e chiudere", "di nuovo la porta."],
            ["Porta sbloccata", "Verificare oggetti", "intorno alla porta", "e alla guarnizione."],
            ["Alimentaz. acqua", "Aprire il rubinetto.", "Verificare pressione", "e tubi di carico."],
            ["Scarico", "Verificare filtro", "scarico ostruito o", "tubo piegato."],
            ["Sbilanciamento", "Ridistribuire il bucato", "e riprendere il ciclo."],
            ["Troppa schiuma", "Non superare la dose", "di detersivo", "consigliata."],
            ["Errore comunicazione", "L'apparecchio non", "funziona bene. Se l'errore", "persiste, chiamare assist."],
            ["Avvio remoto attivo.", "Premere il tasto Power", "per disattivarlo."],
            ["Tenere premuto OK", "per 3 secondi per", "togliere il blocco comandi."],
            ["Si possono mostrare", "al massimo 10 cicli.", "Deselezionare i non usati."],
        ],
        "codes": ["UNKNOWN ERR", "B3", "B4", "B5", "FE1", "EMPERATURE_SENSOR_I", "<PROC_W_SOAKING_PRC", "<CYCLE_D_AIR_DRY_1I"],
    },
    "pt": {
        "cycles": [
            "Normal", "AI Lavar e Secar", "Delicados", "Roupa de cama", "Pesado", "Rápido",
            "Toalhas", "Limpeza do cesto", "Secagem por tempo", "Roupa esportiva", "Enxágue e Centrif.",
            "Só centrifugar", "Só secar", "Enxágue do cesto", "Secar cesto", "Carga pequena",
            "Carga grande", "Carga XL", "Higienização", "Roupa infantil", "Cuidado com cor",
            "Manchas de suor", "Mão/Lã", "Jeans", "Refrescar cobertor", "Passa fácil",
            "ColdWash™", "Allergiene™", "BrightWhites™", "ezDispense™ Limpar bico",
        ],
        "settings": [
            "Config. de lavagem", "Config. de secagem", "Luz do cesto", "Antirrugas", "Trava de controle",
            'Editar "Mais ciclos"', "Config. adicionais", "Wi-Fi", "Idioma", "Sinal",
            "Smart Diagnosis", "Restaurar fábrica", "Licenças open source", "Vapor",
            "TurboWash™", "Pré-lavagem", "Molho", "Uso do compart. 2",
            "Sabão padrão", "Amaciante padrão", "Economia de energia",
            "Sinal de início secagem", "Sinal de úmido", "Centrif. pré-secagem", "Início remoto",
        ],
        "status": [
            "Lavando", "Enxaguando", "Centrifugando", "Secando", "Resfriando", "Limpando", "Refrescando",
            "Pausado", "Lavagem concluída", "Secagem concluída", "Lavar e secar concluído",
            "Antirrugas concluído", "Aguardando conclusão", "Detectando carga",
            "Detectando sujeira", "Dosagem auto de sabão", "Dosagem auto amaciante",
        ],
        "labels": {
            "temp": "Temp.", "spin": "Centrífuga", "soil": "Sujeira", "dry_level": "Nível secagem",
            "dry_temp": "Temp. secagem", "dry_time": "Tempo secagem", "delay": "Início program.",
        },
        "temps": ["Muito quente", "Quente", "Morna", "Fria", "Fria torneira", "Médio-alta", "Alta", "-"],
        "spins": ["Extra alta", "Alta", "Média", "Baixa", "Sem centrif.", "-"],
        "soils": ["Pesada", "Médio-pesada", "Média", "Médio-leve", "Leve", "-"],
        "dry_levels": ["Muito", "Normal", "Úmido", "-"],
        "dry_temps": ["Alta", "Médio-alta", "Média", "Médio-baixa", "Baixa", "-"],
        "detergent": ["Máximo", "Mais", "Normal", "Menos", "Mínimo", "Desligado"],
        "signal": ["Muito alto", "Alto", "Médio", "Baixo", "Desligado"],
        "on_off": ["Ligado", "Desligado"],
        "saved": ["Salvo.", "Ciclo cancelado.", "Início programado cancelado.", "Desativando início remoto."],
        "hint_add": "Pressione para adicionar roupas.",
        "hint_start": "Pressione para iniciar o ciclo.",
        "hr": "h",
        "min": "min",
        "messages": [
            ["Porta aberta", "Abra e feche", "a porta novamente."],
            ["Porta destravada", "Verifique objetos", "ao redor da porta", "e da vedação."],
            ["Abastec. de água", "Abra a água.", "Verifique a pressão", "e as mangueiras."],
            ["Drenagem", "Verifique se o filtro", "ou a mangueira de", "dreno estão obstruídos."],
            ["Desbalanceamento", "Reorganize as roupas", "e retome o ciclo."],
            ["Excesso de espuma", "Não use mais que a", "quantidade de sabão", "recomendada."],
            ["Erro de comunicação", "O aparelho não funciona", "bem. Se o erro persistir,", "chame a assistência."],
            ["Início remoto ativo.", "Pressione o botão Power", "para desativá-lo."],
            ["Mantenha OK pressionado", "por 3 segundos para", "desligar a trava."],
            ["É possível exibir até", "10 ciclos.", "Desmarque os não usados."],
        ],
        "codes": ["UNKNOWN ERR", "B3", "B4", "B5", "FE1", "EMPERATURE_SENSOR_I", "<PROC_W_SOAKING_PRC", "<CYCLE_D_AIR_DRY_1I"],
    },
    "no": {
        "cycles": [
            "Normal", "AI Vask & Tørk", "Finvask", "Sengetøy", "Intensiv", "Hurtig",
            "Håndklær", "Trommelrens", "Tidsstyrt tørk", "Treningstøy", "Skyll & Sentrif.",
            "Kun sentrifug.", "Kun tørk", "Rask trommelskyll", "Trommeltørk", "Liten mengde",
            "Stor mengde", "XL mengde", "Hygiene", "Barneklær", "Fargepleie",
            "Svetteflekker", "Hånd/Ull", "Jeans", "Frisk opp tepper", "Lettstelt",
            "ColdWash™", "Allergiene™", "BrightWhites™", "ezDispense™ Dyserens",
        ],
        "settings": [
            "Vaskeinnstillinger", "Tørkeinnstillinger", "Trommellys", "Antikrøll", "Tastelås",
            'Rediger "Flere prog."', "Flere innstillinger", "Wi-Fi", "Språk", "Signal",
            "Smart Diagnosis", "Fabrikkinnstilling", "Åpen kildekode-lis.", "Damp",
            "TurboWash™", "Forvask", "Bløtlegging", "Bruk av kammer 2",
            "Standard vaskemiddel", "Standard tøymykner", "Energisparing",
            "Tørkestart-signal", "Fukt-signal", "Forsentrifug.", "Fjernstart",
        ],
        "status": [
            "Vasker", "Skyller", "Sentrifugerer", "Tørker", "Kjøler ned", "Rengjør", "Frisker opp",
            "Pause", "Vask fullført", "Tørk fullført", "Vask & tørk fullført",
            "Antikrøll fullført", "Venter på fullføring", "Registrerer mengde",
            "Registrerer smuss", "Auto vaskemiddel", "Auto tøymykner",
        ],
        "labels": {
            "temp": "Temp.", "spin": "Sentrif.", "soil": "Smuss", "dry_level": "Tørkenivå",
            "dry_temp": "Tørketemp.", "dry_time": "Tørketid", "delay": "Utsatt start",
        },
        "temps": ["Ekstra varm", "Varm", "Lunken", "Kald", "Springkald", "Middels høy", "Høy", "-"],
        "spins": ["Ekstra høy", "Høy", "Middels", "Lav", "Ingen sentrif.", "-"],
        "soils": ["Mye", "Middels mye", "Middels", "Middels lite", "Lite", "-"],
        "dry_levels": ["Svært", "Normal", "Fuktig", "-"],
        "dry_temps": ["Høy", "Middels høy", "Middels", "Middels lav", "Lav", "-"],
        "detergent": ["Maksimum", "Mer", "Normal", "Mindre", "Minimum", "Av"],
        "signal": ["Svært høy", "Høy", "Middels", "Lav", "Av"],
        "on_off": ["På", "Av"],
        "saved": ["Lagret.", "Programmet er avbrutt.", "Utsatt start er avbrutt.", "Slår av fjernstart."],
        "hint_add": "Trykk for å legge til klær.",
        "hint_start": "Trykk for å starte programmet.",
        "hr": "t",
        "min": "min",
        "messages": [
            ["Døren er åpen", "Åpne og lukk", "døren igjen."],
            ["Døren er ulåst", "Sjekk for gjenstander", "rundt døren", "og dørpakningen."],
            ["Vanntilførsel", "Skru på vannet.", "Sjekk vanntrykk", "og tilførselsslanger."],
            ["Avløp", "Sjekk om avløpsfilter", "er tett eller", "slangen er bøyd."],
            ["Ubalanse", "Omfordel tøyet", "og fortsett programmet."],
            ["For mye skum", "Ikke bruk mer enn", "anbefalt mengde", "vaskemiddel."],
            ["Kommunikasjonsfeil", "Maskinen fungerer ikke", "som den skal. Ring service", "hvis feilen vedvarer."],
            ["Fjernstart er på.", "Trykk på Power-knappen", "for å slå av fjernstart."],
            ["Hold inne OK", "i 3 sekunder for", "å slå av tastelåsen."],
            ["Maks 10 programmer", "kan vises.", "Fjern merking for ubrukte."],
        ],
        "codes": ["UNKNOWN ERR", "B3", "B4", "B5", "FE1", "EMPERATURE_SENSOR_I", "<PROC_W_SOAKING_PRC", "<CYCLE_D_AIR_DRY_1I"],
    },
}

LANG.update(ZH_LANG)


def contains_cjk(text: str) -> bool:
    return any(
        0x3400 <= ord(ch) <= 0x4DBF
        or 0x4E00 <= ord(ch) <= 0x9FFF
        or 0xF900 <= ord(ch) <= 0xFAFF
        for ch in text
    )


def resolve_font_path(kind: str, cjk: bool = False) -> str | None:
    candidates = CJK_FONT_CANDIDATES[kind] + FONT_CANDIDATES[kind] if cjk else FONT_CANDIDATES[kind]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


@lru_cache(maxsize=512)
def get_font(size: int, bold: bool = False, cjk: bool = False) -> ImageFont.ImageFont:
    kind = "bold" if bold else "regular"
    path = resolve_font_path(kind, cjk=cjk)
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(path, size)


def size_class(size: int) -> str:
    if size < 18:
        return "small"
    if size < 32:
        return "medium"
    if size < 60:
        return "large"
    return "xl"


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_size: int,
             min_size: int = 10, bold: bool = False) -> tuple[ImageFont.ImageFont, int]:
    cjk = contains_cjk(text)
    for size in range(max_size, min_size - 1, -1):
        fnt = get_font(size, bold=bold, cjk=cjk)
        width, _ = text_size(draw, text, fnt)
        if width <= max_width:
            return fnt, size
    return get_font(min_size, bold=bold, cjk=cjk), min_size


def has_drawn_symbol(text: str) -> bool:
    return "▶Ⅱ" in text or "™" in text


def split_symbol_segments(text: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    idx = 0
    while idx < len(text):
        if text.startswith("▶Ⅱ", idx):
            segments.append(("play_pause", "▶Ⅱ"))
            idx += 2
        elif text.startswith("™", idx):
            segments.append(("tm", "™"))
            idx += 1
        else:
            start = idx
            while idx < len(text) and not text.startswith("▶Ⅱ", idx) and not text.startswith("™", idx):
                idx += 1
            segments.append(("text", text[start:idx]))
    return segments


def symbol_text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, size: int) -> tuple[int, int]:
    width = 0
    height = 0
    for kind, value in split_symbol_segments(text):
        if kind == "text":
            seg_w, seg_h = text_size(draw, value, fnt)
        elif kind == "play_pause":
            seg_w, seg_h = max(18, int(size * 1.25)), max(12, int(size * 0.8))
        else:
            tm_font = get_font(max(6, int(size * 0.42)))
            seg_w, seg_h = text_size(draw, "TM", tm_font)
        width += seg_w
        height = max(height, seg_h)
    return width, max(height, size)


def fit_symbol_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_size: int,
                    min_size: int = 10, bold: bool = False) -> tuple[ImageFont.ImageFont, int]:
    cjk = contains_cjk(text)
    for size in range(max_size, min_size - 1, -1):
        fnt = get_font(size, bold=bold, cjk=cjk)
        width, _ = symbol_text_size(draw, text, fnt, size)
        if width <= max_width:
            return fnt, size
    return get_font(min_size, bold=bold, cjk=cjk), min_size


def add_symbol_text(draw: ImageDraw.ImageDraw, elements: list[Element], text: str, x: int, y: int,
                    fnt: ImageFont.ImageFont, size: int, canvas_size: tuple[int, int],
                    color: tuple[int, int, int] = WHITE, selected: bool = False,
                    color_class: str = "white") -> list[int]:
    cursor = x
    for kind, value in split_symbol_segments(text):
        if kind == "text":
            draw.text((cursor, y), value, fill=color, font=fnt)
            seg_w, _ = text_size(draw, value, fnt)
            cursor += seg_w
        elif kind == "play_pause":
            symbol_w = max(18, int(size * 1.25))
            top = y + max(1, int(size * 0.18))
            bottom = y + max(10, int(size * 0.82))
            mid = (top + bottom) // 2
            tri_w = max(7, int(size * 0.35))
            draw.polygon([(cursor + 1, top), (cursor + 1, bottom), (cursor + tri_w, mid)], fill=color)
            bar_x = cursor + tri_w + max(4, int(size * 0.12))
            bar_h = max(9, bottom - top)
            bar_w = max(2, int(size * 0.08))
            draw.rectangle([bar_x, top, bar_x + bar_w, top + bar_h], fill=color)
            draw.rectangle([bar_x + bar_w + 3, top, bar_x + bar_w * 2 + 3, top + bar_h], fill=color)
            cursor += symbol_w
        else:
            tm_font = get_font(max(6, int(size * 0.42)))
            draw.text((cursor, y), "TM", fill=color, font=tm_font)
            seg_w, _ = text_size(draw, "TM", tm_font)
            cursor += seg_w

    width, height = symbol_text_size(draw, text, fnt, size)
    bbox = clamp_bbox([x, y, x + width, y + height], canvas_size)
    elements.append(Element("text", bbox, text=text, selected=selected, color_class=color_class, size_class=size_class(size)))
    return bbox


def clamp_bbox(bbox: list[int], canvas_size: tuple[int, int]) -> list[int]:
    width, height = canvas_size
    x1, y1, x2, y2 = bbox
    return [max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))]


def add_text(draw: ImageDraw.ImageDraw, elements: list[Element], text: str, x: int, y: int,
             fnt: ImageFont.ImageFont, size: int, canvas_size: tuple[int, int],
             color: tuple[int, int, int] = WHITE, selected: bool = False,
             color_class: str = "white") -> list[int]:
    draw.text((x, y), text, fill=color, font=fnt)
    bbox_raw = draw.textbbox((x, y), text, font=fnt)
    bbox = clamp_bbox([bbox_raw[0], bbox_raw[1], bbox_raw[2], bbox_raw[3]], canvas_size)
    elements.append(Element("text", bbox, text=text, selected=selected, color_class=color_class, size_class=size_class(size)))
    return bbox


def add_centered(draw: ImageDraw.ImageDraw, elements: list[Element], text: str, y: int,
                 canvas_size: tuple[int, int], max_size: int, min_size: int = 10,
                 bold: bool = False, color: tuple[int, int, int] = WHITE,
                 selected: bool = False, color_class: str = "white",
                 max_width_ratio: float = 0.92) -> list[int]:
    width, _ = canvas_size
    if has_drawn_symbol(text):
        fnt, size = fit_symbol_font(draw, text, int(width * max_width_ratio), max_size, min_size, bold)
        text_width, _ = symbol_text_size(draw, text, fnt, size)
        x = (width - text_width) // 2
        return add_symbol_text(draw, elements, text, x, y, fnt, size, canvas_size, color, selected, color_class)
    fnt, size = fit_font(draw, text, int(width * max_width_ratio), max_size, min_size, bold)
    text_width, _ = text_size(draw, text, fnt)
    x = (width - text_width) // 2
    return add_text(draw, elements, text, x, y, fnt, size, canvas_size, color, selected, color_class)


def make_background(size: tuple[int, int], rng: random.Random) -> tuple[Image.Image, str]:
    kind = rng.choice(["solid_black", "solid_black", "soft_panel", "bottom_panel", "dim_gradient"])
    shade = rng.randint(0, 8)
    img = Image.new("RGB", size, (shade, shade, shade))
    draw = ImageDraw.Draw(img)
    width, height = size
    if kind == "soft_panel":
        y1 = rng.randint(height // 5, height // 3)
        y2 = rng.randint(height * 2 // 3, height - 20)
        draw.rectangle([8, y1, width - 8, y2], fill=(22, 22, 22))
    elif kind == "bottom_panel":
        draw.rectangle([0, height * 2 // 3, width, height], fill=(42, 42, 42))
    elif kind == "dim_gradient":
        for y in range(0, height, 12):
            value = min(35, shade + y * 35 // max(1, height))
            draw.rectangle([0, y, width, min(height, y + 12)], fill=(value, value, value))
    if rng.random() < 0.25:
        draw.rectangle([0, 0, width, 2], fill=(18, 18, 18))
        draw.rectangle([0, height - 2, width, height], fill=(18, 18, 18))
    return img, kind


def draw_divider(draw: ImageDraw.ImageDraw, y: int, width: int) -> None:
    draw.line([(24, y), (width - 24, y)], fill=(70, 70, 70), width=1)


def draw_progress(draw: ImageDraw.ImageDraw, rng: random.Random, y: int, width: int) -> None:
    margin = 34
    draw.line([(margin, y), (width - margin, y)], fill=(70, 70, 70), width=2)
    pos = margin + int((width - 2 * margin) * rng.uniform(0.05, 0.95))
    draw.line([(margin, y), (pos, y)], fill=BLUE, width=2)


def rand_time(data: dict, rng: random.Random, allow_long: bool = True) -> str:
    if allow_long and rng.random() < 0.08:
        return f"{rng.randint(10, 23)} {data['hr']} {rng.choice([30, 45, 59])} {data['min']}"
    hours = rng.choice([0, 0, 1, 1, 2, 3])
    minutes = rng.choice([0, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 59])
    if hours == 0:
        return f"{max(1, minutes)} {data['min']}"
    if minutes == 0:
        return f"{hours} {data['hr']}"
    return f"{hours} {data['hr']} {minutes} {data['min']}"


def rand_hard_timer(data: dict, rng: random.Random, english_hr: bool = False) -> str:
    hr = "hr" if english_hr else data["hr"]
    minute_unit = data["min"]
    common = [
        f"1 {hr} 30 {minute_unit}",
        f"1 {hr} 30{minute_unit}",
        f"1{hr} 30 {minute_unit}",
        f"1{hr} 30{minute_unit}",
        f"1 {hr} 10 {minute_unit}",
        f"1 {hr} 10{minute_unit}",
        f"40 {minute_unit}",
        f"40{minute_unit}",
        f"15 {minute_unit}",
        f"15{minute_unit}",
    ]
    if rng.random() < 0.45:
        return rng.choice(common)
    hours = rng.choice([0, 0, 0, 1, 1, 1, 2, 2, 3, 4])
    minutes = rng.choice([1, 5, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 59])
    if hours == 0:
        return rng.choice([f"{minutes} {minute_unit}", f"{minutes}{minute_unit}"])
    if minutes == 0:
        return rng.choice([f"{hours} {hr}", f"{hours}{hr}"])
    return rng.choice([
        f"{hours} {hr} {minutes} {minute_unit}",
        f"{hours} {hr} {minutes}{minute_unit}",
        f"{hours}{hr} {minutes} {minute_unit}",
        f"{hours}{hr} {minutes}{minute_unit}",
    ])


def option_value(data: dict, label_key: str, rng: random.Random) -> str:
    pools = {
        "temp": "temps",
        "spin": "spins",
        "soil": "soils",
        "dry_level": "dry_levels",
        "dry_temp": "dry_temps",
    }
    return rng.choice(data[pools[label_key]])


def draw_line_stack(draw: ImageDraw.ImageDraw, elements: list[Element], lines: list[str],
                    canvas_size: tuple[int, int], start_y: int, max_size: int,
                    first_bold: bool = False, selected_index: int | None = None,
                    line_gap: int = 7, max_width_ratio: float = 0.92) -> None:
    y = start_y
    for index, line in enumerate(lines):
        bold = first_bold and index == 0
        color = WHITE if selected_index is None or index == selected_index else DIM
        bbox = add_centered(
            draw, elements, line, y, canvas_size, max_size=max_size if index == 0 else max(12, max_size - 3),
            min_size=9, bold=bold, color=color, selected=index == selected_index,
            color_class="white" if color == WHITE else "gray", max_width_ratio=max_width_ratio,
        )
        y = bbox[3] + line_gap


def gen_cycle_status(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    img, bg = make_background(canvas_size, rng)
    draw = ImageDraw.Draw(img)
    elements: list[Element] = []
    width, height = canvas_size
    cycle = rng.choice(data["cycles"])
    status = rng.choice(data["status"])
    time_text = rand_time(data, rng)
    lines = [cycle, time_text, status]
    draw_line_stack(draw, elements, lines, canvas_size, int(height * 0.12), max(18, height // 9), first_bold=False, line_gap=9)
    if rng.random() < 0.8:
        draw_progress(draw, rng, int(height * 0.64), width)
    if rng.random() < 0.65:
        hint = data["hint_add"]
        add_centered(draw, elements, hint, int(height * 0.80), canvas_size, max_size=max(12, height // 15), min_size=8, max_width_ratio=0.96)
        lines.append(hint)
    return img, elements, bg, "\n".join(lines)


def gen_cycle_options(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    img, bg = make_background(canvas_size, rng)
    draw = ImageDraw.Draw(img)
    elements: list[Element] = []
    cycle = rng.choice(data["cycles"])
    mode = rng.choice(["wash", "dry", "wash_dry"])
    if mode == "dry":
        rows = [
            cycle,
            f"{data['labels']['dry_temp']} {option_value(data, 'dry_temp', rng)}",
            f"{data['labels']['dry_level']} {rng.choice(data['dry_levels'])}",
            f"{data['labels']['dry_time']} {rng.choice(['-', '10min', '20min', '30min', '60min', '120min'])}",
        ]
    else:
        rows = [
            cycle,
            f"{data['labels']['temp']} {option_value(data, 'temp', rng)}",
            f"{data['labels']['spin']} {option_value(data, 'spin', rng)}",
            f"{data['labels']['soil']} {option_value(data, 'soil', rng)}",
        ]
        if mode == "wash_dry" and rng.random() < 0.55:
            rows.append(f"{data['labels']['dry_level']} {rng.choice(data['dry_levels'])}")
    draw_line_stack(draw, elements, rows, canvas_size, 28, max_size=max(17, canvas_size[1] // 11), first_bold=True, line_gap=8)
    return img, elements, bg, "\n".join(rows)


def gen_picker(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    img, bg = make_background(canvas_size, rng)
    draw = ImageDraw.Draw(img)
    elements: list[Element] = []
    width, height = canvas_size
    picker_type = rng.choice(["temp", "spin", "soil", "dry_time", "delay", "detergent", "softener", "signal"])
    if picker_type in {"temp", "spin", "soil"}:
        title = data["labels"][picker_type]
        options = data[{"temp": "temps", "spin": "spins", "soil": "soils"}[picker_type]]
    elif picker_type == "dry_time":
        title = data["labels"]["dry_time"]
        options = [f"{n} {data['min']}" for n in range(10, 155, 5)]
    elif picker_type == "delay":
        title = data["labels"]["delay"]
        options = ["Off"] + [f"{h}{data['hr']}" for h in range(1, 25)] + [f"{h}{data['hr']} 30{data['min']}" for h in range(1, 24)]
    elif picker_type == "signal":
        title = "Signal"
        options = data["signal"]
    else:
        title = data["settings"][-6 if picker_type == "detergent" else -5]
        options = [f"{n / 10:.1f} oz" for n in range(3, 41)] + data["detergent"]
    if len(options) < 3:
        options = options * 3
    start = rng.randint(0, max(0, len(options) - 3))
    visible_count = 3 if height <= 260 else rng.choice([3, 4])
    visible = options[start:start + visible_count]
    if len(visible) < 3:
        visible = (visible + options)[:3]
    add_centered(draw, elements, title, int(height * 0.07), canvas_size, max_size=max(17, height // 12), bold=False)
    middle_y = int(height * 0.33)
    row_gap = max(34, height // 5)
    raw = [title]
    for idx, item in enumerate(visible):
        y = middle_y + idx * row_gap
        selected = idx == min(1, len(visible) - 1)
        if selected:
            draw.rectangle([18, y - 8, width - 18, y + row_gap - 12], fill=PANEL)
            draw_divider(draw, y - 10, width)
            draw_divider(draw, y + row_gap - 12, width)
        color = WHITE if selected else DIM
        add_centered(draw, elements, item, y, canvas_size, max_size=max(18, height // 9), bold=selected, color=color, selected=selected, color_class="white" if selected else "gray")
        raw.append(item)
    return img, elements, bg, "\n".join(raw)


def gen_settings_list(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    img, bg = make_background(canvas_size, rng)
    draw = ImageDraw.Draw(img)
    elements: list[Element] = []
    width, height = canvas_size
    settings = data["settings"]
    start = rng.randint(0, len(settings) - 4)
    visible = settings[start:start + 4]
    if rng.random() < 0.35:
        language_values = data.get("language_menu_values", ["English", "Español", "Français", "Connected", "Disconnected"])
        value = rng.choice(data["on_off"] + data["signal"] + language_values)
        visible[2] = value
    y = int(height * 0.08)
    raw = []
    for idx, item in enumerate(visible):
        selected = idx == 1 and rng.random() < 0.55
        color = WHITE if idx <= 2 else DIM
        bbox = add_centered(draw, elements, item, y, canvas_size, max_size=max(16, height // 12), bold=selected, color=color, selected=selected, color_class="white" if color == WHITE else "gray")
        raw.append(item)
        y = bbox[3] + max(12, height // 18)
        if idx == 0 and rng.random() < 0.35:
            draw_divider(draw, y - 4, width)
    return img, elements, bg, "\n".join(raw)


def gen_edit_list(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    img, bg = make_background(canvas_size, rng)
    draw = ImageDraw.Draw(img)
    elements: list[Element] = []
    title = '"More Cycles" Edit' if lang == "en" else data["settings"][5]
    cycles = data["cycles"]
    start = rng.randint(0, len(cycles) - 3)
    items = cycles[start:start + 3]
    if lang == "en" and rng.random() < 0.12:
        items[0] = rng.choice(["<CYCLE_D_AIR_DR", "<CYCLE_W_SMALL", "<CYCLE_RINSE_SPIN"])
    lines = [title] + items
    draw_line_stack(draw, elements, lines, canvas_size, 22, max_size=max(16, canvas_size[1] // 12), first_bold=False, line_gap=11)
    return img, elements, bg, "\n".join(lines)


def gen_message(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    img, bg = make_background(canvas_size, rng)
    draw = ImageDraw.Draw(img)
    elements: list[Element] = []
    if rng.random() < 0.12:
        lines = [rng.choice(data["codes"])]
        if rng.random() < 0.45:
            service_lines = {
                "en": ["Unplug the appliance", "and call for service."],
                "fr": ["Débranchez l'appareil", "et appelez le service."],
                "es": ["Desconecte el aparato", "y llame al servicio."],
                "de": ["Gerät ausstecken", "und Service rufen."],
                "nl": ["Haal de stekker eruit", "en bel de service."],
                "it": ["Scollegare l'apparecchio", "e chiamare assistenza."],
                "pt": ["Desligue o aparelho", "e chame a assistência."],
                "no": ["Trekk ut støpselet", "og kontakt service."],
            }
            lines.extend(data.get("service_lines", service_lines.get(lang, service_lines["en"])))
    else:
        lines = rng.choice(data["messages"])
    draw_line_stack(draw, elements, lines, canvas_size, int(canvas_size[1] * 0.15), max_size=max(16, canvas_size[1] // 12), first_bold=True, line_gap=8, max_width_ratio=0.95)
    return img, elements, bg, "\n".join(lines)


def gen_toast(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    img = Image.new("RGB", canvas_size, (0, 0, 0))
    draw = ImageDraw.Draw(img)
    width, height = canvas_size
    draw.rectangle([0, int(height * 0.62), width, height], fill=(50, 50, 50))
    elements: list[Element] = []
    msg = rng.choice(data["saved"])
    add_centered(draw, elements, msg, int(height * 0.76), canvas_size, max_size=max(16, height // 12), min_size=9)
    return img, elements, "toast_bottom", msg


def draw_check_icon(draw: ImageDraw.ImageDraw, x: int, y: int, rng: random.Random) -> None:
    radius = rng.choice([7, 8, 9])
    color = rng.choice([BLUE, (95, 95, 95), (190, 190, 190)])
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=2)
    if rng.random() < 0.75:
        draw.line([(x - 4, y), (x - 1, y + 4), (x + 6, y - 5)], fill=color, width=2)


def gen_hard_schedule(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    img, bg = make_background(canvas_size, rng)
    draw = ImageDraw.Draw(img)
    width, height = canvas_size
    elements: list[Element] = []
    cycle = rng.choice(["Normal", data["cycles"][0], data["cycles"][1]])
    schedule_text = {
        "en": (["Today", "TMRW."], ["AM 12:00", "AM 12:30", "PM 12:00", "PM 12:30"], "Delay Start is on"),
        "fr": (["Auj.", "Dem."], ["AM 12:00", "AM 12:30", "PM 12:00", "PM 12:30"], "Départ différé activé"),
        "es": (["Hoy", "MAÑ."], ["a.m. 12:00", "a.m. 12:30", "p.m. 12:00", "p.m. 12:30"], "El Inicio retardado está activo"),
        "de": (["Heute", "Morgen"], ["AM 12:00", "AM 12:30", "PM 12:00", "PM 12:30"], "Startzeitvorwahl aktiv"),
        "nl": (["Vandaag", "Morgen"], ["AM 12:00", "AM 12:30", "PM 12:00", "PM 12:30"], "Uitgestelde start actief"),
        "it": (["Oggi", "Domani"], ["AM 12:00", "AM 12:30", "PM 12:00", "PM 12:30"], "Avvio ritardato attivo"),
        "pt": (["Hoje", "Amanhã"], ["AM 12:00", "AM 12:30", "PM 12:00", "PM 12:30"], "Início programado ativo"),
        "no": (["I dag", "I morgen"], ["AM 12:00", "AM 12:30", "PM 12:00", "PM 12:30"], "Utsatt start aktiv"),
    }
    days, clocks, status = data.get("schedule_text", schedule_text.get(lang, schedule_text["en"]))
    day = rng.choice(days)
    clock = rng.choice(clocks)
    lines = [cycle, day, clock, status, data["hint_add"]]
    y = int(height * 0.06)
    for idx, line in enumerate(lines):
        max_size = max(13, height // (10 if idx < 3 else 16))
        max_width = 0.94 if idx < 3 else 0.98
        bbox = add_centered(draw, elements, line, y, canvas_size, max_size=max_size, min_size=8, max_width_ratio=max_width)
        y = bbox[3] + max(5, height // 45)
    if rng.random() < 0.7:
        draw_progress(draw, rng, int(height * 0.68), width)
    return img, elements, bg, "\n".join(lines)


def gen_hard_duration(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    img, bg = make_background(canvas_size, rng)
    draw = ImageDraw.Draw(img)
    width, height = canvas_size
    elements: list[Element] = []
    cycle = rng.choice([data["cycles"][1], "Normal", rng.choice(data["cycles"])])
    duration = rand_hard_timer(data, rng, english_hr=lang == "fr" and rng.random() < 0.7)
    status = rng.choice(data["status"][:6])
    lines = [cycle, duration, status]
    if rng.random() < 0.8:
        lines.append(data["hint_add"])
    draw_line_stack(draw, elements, lines, canvas_size, int(height * 0.09), max_size=max(18, height // 10), line_gap=max(6, height // 45), max_width_ratio=0.97)
    if rng.random() < 0.65:
        draw_progress(draw, rng, int(height * 0.65), width)
    return img, elements, bg, "\n".join(lines)


def gen_hard_progress_timer(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    img, bg = make_background(canvas_size, rng)
    draw = ImageDraw.Draw(img)
    width, height = canvas_size
    elements: list[Element] = []
    cycle_pool = data.get("progress_cycles", {
        "fr": ["Normal", "Lavage AI", "Jeans", "Robes", "Hygiène", "Literie", "Nettoyage de cuve", "Lavage nocturne"],
        "en": ["Normal", "AI Wash", "Jeans", "Bedding", "Sanitary", "Tub Clean", "Night Wash"],
        "es": ["Normal", "Lavado AI", "Jeans", "Ropa de cama", "Higiénico", "Limpieza de tina"],
    }.get(lang, ["Normal", data["cycles"][1], rng.choice(data["cycles"])]))
    status_pool = data.get("progress_status", {
        "fr": ["Lavage en cours", "Rinçage en cours", "Essorage en cours", "Trempage", "Prélavage en cours", "Nettoyage", "En attente d'être terminé"],
        "en": ["Washing", "Rinsing", "Spinning", "Soaking", "Pre-washing", "Cleaning", "Waiting to be completed"],
        "es": ["Lavando", "Enjuagando", "Centrifugando", "Remojo", "Prelavado", "Limpiando", "Esperando finalizar"],
    }.get(lang, data["status"]))
    hint_pool = data.get("progress_hints", {
        "fr": ["Appuyez sur ▶Ⅱ pour ajoute", "Appuyez sur ▶Ⅱ pour ajouter", "Appuyez ▶Ⅱ pour démarrer.", "Appuyez sur ▶Ⅱ pour démarrer"],
        "en": ["Press ▶Ⅱ to add garments", "Press ▶Ⅱ to start the cycle", "Press ▶Ⅱ to resume"],
        "es": ["Presione ▶Ⅱ para agregar", "Presione ▶Ⅱ para iniciar", "Presione ▶Ⅱ para continuar"],
    }.get(lang, [data["hint_add"], data["hint_start"]]))
    lines = [rng.choice(cycle_pool), rand_hard_timer(data, rng, english_hr=lang == "fr"), rng.choice(status_pool)]
    if rng.random() < 0.92:
        lines.append(rng.choice(hint_pool))
    draw_line_stack(
        draw,
        elements,
        lines,
        canvas_size,
        int(height * rng.uniform(0.06, 0.12)),
        max_size=max(18, height // 10),
        line_gap=max(6, height // 45),
        max_width_ratio=0.98,
    )
    if rng.random() < 0.85:
        draw_progress(draw, rng, int(height * rng.uniform(0.64, 0.72)), width)
    return img, elements, bg, "\n".join(lines)


def gen_hard_numeric_options(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    img, bg = make_background(canvas_size, rng)
    draw = ImageDraw.Draw(img)
    width, height = canvas_size
    elements: list[Element] = []
    if rng.random() < 0.55:
        title = {
            "en": "Extra Rinse", "fr": "Rinçage extra", "es": "Enjuague extra",
            "de": "Extra Spülen", "nl": "Extra spoelen", "it": "Extra risciacquo",
            "pt": "Enxágue extra", "no": "Ekstra skylling",
        }.get(lang, data.get("extra_rinse", "Extra Rinse"))
        values = rng.choice([["+2", "+1", "0"], ["+1", "0"]])
    else:
        title = data["labels"]["dry_time"]
        center = rng.choice([45, 65, 95, 105, 130])
        values = [str(center + 5), f"{center} {data['min']}", str(max(0, center - 5))]
    lines = [title] + values
    add_centered(draw, elements, title, int(height * 0.07), canvas_size, max_size=max(17, height // 12), min_size=9)
    row_gap = max(36, height // 5)
    y = int(height * 0.28)
    for idx, value in enumerate(values):
        selected = idx == min(1, len(values) - 1)
        if selected:
            draw.rectangle([max(12, width // 12), y - 7, width - max(12, width // 12), y + row_gap - 12], fill=PANEL)
        add_centered(draw, elements, value, y, canvas_size, max_size=max(20, height // 8), min_size=11, bold=selected, color=WHITE if selected else DIM, selected=selected, color_class="white" if selected else "gray")
        y += row_gap
    return img, elements, bg, "\n".join(lines)


def gen_hard_dense_description(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    descriptions = {
        "en": [
            ["Tumble", "Washes clothes with a", "light tapping action to", "untangle them."],
            ["Swing", "Washes clothes with a", "light, swinging action that", "minimizes damage to", "clothing."],
            ["Scrub", "Washes clothes with a", "scrubbing action that", "dissolves the detergent", "quickly."],
        ],
        "fr": [
            ["Balancement", "Lave les vêtements avec", "un mouvement léger", "pour réduire les plis."],
            ["Frottement", "Lave avec une action", "de frottement qui dissout", "rapidement la lessive."],
            ["Défroissage", "Aide à réduire", "les plis pendant", "le cycle."],
        ],
        "es": [
            ["Balanceo", "Lava la ropa con", "un movimiento suave", "para reducir daños."],
            ["Frotar", "Lava con una acción", "de frotado que disuelve", "rápido el detergente."],
            ["Tumble", "Lava la ropa con", "golpes ligeros para", "desenredarla."],
        ],
        "de": [
            ["Trommeln", "Wäscht mit sanften", "Trommelbewegungen", "gegen Verknoten."],
            ["Schwingen", "Wäscht mit leichter", "Schwingbewegung", "für weniger Schäden."],
            ["Reiben", "Löst Waschmittel", "mit Reibbewegungen", "schneller auf."],
        ],
        "nl": [
            ["Trommelen", "Wast met lichte", "trommelbewegingen", "om te ontwarren."],
            ["Schommelen", "Wast met zachte", "bewegingen en beperkt", "kledingschade."],
            ["Schrobben", "Wast met schrobactie", "die wasmiddel", "snel oplost."],
        ],
        "it": [
            ["Rivoltamento", "Lava i capi con", "tocchi leggeri", "per separarli."],
            ["Oscillazione", "Lava con movimento", "delicato e riduce", "i danni ai capi."],
            ["Strofinio", "Lava con azione", "di strofinio che scioglie", "rapido il detersivo."],
        ],
        "pt": [
            ["Tombar", "Lava roupas com", "toques leves para", "desembaraçar."],
            ["Balançar", "Lava com movimento", "suave e reduz", "danos às roupas."],
            ["Esfregar", "Lava com ação", "de esfregar que dissolve", "o sabão rapidamente."],
        ],
        "no": [
            ["Tromling", "Vasker klær med", "lett tromling for", "å løsne dem."],
            ["Sving", "Vasker med myk", "svingbevegelse som", "skåner klærne."],
            ["Skrubb", "Vasker med skrubb", "som løser opp", "vaskemiddel raskt."],
        ],
    }
    img, bg = make_background(canvas_size, rng)
    draw = ImageDraw.Draw(img)
    elements: list[Element] = []
    lines = rng.choice(data.get("dense_descriptions", descriptions.get(lang, descriptions["en"])))
    y = int(canvas_size[1] * 0.08)
    for idx, line in enumerate(lines):
        max_size = max(15, canvas_size[1] // (13 if idx == 0 else 18))
        bbox = add_centered(draw, elements, line, y, canvas_size, max_size=max_size, min_size=8, bold=idx == 0, max_width_ratio=0.96)
        y = bbox[3] + max(5, canvas_size[1] // 60)
    return img, elements, bg, "\n".join(lines)


def gen_hard_blue_microtext(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    blue_lines = {
        "en": [
            ["TurboWash™", "Washes clothes quickly", "while saving energy."],
            ["Scrub", "Washes clothes with a", "scrubbing action that", "dissolves the detergent", "quickly."],
            ["Filtration", "Filters water through", "the laundry for a", "cleaner wash."],
            ["Stepping", "Moves clothes through", "water in a gentle", "stepping motion."],
            ["Rolling", "Rolls clothes below", "the water level for", "a quiet wash."],
            ["Swing", "Washes clothes with a", "light swinging action", "to minimize damage."],
            ["Tumble", "Washes clothes with a", "light tapping action", "to untangle them."],
            ["More Cycles", "Reduce wrinkles", "Use with Normal", "and Bedding."],
        ],
        "fr": [
            ["Plus de cycles", "Réduire les plis", "Utiliser avec Normal", "et Literie."],
            ["Plus de cycles", "TurboWash™", "Lavage rapide", "avec jet puissant."],
            ["Plus de cycles", "Frottement", "Dissout la lessive", "rapidement."],
            ["Plus de cycles", "Filtration", "Filtre l'eau pendant", "le lavage."],
            ["Plus de cycles", "Balancement", "Lave avec un", "mouvement léger."],
            ["Plus de cycles", "Défroissage", "Aide à réduire", "les plis."],
            ["Plus de cycles", "Rinçage+Essorage", "Ajoutez un rinçage", "au cycle."],
            ["Plus de cycles", "Séchage minuté", "Choisissez la durée", "de séchage."],
        ],
        "es": [
            ["Más ciclos", "Reducir arrugas", "Usar con Normal", "y Ropa de cama."],
            ["Más ciclos", "TurboWash™", "Lavado rápido", "con chorro potente."],
            ["Más ciclos", "Frotar", "Disuelve el detergente", "rápidamente."],
            ["Más ciclos", "Filtración", "Filtra el agua", "durante el lavado."],
            ["Más ciclos", "Balanceo", "Lava con un", "movimiento suave."],
        ],
    }
    lines = rng.choice(data.get("blue_microtext", blue_lines.get(lang, blue_lines["en"])))
    width, height = canvas_size
    elements: list[Element] = []
    palette = [(0, 174, 239), (0, 145, 219), (0, 118, 196), (11, 91, 170)]
    base = rng.choice(palette)
    bg_mode = rng.choice(["blue_panel", "blue_panel", "blue_full", "blue_pill"])
    img = Image.new("RGB", canvas_size, (5, 7, 10) if bg_mode != "blue_full" else base)
    draw = ImageDraw.Draw(img)

    if bg_mode == "blue_full":
        for y in range(height):
            scale = y / max(1, height - 1)
            color = tuple(max(0, min(255, int(channel * (0.92 + 0.18 * scale)))) for channel in base)
            draw.line([(0, y), (width, y)], fill=color)
    else:
        panel_margin_x = max(16, width // 14)
        panel_margin_y = max(12, height // 14)
        panel_box = [panel_margin_x, panel_margin_y, width - panel_margin_x, height - panel_margin_y]
        radius = max(10, height // (5 if bg_mode == "blue_pill" else 12))
        if bg_mode == "blue_pill":
            panel_box[1] = max(10, int(height * 0.12))
            panel_box[3] = min(height - 10, int(height * 0.86))
        draw.rounded_rectangle(panel_box, radius=radius, fill=base)
        if rng.random() < 0.5:
            draw.rounded_rectangle(
                [panel_box[0] + 2, panel_box[1] + 2, panel_box[2] - 2, panel_box[3] - 2],
                radius=max(4, radius - 3),
                outline=(40, 195, 255),
                width=1,
            )

    if rng.random() < 0.7:
        for _ in range(rng.randint(12, 32)):
            x = rng.randrange(width)
            y = rng.randrange(height)
            shade = rng.randint(110, 210)
            draw.point((x, y), fill=(shade, shade, shade))

    max_width_ratio = rng.uniform(0.72, 0.92)
    top = int(height * rng.uniform(0.08, 0.18))
    available = int(height * rng.uniform(0.72, 0.84))
    line_gap = max(3, height // rng.randint(44, 58))
    max_size = max(13, min(height // 11, available // max(1, len(lines)) - line_gap))
    y = top
    for index, line in enumerate(lines):
        is_title = index == 0
        size = max(10, max_size + (2 if is_title else -1))
        bbox = add_centered(
            draw,
            elements,
            line,
            y,
            canvas_size,
            max_size=size,
            min_size=8,
            bold=is_title and rng.random() < 0.85,
            color=WHITE,
            selected=True,
            color_class="blue_white",
            max_width_ratio=max_width_ratio,
        )
        y = bbox[3] + line_gap

    return img, elements, bg_mode, "\n".join(lines)


def gen_hard_icon_negative_list(canvas_size: tuple[int, int], lang: str, rng: random.Random) -> tuple[Image.Image, list[Element], str, str]:
    data = LANG[lang]
    img, bg = make_background(canvas_size, rng)
    draw = ImageDraw.Draw(img)
    width, height = canvas_size
    elements: list[Element] = []
    title = '"More Cycles" Edit' if lang == "en" else data["settings"][5]
    pool = data["cycles"][:]
    if lang == "en" and rng.random() < 0.35:
        pool.extend(["<CYCLE_D_AIR_DR", "<CYCLE_W_SMALL", "<CYCLE_RINSE_SPIN"])
    start = rng.randint(0, max(0, len(pool) - 3))
    rows = pool[start:start + 3]
    lines = [title] + rows
    add_centered(draw, elements, title, int(height * 0.06), canvas_size, max_size=max(16, height // 13), min_size=9)
    row_gap = max(36, height // 5)
    y = int(height * 0.28)
    text_x = max(52, width // 6)
    for row in rows:
        draw_check_icon(draw, max(25, width // 10), y + 8, rng)
        if rng.random() < 0.25:
            draw.line([(text_x - 16, y + 12), (text_x - 4, y + 12)], fill=(95, 95, 95), width=1)
        fnt, size = fit_font(draw, row, int(width * 0.74), max(18, height // 12), 9, False)
        add_text(draw, elements, row, text_x, y, fnt, size, canvas_size, color=WHITE, selected=False)
        y += row_gap
    return img, elements, bg, "\n".join(lines)


PATTERNS: dict[str, Callable[[tuple[int, int], str, random.Random], tuple[Image.Image, list[Element], str, str]]] = {
    "cycle_status": gen_cycle_status,
    "cycle_options": gen_cycle_options,
    "picker": gen_picker,
    "settings_list": gen_settings_list,
    "edit_list": gen_edit_list,
    "message": gen_message,
    "toast": gen_toast,
    "hard_schedule": gen_hard_schedule,
    "hard_duration": gen_hard_duration,
    "hard_progress_timer": gen_hard_progress_timer,
    "hard_numeric": gen_hard_numeric_options,
    "hard_dense": gen_hard_dense_description,
    "hard_blue_microtext": gen_hard_blue_microtext,
    "hard_icon_negative": gen_hard_icon_negative_list,
}

DEFAULT_PATTERN_WEIGHTS = {
    "cycle_status": 22,
    "cycle_options": 18,
    "picker": 20,
    "settings_list": 14,
    "edit_list": 8,
    "message": 15,
    "toast": 3,
    "hard_schedule": 8,
    "hard_duration": 10,
    "hard_progress_timer": 14,
    "hard_numeric": 8,
    "hard_dense": 6,
    "hard_blue_microtext": 12,
    "hard_icon_negative": 10,
}


def weighted_choice(items: list[str], rng: random.Random) -> str:
    weights = [DEFAULT_PATTERN_WEIGHTS.get(item, 1) for item in items]
    return rng.choices(items, weights=weights, k=1)[0]


def build_balanced_schedule(items: list[str], total_count: int, rng: random.Random) -> list[str]:
    unique = list(dict.fromkeys(items))
    if not unique:
        raise ValueError("At least one item is required.")
    base, remainder = divmod(total_count, len(unique))
    order = unique[:]
    rng.shuffle(order)
    schedule: list[str] = []
    for idx, item in enumerate(order):
        schedule.extend([item] * (base + (1 if idx < remainder else 0)))
    rng.shuffle(schedule)
    return schedule


def clean_record(label: Label) -> dict:
    record = asdict(label)
    for element in record["elements"]:
        for key in list(element.keys()):
            if element[key] is None:
                del element[key]
    return record


def generate_one(index: int, canvas_size: tuple[int, int], lang: str, pattern: str,
                 seed: int) -> tuple[Image.Image, Label]:
    rng = random.Random(seed + index * 1009)
    img, elements, bg, raw = PATTERNS[pattern](canvas_size, lang, rng)
    label = Label(
        image_path="",
        pattern=pattern,
        language=lang,
        background=bg,
        canvas_size=list(canvas_size),
        elements=elements,
        raw_text=raw,
    )
    return img, label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--chunk-count", type=int)
    parser.add_argument("--append-labels", action="store_true")
    parser.add_argument("--canvas", choices=["small", "large"], default="small")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--languages", nargs="+", default=["en", "fr", "es"], choices=sorted(LANG.keys()))
    parser.add_argument("--patterns", nargs="+", default=list(PATTERNS.keys()), choices=sorted(PATTERNS.keys()))
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive.")
    if args.start_index < 0 or args.start_index >= args.count:
        raise ValueError("--start-index must be between 0 and count - 1.")

    canvas_size = CANVAS_LARGE if args.canvas == "large" else CANVAS_SMALL
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.jsonl"
    if args.start_index > 0 and labels_path.exists() and labels_path.stat().st_size > 0 and not args.append_labels:
        raise ValueError("Use --append-labels when resuming into an existing labels.jsonl.")

    schedule_rng = random.Random(args.seed)
    language_schedule = build_balanced_schedule(args.languages, args.count, schedule_rng)
    end_index = args.count if args.chunk_count is None else min(args.count, args.start_index + args.chunk_count)
    file_mode = "a" if args.append_labels else "w"
    total_here = end_index - args.start_index
    progress_interval = max(20, total_here // 100)

    with labels_path.open(file_mode, encoding="utf-8", newline="\n") as labels_file:
        for ordinal, index in enumerate(range(args.start_index, end_index), start=1):
            lang = language_schedule[index]
            pattern_rng = random.Random(args.seed + index * 3571)
            pattern = weighted_choice(args.patterns, pattern_rng)
            img, label = generate_one(index, canvas_size, lang, pattern, args.seed)
            filename = f"realui_{index:07d}_{pattern}_{lang}.png"
            img.save(images_dir / filename)
            label.image_path = f"images/{filename}"
            labels_file.write(json.dumps(clean_record(label), ensure_ascii=False) + "\n")
            if ordinal % progress_interval == 0 or index + 1 == end_index:
                print(f"  generated {index + 1}/{args.count}", flush=True)

    print(f"\nDone. Generated indices {args.start_index} to {end_index - 1} in {images_dir}")
    print(f"Labels in {labels_path}")


if __name__ == "__main__":
    main()