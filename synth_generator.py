"""
LG Washer UI Synthetic Image Generator
=========================================

Generates synthetic UI screenshots with ground-truth labels for OCR training.

Patterns (modeled after real LG washer UI screenshots):
  - carousel       : Header + 3-item list with middle one selected
  - message        : Multi-line informational message (with optional warning icon)
  - timer          : Big "1 hr 30 min" style timer with status line
  - list_check     : List items with blue check circles (like cycle selection)
  - toast          : Single-line status message in toast bar
  - title_subtitle : Truncated title + section break + subtitle + button hint

Backgrounds:
  - solid_black, vignette, gradient_horiz, gradient_vert, toast_bottom

Languages (extensible):
  - en, bg (Bulgarian), ru (Russian), el (Greek)

Usage:
    python synth_generator.py --output-dir ./synthetic --count 100
    python synth_generator.py --output-dir ./synthetic --count 1000 --canvas large

Outputs:
    <output-dir>/images/screen_0001.png ...
    <output-dir>/labels.jsonl
"""

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Tuple, List, Optional

from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    arabic_reshaper = None
    get_display = None

# ---------- Configuration ----------

CANVAS_SMALL = (320, 240)
CANVAS_LARGE = (1280, 480)

WHITE = (255, 255, 255)
DIM_GRAY = (140, 140, 140)
BRIGHT_GRAY = (200, 200, 200)
BLUE_CHECK = (0, 122, 255)
WARN_YELLOW = (255, 200, 0)
ORANGE = (255, 140, 0)
BLUE_TEXT = (0, 174, 239)

FONT_CANDIDATES = {
    "regular": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ],
    "bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/tahomabd.ttf",
    ],
    "condensed_bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/tahomabd.ttf",
    ],
}

LANG_FONT_CANDIDATES = {
    "zh_cn": {
        "regular": ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc"],
        "bold": ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simsunb.ttf"],
        "condensed_bold": ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simsunb.ttf"],
    },
    "zh_tw": {
        "regular": ["C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/mingliub.ttc"],
        "bold": ["C:/Windows/Fonts/msjhbd.ttc", "C:/Windows/Fonts/mingliub.ttc"],
        "condensed_bold": ["C:/Windows/Fonts/msjhbd.ttc", "C:/Windows/Fonts/mingliub.ttc"],
    },
    "th": {
        "regular": ["C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/tahomabd.ttf"],
        "bold": ["C:/Windows/Fonts/tahomabd.ttf", "C:/Windows/Fonts/tahoma.ttf"],
        "condensed_bold": ["C:/Windows/Fonts/tahomabd.ttf", "C:/Windows/Fonts/tahoma.ttf"],
    },
    "vi": {
        "regular": ["C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"],
        "bold": ["C:/Windows/Fonts/tahomabd.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"],
        "condensed_bold": ["C:/Windows/Fonts/tahomabd.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"],
    },
    "ar": {
        "regular": ["C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/segoeui.ttf"],
        "bold": ["C:/Windows/Fonts/tahomabd.ttf", "C:/Windows/Fonts/segoeuib.ttf"],
        "condensed_bold": ["C:/Windows/Fonts/tahomabd.ttf", "C:/Windows/Fonts/segoeuib.ttf"],
    },
    "ar_eg": {
        "regular": ["C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/segoeui.ttf"],
        "bold": ["C:/Windows/Fonts/tahomabd.ttf", "C:/Windows/Fonts/segoeuib.ttf"],
        "condensed_bold": ["C:/Windows/Fonts/tahomabd.ttf", "C:/Windows/Fonts/segoeuib.ttf"],
    },
}

RTL_LANGS = {"ar", "ar_eg"}

# ---------- Vocabulary ----------

VOCAB = {
    "en": {
        "soil_levels": ["Med. Heavy", "Medium", "Med. Light", "Heavy", "Light", "Extra"],
        "cycles": [
            "Pet Care", "Microplastics Care", "Rinse & Spin", "Allergiene",
            "Sanitize", "Quick Wash", "Bedding", "Heavy Duty", "Normal", "Delicate",
            "Cold Wash", "Tub Clean", "Steam Refresh",
        ],
        "headers": ["Soil", "Spin", "Temp", "Rinse", "Wash Time"],
        "title_long": [
            "ezDispense\u2122 Nozzle", "Default Det. Dispense", '"More Cycles" Edit',
            "Allergiene\u2122", "Additional Settings", "Network Settings",
        ],
        "subtitles": ["Additional Settings", "Network Settings", "Display Options"],
        "button_hints": ["Press OK to enter.", "Press OK to continue."],
        "messages": [
            ("Wi-Fi is not connected.", "Press OK to connect", "to Wi-Fi."),
            ("Drainage", "Check for a blocked", "drain filter or", "bent drain hose."),
            ("Door is open.", "Close the door", "to start the cycle."),
            ("Cycle paused.", "Press \u25b6\u2016 to resume."),
            ("Update available.", "Press OK to install", "the latest firmware."),
        ],
        "alerts": [
            ("Drainage", "Check for a blocked", "drain filter or", "bent drain hose."),
            ("Overload", "Reduce the load", "and try again."),
        ],
        "toasts": [
            "Delay Start is canceled.", "Cycle complete.", "Settings saved.",
            "Wi-Fi connected.", "Update complete.", "Connection lost.",
        ],
        "status_lines": ["Washing", "Rinsing", "Spinning", "Soaking", "Heating"],
        "units": {"hr": "hr", "min": "min"},
    },
    "fr": {
        "soil_levels": ["Très sale", "Moyen", "Léger", "Intense"],
        "cycles": [
            "Soin animaux", "Microplastiques", "Rinçage essorage",
            "Allergie", "Lavage rapide", "Normal",
        ],
        "headers": ["Salissure", "Essorage", "Temp.", "Rinçage"],
        "title_long": ["Lavage AI", "Paramètres avancés"],
        "subtitles": ["Paramètres avancés", "Paramètres réseau", "Options d'affichage"],
        "button_hints": ["Appuyez sur OK.", "Appuyez sur OK pour continuer."],
        "messages": [
            ("Wi-Fi non connecté.", "Appuyez sur OK pour", "connecter le Wi-Fi."),
            ("Porte ouverte.", "Fermez la porte", "pour démarrer."),
        ],
        "toasts": ["Départ différé annulé.", "Cycle terminé.", "Paramètres enregistrés."],
        "status_lines": ["Lavage", "Rinçage", "Essorage"],
        "units": {"hr": "h", "min": "min"},
    },
    "de": {
        "soil_levels": ["Stark", "Mittel", "Leicht"],
        "cycles": [
            "Haustierpflege", "Mikroplastik", "Spülen & Schleudern",
            "Allergie", "Schnellwäsche", "Normal",
        ],
        "headers": ["Verschm.", "Schleud.", "Temp.", "Spülen"],
        "title_long": ["AI-Wäsche", "Zusätzliche Einstellungen"],
        "subtitles": ["Zusätzliche Einstellungen", "Netzwerkeinstellungen", "Anzeigeoptionen"],
        "button_hints": ["OK drücken.", "OK zum Fortfahren drücken."],
        "messages": [
            ("WLAN nicht verbunden.", "OK drücken zum", "Verbinden mit WLAN."),
            ("Tür ist offen.", "Tür schließen", "zum Starten."),
        ],
        "toasts": ["Zeitvorwahl abgebrochen.", "Programm beendet.", "Einstellungen gespeichert."],
        "status_lines": ["Waschen", "Spülen", "Schleudern"],
        "units": {"hr": "Std", "min": "Min"},
    },
    "nl": {
        "soil_levels": ["Zwaar", "Normaal", "Licht"],
        "cycles": [
            "Dierenzorg", "Microplastics", "Spoelen & centrif.",
            "Allergie", "Snel wassen", "Normaal",
        ],
        "headers": ["Vuil", "Centr.", "Temp.", "Spoelen"],
        "title_long": ["AI-was", "Extra instellingen"],
        "subtitles": ["Extra instellingen", "Netwerkinstellingen", "Schermopties"],
        "button_hints": ["Druk op OK.", "Druk op OK om door te gaan."],
        "messages": [
            ("Wi-Fi is niet verbonden.", "Druk op OK om", "Wi-Fi te verbinden."),
            ("Deur is open.", "Sluit de deur", "om te starten."),
        ],
        "toasts": ["Startuitstel geannuleerd.", "Cyclus voltooid.", "Instellingen opgeslagen."],
        "status_lines": ["Wassen", "Spoelen", "Centrifugeren"],
        "units": {"hr": "u", "min": "min"},
    },
    "zh_cn": {
        "soil_levels": ["重污", "标准", "轻柔"],
        "cycles": ["宠物护理", "微塑料护理", "漂洗和脱水", "除敏", "快速洗", "标准洗"],
        "headers": ["污渍", "脱水", "温度", "漂洗"],
        "title_long": ["AI 洗衣", "附加设置"],
        "subtitles": ["附加设置", "网络设置", "显示选项"],
        "button_hints": ["按 OK 进入。", "按 OK 继续。"],
        "messages": [
            ("Wi-Fi 未连接。", "按 OK 连接", "到 Wi-Fi。"),
            ("门未关闭。", "请关闭机门", "以启动程序。"),
        ],
        "toasts": ["预约已取消。", "程序结束。", "设置已保存。"],
        "status_lines": ["洗涤", "漂洗", "脱水"],
        "units": {"hr": "小时", "min": "分钟"},
    },
    "zh_tw": {
        "soil_levels": ["重污", "標準", "輕柔"],
        "cycles": ["寵物護理", "微塑膠護理", "洗清與脫水", "抗敏", "快速洗", "標準洗"],
        "headers": ["汙漬", "脫水", "溫度", "洗清"],
        "title_long": ["AI 洗衣", "附加設定"],
        "subtitles": ["附加設定", "網路設定", "顯示選項"],
        "button_hints": ["按 OK 進入。", "按 OK 繼續。"],
        "messages": [
            ("Wi-Fi 未連線。", "按 OK 連線", "到 Wi-Fi。"),
            ("機門未關閉。", "請關閉機門", "以開始程序。"),
        ],
        "toasts": ["預約已取消。", "程序完成。", "設定已儲存。"],
        "status_lines": ["洗衣", "洗清", "脫水"],
        "units": {"hr": "小時", "min": "分鐘"},
    },
    "ar": {
        "soil_levels": ["ثقيل", "متوسط", "خفيف"],
        "cycles": ["عناية بالحيوانات", "العناية بالميكروبلاستيك", "شطف وعصر", "حساسية", "غسيل سريع", "عادي"],
        "headers": ["اتساخ", "عصر", "حرارة", "شطف"],
        "title_long": ["غسل ذكي", "إعدادات إضافية"],
        "subtitles": ["إعدادات إضافية", "إعدادات الشبكة", "خيارات العرض"],
        "button_hints": ["اضغط OK للدخول.", "اضغط OK للمتابعة."],
        "messages": [
            ("Wi-Fi غير متصل.", "اضغط OK", "للاتصال بالشبكة."),
            ("الباب مفتوح.", "أغلق الباب", "لبدء الدورة."),
        ],
        "toasts": ["تم إلغاء تأخير البدء.", "اكتملت الدورة.", "تم حفظ الإعدادات."],
        "status_lines": ["غسيل", "شطف", "عصر"],
        "units": {"hr": "سا", "min": "د"},
    },
    "th": {
        "soil_levels": ["หนัก", "ปานกลาง", "เบา"],
        "cycles": ["ดูแลสัตว์เลี้ยง", "ไมโครพลาสติก", "ล้างและปั่น", "ป้องกันภูมิแพ้", "ซักด่วน", "ปกติ"],
        "headers": ["คราบ", "ปั่น", "อุณหภูมิ", "ล้าง"],
        "title_long": ["AI ซักผ้า", "การตั้งค่าเพิ่มเติม"],
        "subtitles": ["การตั้งค่าเพิ่มเติม", "การตั้งค่าเครือข่าย", "ตัวเลือกหน้าจอ"],
        "button_hints": ["กด OK เพื่อเข้า", "กด OK เพื่อดำเนินการต่อ"],
        "messages": [
            ("ยังไม่ได้เชื่อมต่อ Wi-Fi", "กด OK เพื่อ", "เชื่อมต่อ Wi-Fi"),
            ("ประตูเปิดอยู่", "ปิดประตู", "เพื่อเริ่มการซัก"),
        ],
        "toasts": ["ยกเลิกตั้งเวลาแล้ว", "ซักเสร็จแล้ว", "บันทึกการตั้งค่าแล้ว"],
        "status_lines": ["ซัก", "ล้าง", "ปั่น"],
        "units": {"hr": "ชม.", "min": "นาที"},
    },
    "it": {
        "soil_levels": ["Pesante", "Medio", "Leggero"],
        "cycles": ["Cura animali", "Microplastiche", "Risciacquo e centrif.", "Allergie", "Lavaggio rapido", "Normale"],
        "headers": ["Sporco", "Centrif.", "Temp.", "Risciacq."],
        "title_long": ["AI Wash", "Impostazioni aggiuntive"],
        "subtitles": ["Impostazioni aggiuntive", "Impostazioni di rete", "Opzioni display"],
        "button_hints": ["Premi OK.", "Premi OK per continuare."],
        "messages": [
            ("Wi-Fi non connesso.", "Premi OK per", "connettere il Wi-Fi."),
            ("Porta aperta.", "Chiudi la porta", "per avviare il ciclo."),
        ],
        "toasts": ["Avvio ritardato annullato.", "Ciclo completato.", "Impostazioni salvate."],
        "status_lines": ["Lavaggio", "Risciacquo", "Centrifuga"],
        "units": {"hr": "h", "min": "min"},
    },
    "es": {
        "soil_levels": ["Intenso", "Medio", "Ligero"],
        "cycles": ["Cuidado mascotas", "Microplásticos", "Enjuague y centrif.", "Alergias", "Lavado rápido", "Normal"],
        "headers": ["Suciedad", "Centrif.", "Temp.", "Enjuague"],
        "title_long": ["Lavado AI", "Ajustes adicionales"],
        "subtitles": ["Ajustes adicionales", "Ajustes de red", "Opciones de pantalla"],
        "button_hints": ["Pulse OK.", "Pulse OK para continuar."],
        "messages": [
            ("Wi-Fi no está conectado.", "Pulse OK para", "conectar el Wi-Fi."),
            ("La puerta está abierta.", "Cierre la puerta", "para iniciar el ciclo."),
        ],
        "toasts": ["Inicio diferido cancelado.", "Ciclo completado.", "Ajustes guardados."],
        "status_lines": ["Lavado", "Enjuague", "Centrifugado"],
        "units": {"hr": "h", "min": "min"},
    },
    "pt": {
        "soil_levels": ["Pesado", "Médio", "Leve"],
        "cycles": ["Cuidado pet", "Microplásticos", "Enxágue e centrif.", "Alergias", "Lavagem rápida", "Normal"],
        "headers": ["Sujeira", "Centrif.", "Temp.", "Enxágue"],
        "title_long": ["Lavagem AI", "Configurações extras"],
        "subtitles": ["Configurações extras", "Configurações de rede", "Opções de tela"],
        "button_hints": ["Pressione OK.", "Pressione OK para continuar."],
        "messages": [
            ("Wi-Fi não conectado.", "Pressione OK para", "conectar ao Wi-Fi."),
            ("Porta aberta.", "Feche a porta", "para iniciar o ciclo."),
        ],
        "toasts": ["Início diferido cancelado.", "Ciclo concluído.", "Configurações salvas."],
        "status_lines": ["Lavagem", "Enxágue", "Centrifugação"],
        "units": {"hr": "h", "min": "min"},
    },
    "vi": {
        "soil_levels": ["Nặng", "Trung bình", "Nhẹ"],
        "cycles": ["Chăm sóc thú cưng", "Vi nhựa", "Xả và vắt", "Dị ứng", "Giặt nhanh", "Thông thường"],
        "headers": ["Bẩn", "Vắt", "Nhiệt độ", "Xả"],
        "title_long": ["Giặt AI", "Cài đặt bổ sung"],
        "subtitles": ["Cài đặt bổ sung", "Cài đặt mạng", "Tùy chọn hiển thị"],
        "button_hints": ["Nhấn OK.", "Nhấn OK để tiếp tục."],
        "messages": [
            ("Wi-Fi chưa kết nối.", "Nhấn OK để", "kết nối Wi-Fi."),
            ("Cửa đang mở.", "Đóng cửa lại", "để bắt đầu chu trình."),
        ],
        "toasts": ["Đã hủy hẹn giờ.", "Chu trình đã hoàn tất.", "Đã lưu cài đặt."],
        "status_lines": ["Giặt", "Xả", "Vắt"],
        "units": {"hr": "giờ", "min": "phút"},
    },
    "no": {
        "soil_levels": ["Tung", "Middels", "Lett"],
        "cycles": ["Kjæledyrpleie", "Mikroplast", "Skyll og sentrif.", "Allergi", "Hurtigvask", "Normal"],
        "headers": ["Smuss", "Sentrif.", "Temp.", "Skyll"],
        "title_long": ["AI-vask", "Ekstra innstillinger"],
        "subtitles": ["Ekstra innstillinger", "Nettverksinnstillinger", "Skjermvalg"],
        "button_hints": ["Trykk OK.", "Trykk OK for å fortsette."],
        "messages": [
            ("Wi-Fi er ikke tilkoblet.", "Trykk OK for å", "koble til Wi-Fi."),
            ("Døren er åpen.", "Lukk døren", "for å starte."),
        ],
        "toasts": ["Utsatt start avbrutt.", "Program fullført.", "Innstillinger lagret."],
        "status_lines": ["Vask", "Skylling", "Sentrifugering"],
        "units": {"hr": "t", "min": "min"},
    },
    "pl": {
        "soil_levels": ["Mocne", "Średnie", "Lekkie"],
        "cycles": ["Opieka nad zwierz.", "Mikroplastik", "Płukanie i wirowanie", "Alergie", "Szybkie pranie", "Normalny"],
        "headers": ["Brud", "Wirow.", "Temp.", "Płukanie"],
        "title_long": ["Pranie AI", "Ustawienia dodatkowe"],
        "subtitles": ["Ustawienia dodatkowe", "Ustawienia sieci", "Opcje ekranu"],
        "button_hints": ["Naciśnij OK.", "Naciśnij OK, aby kontynuować."],
        "messages": [
            ("Wi-Fi nie jest połączone.", "Naciśnij OK, aby", "połączyć z Wi-Fi."),
            ("Drzwi są otwarte.", "Zamknij drzwi", "aby rozpocząć."),
        ],
        "toasts": ["Opóźniony start anulowany.", "Cykl zakończony.", "Ustawienia zapisane."],
        "status_lines": ["Pranie", "Płukanie", "Wirowanie"],
        "units": {"hr": "g", "min": "min"},
    },
    "el": {
        "soil_levels": ["Βαρύ", "Μεσαίο", "Ελαφρύ"],
        "cycles": [
            "Φροντίδα κατοικιδίων", "Μικροπλαστικά", "Ξέβγαλμα & Στύψιμο",
            "Γρήγορο πλύσιμο", "Κανονικό", "Ευαίσθητα",
        ],
        "headers": ["Λέρωμα", "Στύψιμο", "Θερμ.", "Ξέβγαλμα"],
        "title_long": ["AI πλύσιμο", "Πρόσθετες ρυθμίσεις"],
        "subtitles": ["Πρόσθετες ρυθμίσεις", "Ρυθμίσεις δικτύου", "Επιλογές οθόνης"],
        "button_hints": ["Πατήστε OK.", "Πατήστε OK για συνέχεια."],
        "messages": [
            ("Το Wi-Fi δεν είναι συνδεδεμένο.", "Πατήστε OK για", "σύνδεση Wi-Fi."),
            ("Η πόρτα είναι ανοιχτή.", "Κλείστε την πόρτα", "για να ξεκινήσει."),
        ],
        "toasts": ["Καθυστέρηση εκκίνησης ακυρώθηκε.", "Ο κύκλος ολοκληρώθηκε.", "Οι ρυθμίσεις αποθηκεύτηκαν."],
        "status_lines": ["Πλύσιμο", "Ξέβγαλμα", "Στύψιμο"],
        "units": {"hr": "ωρ.", "min": "λεπ."},
    },
    "ar_eg": {
        "soil_levels": ["تقيل", "متوسط", "خفيف"],
        "cycles": ["عناية بالحيوانات", "ميكروبلاستيك", "شطف وعصر", "حساسية", "غسيل سريع", "عادي"],
        "headers": ["اتساخ", "عصر", "حرارة", "شطف"],
        "title_long": ["غسيل AI", "إعدادات إضافية"],
        "subtitles": ["إعدادات إضافية", "إعدادات الشبكة", "خيارات الشاشة"],
        "button_hints": ["اضغط OK للدخول.", "اضغط OK للتكملة."],
        "messages": [
            ("الواي فاي غير متصل.", "اضغط OK", "عشان تتصل بالشبكة."),
            ("الباب مفتوح.", "اقفل الباب", "علشان تبدأ الدورة."),
        ],
        "toasts": ["تم إلغاء تأخير البدء.", "الدورة خلصت.", "تم حفظ الإعدادات."],
        "status_lines": ["غسيل", "شطف", "عصر"],
        "units": {"hr": "سا", "min": "د"},
    },
    "bg": {
        "soil_levels": ["Силно", "Средно", "Леко", "Допълнително"],
        "cycles": [
            "Грижа за домашни любимци", "Микропластмаси", "Изплакване и центрофуга",
            "Алергиене", "Бързо пране", "Обикновено пране",
        ],
        "headers": ["Замърсяване", "Центрофуга", "Темп.", "Изплакване"],
        "title_long": ["AI пране", "Допълнителни настройки"],
        "subtitles": ["Допълнителни настройки", "Мрежови настройки", "Опции на дисплея"],
        "button_hints": ["Натиснете OK.", "Натиснете OK за продължаване."],
        "messages": [
            ("Wi-Fi не е свързан.", "Натиснете OK за", "свързване с Wi-Fi."),
            ("Вратата е отворена.", "Затворете вратата", "за да започне."),
        ],
        "toasts": ["Отложеният старт е отменен.", "Цикълът завърши.", "Настройките са запазени."],
        "status_lines": ["Пране", "Изплакване", "Центрофуга"],
        "units": {"hr": "ч", "min": "мин"},
    },
    "cs": {
        "soil_levels": ["Silné", "Střední", "Lehké"],
        "cycles": ["Péče o mazlíčky", "Mikroplasty", "Máchání a odstř.", "Alergie", "Rychlé praní", "Normální"],
        "headers": ["Špína", "Odstř.", "Tepl.", "Máchání"],
        "title_long": ["AI praní", "Další nastavení"],
        "subtitles": ["Další nastavení", "Nastavení sítě", "Možnosti displeje"],
        "button_hints": ["Stiskněte OK.", "Stiskněte OK pro pokračování."],
        "messages": [
            ("Wi-Fi není připojeno.", "Stiskněte OK pro", "připojení k Wi-Fi."),
            ("Dveře jsou otevřené.", "Zavřete dveře", "pro spuštění cyklu."),
        ],
        "toasts": ["Odložený start zrušen.", "Cyklus dokončen.", "Nastavení uloženo."],
        "status_lines": ["Praní", "Máchání", "Odstřeďování"],
        "units": {"hr": "h", "min": "min"},
    },
    "uk": {
        "soil_levels": ["Сильне", "Середнє", "Легке"],
        "cycles": ["Догляд за тваринами", "Мікропластик", "Полоскання і віджим", "Алергія", "Швидке прання", "Нормальний"],
        "headers": ["Бруд", "Віджим", "Темп.", "Полоск."],
        "title_long": ["AI прання", "Додаткові налаштування"],
        "subtitles": ["Додаткові налаштування", "Налаштування мережі", "Параметри дисплея"],
        "button_hints": ["Натисніть OK.", "Натисніть OK для продовження."],
        "messages": [
            ("Wi-Fi не підключено.", "Натисніть OK для", "підключення до Wi-Fi."),
            ("Дверцята відчинені.", "Зачиніть дверцята", "щоб почати."),
        ],
        "toasts": ["Відкладений старт скасовано.", "Цикл завершено.", "Налаштування збережено."],
        "status_lines": ["Прання", "Полоскання", "Віджим"],
        "units": {"hr": "год", "min": "хв"},
    },
    "ru": {
        "soil_levels": ["Сильно", "Средне", "Слегка"],
        "cycles": [
            "Уход за животными", "Микропластик", "Полоскание и отжим",
            "Быстрая стирка", "Стандарт", "Деликатная",
        ],
        "headers": ["Загрязнение", "Отжим", "Темп.", "Полоск."],
        "title_long": ["AI стирка", "Дополнительные настройки"],
        "subtitles": ["Дополнительные настройки", "Настройки сети", "Параметры дисплея"],
        "button_hints": ["Нажмите OK.", "Нажмите OK для продолжения."],
        "messages": [
            ("Wi-Fi не подключен.", "Нажмите OK для", "подключения к Wi-Fi."),
            ("Дверца открыта.", "Закройте дверцу", "чтобы начать."),
        ],
        "toasts": ["Отложенный старт отменён.", "Цикл завершён.", "Настройки сохранены."],
        "status_lines": ["Стирка", "Полоскание", "Отжим"],
        "units": {"hr": "ч", "min": "мин"},
    },
    "lt": {
        "soil_levels": ["Stiprus", "Vidutinis", "Lengvas"],
        "cycles": ["Gyvūnų priežiūra", "Mikroplastikas", "Skalavimas ir gręž.", "Alergija", "Greitas skalbimas", "Normalus"],
        "headers": ["Purvas", "Gręž.", "Temp.", "Skalav."],
        "title_long": ["AI skalbimas", "Papildomi nustatymai"],
        "subtitles": ["Papildomi nustatymai", "Tinklo nustatymai", "Ekrano parinktys"],
        "button_hints": ["Paspauskite OK.", "Paspauskite OK tęsti."],
        "messages": [
            ("Wi-Fi neprijungtas.", "Paspauskite OK, kad", "prisijungtumėte prie Wi-Fi."),
            ("Durys atidarytos.", "Uždarykite duris", "kad pradėtumėte."),
        ],
        "toasts": ["Atidėtas paleidimas atšauktas.", "Ciklas baigtas.", "Nustatymai išsaugoti."],
        "status_lines": ["Skalbimas", "Skalavimas", "Gręžimas"],
        "units": {"hr": "val.", "min": "min."},
    },
    "lv": {
        "soil_levels": ["Smags", "Vidējs", "Viegls"],
        "cycles": ["Mājdzīvnieku kopšana", "Mikroplastmasa", "Skalošana un izgr.", "Alerģija", "Ātrā mazgāšana", "Normāls"],
        "headers": ["Netīr.", "Izgr.", "Temp.", "Skaloš."],
        "title_long": ["AI mazgāšana", "Papildu iestatījumi"],
        "subtitles": ["Papildu iestatījumi", "Tīkla iestatījumi", "Displeja opcijas"],
        "button_hints": ["Nospiediet OK.", "Nospiediet OK, lai turpinātu."],
        "messages": [
            ("Wi-Fi nav savienots.", "Nospiediet OK, lai", "savienotu Wi-Fi."),
            ("Durvis ir atvērtas.", "Aizveriet durvis", "lai sāktu ciklu."),
        ],
        "toasts": ["Atliktais starts atcelts.", "Cikls pabeigts.", "Iestatījumi saglabāti."],
        "status_lines": ["Mazgāšana", "Skalošana", "Izgriešana"],
        "units": {"hr": "st.", "min": "min."},
    },
}


# ---------- Data classes ----------

@dataclass
class Element:
    type: str  # "text" | "icon"
    bbox: List[int]
    text: Optional[str] = None
    icon_name: Optional[str] = None
    selected: Optional[bool] = None
    truncated: Optional[bool] = None
    color_class: Optional[str] = None
    size_class: Optional[str] = None


@dataclass
class Label:
    image_path: str
    pattern: str
    language: str
    background: str
    canvas_size: List[int]
    elements: List[Element]
    raw_text: str  # all visible text concatenated, with [ICON] tokens


# ---------- Drawing helpers ----------

def get_font_candidates(kind: str, lang: Optional[str] = None) -> List[str]:
    candidates = []
    if lang in LANG_FONT_CANDIDATES:
        candidates.extend(LANG_FONT_CANDIDATES[lang].get(kind, []))
    candidates.extend(FONT_CANDIDATES[kind])
    return candidates


def resolve_font_path(kind: str, lang: Optional[str] = None) -> Optional[str]:
    for candidate in get_font_candidates(kind, lang):
        if Path(candidate).exists():
            return candidate
    return None


def normalize_text_for_rendering(text: str, lang: Optional[str] = None) -> str:
    if lang in RTL_LANGS and arabic_reshaper and get_display:
        return get_display(arabic_reshaper.reshape(text))
    return text


def font(size: int, bold: bool = False, condensed: bool = False,
         lang: Optional[str] = None) -> ImageFont.ImageFont:
    kind = "condensed_bold" if condensed else "bold" if bold else "regular"
    path = resolve_font_path(kind, lang)
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(path, size)


def measure(draw: ImageDraw.ImageDraw, text: str, fnt,
            lang: Optional[str] = None) -> Tuple[int, int]:
    rendered = normalize_text_for_rendering(text, lang)
    bbox = draw.textbbox((0, 0), rendered, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_text(draw, xy, text, color, fnt, lang: Optional[str] = None):
    rendered = normalize_text_for_rendering(text, lang)
    draw.text(xy, rendered, fill=color, font=fnt)


def draw_centered(draw, text, y, fnt, color, canvas_w,
                  lang: Optional[str] = None) -> List[int]:
    w, h = measure(draw, text, fnt, lang)
    x = (canvas_w - w) // 2
    draw_text(draw, (x, y), text, color, fnt, lang=lang)
    return [x, y, x + w, y + h]


def size_class_for(px: int) -> str:
    if px < 18: return "small"
    if px < 32: return "medium"
    if px < 60: return "large"
    return "xl"


# ---------- Background generation ----------

def make_background(size: Tuple[int, int], kind: str) -> Image.Image:
    w, h = size
    if kind == "solid_black":
        return Image.new("RGB", size, (0, 0, 0))

    if kind == "solid_gray":
        shade = random.randint(35, 55)
        return Image.new("RGB", size, (shade, shade, shade))

    if kind == "vignette":
        img = Image.new("RGB", size, (0, 0, 0))
        pixels = img.load()
        cx, cy = w / 2, h / 2
        max_d = math.hypot(cx, cy)
        for y in range(h):
            for x in range(w):
                d = math.hypot(x - cx, y - cy) / max_d
                shade = int(45 * (1 - min(d, 1)) ** 1.5)
                pixels[x, y] = (shade, shade, shade)
        return img

    if kind == "gradient_horiz":
        img = Image.new("RGB", size, (0, 0, 0))
        pixels = img.load()
        for x in range(w):
            d = abs(x - w / 2) / (w / 2)
            shade = int(50 * d)
            for y in range(h):
                pixels[x, y] = (shade, shade, shade)
        return img

    if kind == "gradient_vert":
        img = Image.new("RGB", size, (0, 0, 0))
        pixels = img.load()
        for y in range(h):
            d = abs(y - h / 2) / (h / 2)
            shade = int(50 * d)
            for x in range(w):
                pixels[x, y] = (shade, shade, shade)
        return img

    if kind == "toast_bottom":
        img = Image.new("RGB", size, (0, 0, 0))
        toast_h = h // 3
        toast = Image.new("RGB", (w, toast_h), (60, 60, 60))
        img.paste(toast, (0, h - toast_h))
        return img

    return Image.new("RGB", size, (0, 0, 0))


# ---------- Icon drawing ----------

def draw_check_circle(draw, x, y, r, color=BLUE_CHECK) -> List[int]:
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    # Check mark (3-segment polyline)
    pts = [(x - r * 0.5, y), (x - r * 0.1, y + r * 0.4), (x + r * 0.55, y - r * 0.4)]
    draw.line(pts, fill=WHITE, width=max(2, int(r * 0.25)))
    return [x - r, y - r, x + r, y + r]


def draw_warning_triangle(draw, cx, cy, size, color=WARN_YELLOW) -> List[int]:
    half = size // 2
    pts = [(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)]
    draw.polygon(pts, fill=color)
    # Exclamation
    draw.line([(cx, cy - half * 0.4), (cx, cy + half * 0.2)], fill=(0, 0, 0), width=max(2, size // 12))
    draw.ellipse([cx - size // 16, cy + half * 0.4, cx + size // 16, cy + half * 0.55],
                 fill=(0, 0, 0))
    return [cx - half, cy - half, cx + half, cy + half]


def draw_play_pause_inline(draw, x, y, h, color=WHITE) -> List[int]:
    """Draw a small play+pause combined glyph at baseline (x,y) with height h."""
    tri_w = int(h * 0.55)
    bar_w = max(2, int(h * 0.18))
    gap = max(2, int(h * 0.18))
    # Triangle (play)
    draw.polygon([(x, y), (x, y + h), (x + tri_w, y + h // 2)], fill=color)
    # Two bars (pause)
    bar_x = x + tri_w + gap
    draw.rectangle([bar_x, y, bar_x + bar_w, y + h], fill=color)
    draw.rectangle([bar_x + bar_w + gap, y, bar_x + 2 * bar_w + gap, y + h], fill=color)
    total_w = tri_w + gap + 2 * bar_w + gap
    return [x, y, x + total_w, y + h]


def draw_paginator_dots(draw, cx, y, count, active_idx, r=2, gap=8) -> List[List[int]]:
    total_w = count * (2 * r) + (count - 1) * gap
    start_x = cx - total_w // 2
    bboxes = []
    for i in range(count):
        x = start_x + i * (2 * r + gap)
        color = WHITE if i == active_idx else (90, 90, 90)
        draw.ellipse([x, y - r, x + 2 * r, y + r], fill=color)
        bboxes.append([x, y - r, x + 2 * r, y + r])
    return bboxes


def draw_horizontal_divider(draw, y, w, margin=20, color=(80, 80, 80), thickness=1):
    draw.line([(margin, y), (w - margin, y)], fill=color, width=thickness)


# ---------- Pattern generators ----------

def gen_carousel(canvas_size, lang) -> Tuple[Image.Image, List[Element], str, str]:
    """Header + 3-item carousel (middle selected, top/bottom dimmed)."""
    w, h = canvas_size
    bg_kind = random.choice(["solid_black", "solid_black", "vignette"])
    img = make_background(canvas_size, bg_kind)
    draw = ImageDraw.Draw(img)
    elems: List[Element] = []
    raw_lines: List[str] = []

    # Header
    headers = VOCAB[lang].get("headers", ["Soil"])
    header = random.choice(headers)
    h_size = max(14, h // 14)
    h_font = font(h_size, bold=False, lang=lang)
    bbox = draw_centered(draw, header, h // 16, h_font, WHITE, w, lang=lang)
    elems.append(Element("text", bbox, text=header, selected=False,
                         color_class="white", size_class=size_class_for(h_size)))
    raw_lines.append(header)

    # Carousel items
    items_pool = VOCAB[lang].get("soil_levels") or VOCAB[lang].get("cycles", ["A", "B", "C"])
    chosen = random.sample(items_pool, k=min(3, len(items_pool)))
    while len(chosen) < 3:
        chosen.append(random.choice(items_pool))

    item_size = max(20, h // 7)
    item_font_dim = font(int(item_size * 0.85), bold=False, lang=lang)
    item_font_sel = font(item_size, bold=True, lang=lang)

    y_top = h * 0.30
    y_mid = h * 0.50
    y_bot = h * 0.72

    # Top (dimmed)
    bbox = draw_centered(draw, chosen[0], int(y_top), item_font_dim, DIM_GRAY, w, lang=lang)
    elems.append(Element("text", bbox, text=chosen[0], selected=False,
                         color_class="gray", size_class=size_class_for(int(item_size * 0.85))))
    raw_lines.append(chosen[0])

    # Divider above middle
    draw_horizontal_divider(draw, int(y_mid - item_size * 0.6), w)

    # Middle (selected)
    bbox = draw_centered(draw, chosen[1], int(y_mid - item_size * 0.4), item_font_sel, WHITE, w, lang=lang)
    elems.append(Element("text", bbox, text=chosen[1], selected=True,
                         color_class="white", size_class=size_class_for(item_size)))
    raw_lines.append(chosen[1])

    # Divider below middle
    draw_horizontal_divider(draw, int(y_mid + item_size * 0.7), w)

    # Bottom (dimmed)
    bbox = draw_centered(draw, chosen[2], int(y_bot), item_font_dim, DIM_GRAY, w, lang=lang)
    elems.append(Element("text", bbox, text=chosen[2], selected=False,
                         color_class="gray", size_class=size_class_for(int(item_size * 0.85))))
    raw_lines.append(chosen[2])

    return img, elems, bg_kind, "\n".join(raw_lines)


def gen_message(canvas_size, lang) -> Tuple[Image.Image, List[Element], str, str]:
    """Multi-line message, optionally with warning icon at top."""
    w, h = canvas_size
    bg_kind = random.choice(["solid_black", "solid_black", "vignette", "gradient_vert"])
    img = make_background(canvas_size, bg_kind)
    draw = ImageDraw.Draw(img)
    elems: List[Element] = []
    raw_lines: List[str] = []

    msg_pool = VOCAB[lang].get("alerts") or VOCAB[lang].get("messages", [])
    if not msg_pool:
        msg_pool = VOCAB["en"]["messages"]
    msg = random.choice(msg_pool)

    use_warning = random.random() < 0.35
    y = int(h * 0.08)

    if use_warning:
        icon_size = max(20, h // 8)
        cx = w // 2
        cy = y + icon_size // 2
        bbox = draw_warning_triangle(draw, cx, cy, icon_size)
        elems.append(Element("icon", bbox, icon_name="warning", color_class="yellow"))
        raw_lines.append("[WARNING]")
        y += icon_size + 6

    # First line is the title (bold, larger)
    title_size = max(18, h // 11)
    title_font = font(title_size, bold=True, lang=lang)
    rest_size = max(14, h // 14)
    rest_font = font(rest_size, bold=False, lang=lang)

    title = msg[0]
    bbox = draw_centered(draw, title, y, title_font, WHITE, w, lang=lang)
    elems.append(Element("text", bbox, text=title, selected=True,
                         color_class="white", size_class=size_class_for(title_size)))
    raw_lines.append(title)
    y = bbox[3] + 6

    for line in msg[1:]:
        bbox = draw_centered(draw, line, y, rest_font, WHITE, w, lang=lang)
        elems.append(Element("text", bbox, text=line, selected=False,
                             color_class="white", size_class=size_class_for(rest_size)))
        raw_lines.append(line)
        y = bbox[3] + 4

    return img, elems, bg_kind, "\n".join(raw_lines)


def gen_timer(canvas_size, lang) -> Tuple[Image.Image, List[Element], str, str]:
    """Big '1 hr 30 min' timer with status and button hint."""
    w, h = canvas_size
    bg_kind = random.choice(["solid_black", "solid_black", "vignette"])
    img = make_background(canvas_size, bg_kind)
    draw = ImageDraw.Draw(img)
    elems: List[Element] = []
    raw_lines: List[str] = []

    # Cycle name on top
    cycles = VOCAB[lang].get("cycles", ["Normal"])
    cycle = random.choice(cycles)
    cycle_size = max(14, h // 14)
    cycle_font = font(cycle_size, bold=False, lang=lang)
    bbox = draw_centered(draw, cycle, int(h * 0.06), cycle_font, WHITE, w, lang=lang)
    elems.append(Element("text", bbox, text=cycle, selected=False,
                         color_class="white", size_class=size_class_for(cycle_size)))
    raw_lines.append(cycle)

    # Big timer "X hr Y min" or "Today AM HH:MM"
    use_clock = random.random() < 0.4 and lang == "en"
    big_size = max(36, int(h * 0.30))
    small_size = max(14, int(big_size * 0.32))
    big_font = font(big_size, bold=True, lang=lang)
    small_font = font(small_size, bold=False, lang=lang)
    unit_labels = VOCAB[lang].get("units", {"hr": "hr", "min": "min"})

    timer_y = int(h * 0.27)
    if use_clock:
        # "Today AM 12:30"
        small_text_left = "Today\nAM"
        time_text = f"{random.randint(1, 12)}:{random.choice(['00', '15', '30', '45'])}"
        # Stacked small text on the left
        sw, sh = measure(draw, "Today", small_font)
        tw, th = measure(draw, time_text, big_font)
        gap = 8
        total_w = sw + gap + tw
        start_x = (w - total_w) // 2
        # Two lines on left
        draw.text((start_x, timer_y), "Today", fill=WHITE, font=small_font)
        small_y2 = timer_y + sh + 2
        draw.text((start_x, small_y2), "AM", fill=WHITE, font=small_font)
        elems.append(Element("text", [start_x, timer_y, start_x + sw, timer_y + sh],
                             text="Today", color_class="white", size_class=size_class_for(small_size)))
        elems.append(Element("text", [start_x, small_y2, start_x + measure(draw, "AM", small_font)[0], small_y2 + sh],
                             text="AM", color_class="white", size_class=size_class_for(small_size)))
        time_x = start_x + sw + gap
        draw.text((time_x, timer_y), time_text, fill=WHITE, font=big_font)
        elems.append(Element("text", [time_x, timer_y, time_x + tw, timer_y + th],
                             text=time_text, color_class="white", size_class=size_class_for(big_size)))
        raw_lines.append(f"Today AM {time_text}")
    else:
        hours = random.randint(0, 4)
        minutes = random.choice([0, 15, 30, 45])
        if hours == 0:
            minutes = random.choice([15, 30, 45, 60, 90])
        # "1 hr 30 min" - draw with mixed sizes
        if hours > 0:
            num_h, unit_h = str(hours), unit_labels.get("hr", "hr")
        else:
            num_h, unit_h = None, None
        num_m, unit_m = str(minutes), unit_labels.get("min", "min")

        # Compose pieces
        pieces = []
        if num_h:
            pieces.append((num_h, big_font, "white", size_class_for(big_size)))
            pieces.append((unit_h, small_font, "white", size_class_for(small_size)))
        pieces.append((num_m, big_font, "white", size_class_for(big_size)))
        pieces.append((unit_m, small_font, "white", size_class_for(small_size)))

        # Compute total width
        widths = [measure(draw, t, f, lang)[0] for t, f, *_ in pieces]
        spacing = 6
        total_w = sum(widths) + spacing * (len(pieces) - 1)
        x = (w - total_w) // 2
        # Draw each, baseline-aligned to bottom of big text
        big_h = measure(draw, "0", big_font, lang)[1]
        baseline_y = timer_y + big_h
        for (text, fnt, color, sz_class), tw in zip(pieces, widths):
            th_ = measure(draw, text, fnt, lang)[1]
            ty = baseline_y - th_  # align bottom
            draw_text(draw, (x, ty), text, WHITE, fnt, lang=lang)
            elems.append(Element("text", [x, ty, x + tw, ty + th_],
                                 text=text, color_class="white", size_class=sz_class))
            x += tw + spacing
        time_str = f"{hours} {unit_h} {minutes} {unit_m}" if hours > 0 else f"{minutes} {unit_m}"
        raw_lines.append(time_str)

    # Progress bar
    bar_y = int(h * 0.62)
    bar_margin = 30
    draw.line([(bar_margin, bar_y), (w - bar_margin, bar_y)], fill=(80, 80, 80), width=2)
    progress_x = bar_margin + int((w - 2 * bar_margin) * random.uniform(0.05, 0.4))
    draw.line([(bar_margin, bar_y), (progress_x, bar_y)], fill=BLUE_TEXT, width=2)

    # Status line
    statuses = VOCAB[lang].get("status_lines", ["Washing"])
    status = random.choice(statuses)
    status_size = max(14, h // 14)
    status_font = font(status_size, bold=False, lang=lang)
    bbox = draw_centered(draw, status, int(h * 0.70), status_font, WHITE, w, lang=lang)
    elems.append(Element("text", bbox, text=status, selected=False,
                         color_class="white", size_class=size_class_for(status_size)))
    raw_lines.append(status)

    # Bottom button hint with inline play/pause icon (English only for now)
    if lang == "en" and random.random() < 0.7:
        hint_size = max(12, h // 17)
        hint_font = font(hint_size, bold=False)
        prefix, suffix = "Press ", " to add garments."
        pw = measure(draw, prefix, hint_font)[0]
        sw = measure(draw, suffix, hint_font)[0]
        icon_h = hint_size
        icon_w_est = int(icon_h * 1.6)
        total = pw + 4 + icon_w_est + 4 + sw
        x = (w - total) // 2
        y = int(h * 0.86)
        draw.text((x, y), prefix, fill=WHITE, font=hint_font)
        elems.append(Element("text", [x, y, x + pw, y + icon_h],
                             text="Press", color_class="white", size_class=size_class_for(hint_size)))
        x += pw + 4
        ibbox = draw_play_pause_inline(draw, x, y + 2, icon_h - 4)
        elems.append(Element("icon", ibbox, icon_name="play_pause", color_class="white"))
        x = ibbox[2] + 4
        draw.text((x, y), suffix.lstrip(), fill=WHITE, font=hint_font)
        elems.append(Element("text", [x, y, x + sw, y + icon_h],
                             text=suffix.strip(), color_class="white",
                             size_class=size_class_for(hint_size)))
        raw_lines.append(f"Press [PLAY_PAUSE]{suffix}")

    return img, elems, bg_kind, "\n".join(raw_lines)


def gen_list_check(canvas_size, lang) -> Tuple[Image.Image, List[Element], str, str]:
    """List with check circles, top item highlighted."""
    w, h = canvas_size
    bg_kind = "solid_black"
    img = make_background(canvas_size, bg_kind)
    draw = ImageDraw.Draw(img)
    elems: List[Element] = []
    raw_lines: List[str] = []

    # Title
    titles_en = ['"More Cycles" Edit', "Selected Cycles", "Active Options"]
    title = random.choice(titles_en) if lang == "en" else (
        VOCAB[lang].get("title_long", ["Допълнителни"])[0]
    )
    title_size = max(16, h // 12)
    title_font = font(title_size, bold=False, lang=lang)
    bbox = draw_centered(draw, title, int(h * 0.04), title_font, WHITE, w, lang=lang)
    elems.append(Element("text", bbox, text=title, selected=False,
                         color_class="white", size_class=size_class_for(title_size)))
    raw_lines.append(title)

    cycles = VOCAB[lang].get("cycles", ["A", "B", "C"])
    items = random.sample(cycles, k=min(3, len(cycles)))
    while len(items) < 3:
        items.append(random.choice(cycles))

    item_size = max(16, h // 11)
    item_font = font(item_size, bold=False, lang=lang)

    y = int(h * 0.22)
    line_height = int(h * 0.20)
    check_r = max(8, item_size // 2 - 2)

    for i, item in enumerate(items):
        # Selection background for first item
        if i == 0:
            draw.rectangle([(8, y - 4), (w - 8, y + line_height - 12)],
                           fill=(45, 45, 45))

        # Check circle
        cx = 30
        cy = y + line_height // 2 - 8
        ibbox = draw_check_circle(draw, cx, cy, check_r)
        elems.append(Element("icon", ibbox, icon_name="check", color_class="blue"))

        # Item text
        tx = cx + check_r + 12
        ty = y + (line_height - item_size) // 2 - 8
        tw, th_ = measure(draw, item, item_font, lang)
        draw_text(draw, (tx, ty), item, WHITE, item_font, lang=lang)
        elems.append(Element("text", [tx, ty, tx + tw, ty + th_],
                             text=item, selected=(i == 0),
                             color_class="white", size_class=size_class_for(item_size)))
        # Divider
        if i < len(items) - 1:
            draw.line([(20, y + line_height - 8), (w - 20, y + line_height - 8)],
                      fill=(70, 70, 70), width=1)
        raw_lines.append(f"[CHECK] {item}")
        y += line_height

    return img, elems, bg_kind, "\n".join(raw_lines)


def gen_toast(canvas_size, lang) -> Tuple[Image.Image, List[Element], str, str]:
    """Single line toast at bottom on partial gray strip."""
    w, h = canvas_size
    bg_kind = "toast_bottom"
    img = make_background(canvas_size, bg_kind)
    draw = ImageDraw.Draw(img)
    elems: List[Element] = []
    raw_lines: List[str] = []

    toasts = VOCAB[lang].get("toasts", ["OK"])
    msg = random.choice(toasts)
    sz = max(16, h // 12)
    fnt = font(sz, bold=False, lang=lang)
    bbox = draw_centered(draw, msg, int(h * 0.78), fnt, WHITE, w, lang=lang)
    elems.append(Element("text", bbox, text=msg, selected=True,
                         color_class="white", size_class=size_class_for(sz)))
    raw_lines.append(msg)

    return img, elems, bg_kind, "\n".join(raw_lines)


def gen_title_subtitle(canvas_size, lang) -> Tuple[Image.Image, List[Element], str, str]:
    """Truncated title at top + divider + section title + button hint (like Image 2)."""
    w, h = canvas_size
    bg_kind = random.choice(["solid_black", "solid_black", "vignette"])
    img = make_background(canvas_size, bg_kind)
    draw = ImageDraw.Draw(img)
    elems: List[Element] = []
    raw_lines: List[str] = []

    titles = VOCAB[lang].get("title_long", ["Settings"])
    title = random.choice(titles)
    sz_top = max(16, h // 11)
    fnt_top = font(sz_top, bold=False, lang=lang)
    # Truncated rendering: pick a long string and over-render to simulate cut-off
    truncated = False
    tw_full, th_ = measure(draw, title, fnt_top, lang)
    if tw_full > w - 8 and random.random() < 0.5:
        # Force truncation by taking substring
        for end in range(len(title), 1, -1):
            if measure(draw, title[:end], fnt_top, lang)[0] <= w - 8:
                title = title[:end]
                truncated = True
                break
    bbox = draw_centered(draw, title, int(h * 0.07), fnt_top, WHITE, w, lang=lang)
    elems.append(Element("text", bbox, text=title, selected=False, truncated=truncated,
                         color_class="white", size_class=size_class_for(sz_top)))
    raw_lines.append(title)

    # Divider
    div_y = int(h * 0.30)
    draw_horizontal_divider(draw, div_y, w)

    # Section title (bold)
    subtitles = VOCAB[lang].get("subtitles", [title])
    sub = random.choice(subtitles)
    sz_sub = max(20, h // 9)
    fnt_sub = font(sz_sub, bold=True, lang=lang)
    bbox = draw_centered(draw, sub, int(h * 0.42), fnt_sub, WHITE, w, lang=lang)
    elems.append(Element("text", bbox, text=sub, selected=True,
                         color_class="white", size_class=size_class_for(sz_sub)))
    raw_lines.append(sub)

    # Button hint
    hints = VOCAB[lang].get("button_hints", ["OK"])
    hint = random.choice(hints)
    sz_hint = max(14, h // 13)
    fnt_hint = font(sz_hint, bold=False, lang=lang)
    bbox = draw_centered(draw, hint, int(h * 0.70), fnt_hint, WHITE, w, lang=lang)
    elems.append(Element("text", bbox, text=hint, selected=False,
                         color_class="white", size_class=size_class_for(sz_hint)))
    raw_lines.append(hint)

    return img, elems, bg_kind, "\n".join(raw_lines)


# ---------- Dispatcher ----------

PATTERNS = {
    "carousel": gen_carousel,
    "message": gen_message,
    "timer": gen_timer,
    "list_check": gen_list_check,
    "toast": gen_toast,
    "title_subtitle": gen_title_subtitle,
}


def generate_one(canvas_size, lang, pattern) -> Tuple[Image.Image, Label]:
    fn = PATTERNS[pattern]
    img, elements, bg_kind, raw = fn(canvas_size, lang)
    label = Label(
        image_path="",  # filled in by caller
        pattern=pattern,
        language=lang,
        background=bg_kind,
        canvas_size=list(canvas_size),
        elements=elements,
        raw_text=raw,
    )
    return img, label


def build_balanced_schedule(items: List[str], total_count: int,
                            rng: random.Random) -> List[str]:
    unique_items = list(dict.fromkeys(items))
    if not unique_items:
        raise ValueError("At least one language must be provided.")

    base_count, remainder = divmod(total_count, len(unique_items))
    allocation_order = unique_items[:]
    rng.shuffle(allocation_order)

    schedule: List[str] = []
    for idx, item in enumerate(allocation_order):
        copies = base_count + (1 if idx < remainder else 0)
        schedule.extend([item] * copies)

    rng.shuffle(schedule)
    return schedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--chunk-count", type=int)
    ap.add_argument("--append-labels", action="store_true")
    ap.add_argument("--canvas", choices=["small", "large"], default="small")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--languages", nargs="+", default=list(VOCAB.keys()))
    ap.add_argument("--patterns", nargs="+", default=list(PATTERNS.keys()))
    args = ap.parse_args()

    unsupported_languages = [lang for lang in args.languages if lang not in VOCAB]
    if unsupported_languages:
        raise ValueError(f"Unsupported languages: {', '.join(unsupported_languages)}")
    if args.start_index < 0 or args.start_index >= args.count:
        raise ValueError("--start-index must be between 0 and count - 1.")

    canvas = CANVAS_LARGE if args.canvas == "large" else CANVAS_SMALL
    out = Path(args.output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    labels_path = out / "labels.jsonl"
    if args.start_index > 0 and not args.append_labels and labels_path.exists() and labels_path.stat().st_size > 0:
        raise ValueError("Use --append-labels when resuming in an existing output directory.")
    schedule_rng = random.Random(args.seed)
    language_schedule = build_balanced_schedule(args.languages, args.count, schedule_rng)
    end_index = args.count if args.chunk_count is None else min(args.count, args.start_index + args.chunk_count)
    items_to_generate = end_index - args.start_index
    progress_interval = max(20, items_to_generate // 100)
    file_mode = "a" if args.append_labels else "w"

    with open(labels_path, file_mode, encoding="utf-8") as fout:
        for offset, i in enumerate(range(args.start_index, end_index), start=1):
            random.seed(args.seed + i)
            lang = language_schedule[i]
            pattern = random.choice(args.patterns)
            img, label = generate_one(canvas, lang, pattern)
            fname = f"screen_{i:04d}_{pattern}_{lang}.png"
            img.save(out / "images" / fname)
            label.image_path = f"images/{fname}"
            rec = asdict(label)
            # Remove None fields for cleanliness
            for el in rec["elements"]:
                for k in list(el.keys()):
                    if el[k] is None:
                        del el[k]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if offset % progress_interval == 0 or i + 1 == end_index:
                print(f"  generated {i + 1}/{args.count}")

    print(f"\nDone. Generated indices {args.start_index} to {end_index - 1} in {out / 'images'}")
    print(f"Labels in {labels_path}")


if __name__ == "__main__":
    main()
