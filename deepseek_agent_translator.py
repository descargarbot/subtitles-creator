import requests
import re
import time
import unicodedata
from functools import lru_cache

# -------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------
API_KEY = ""  # DeepSeek API key
INPUT_FILE = "" # file name of source, to translate (.srt)
OUTPUT_FILE = "" # file name of the output of tranlation (.srt)

# Fuente: "auto" o código ISO: it, en, ja, zh, ru, etc. IMPORTANT: Recommended ISO code.
SOURCE_LANGUAGE = ""

# Destino: es, en, it, fr, pt, de, ar, zh, ja, ko, ru, sr, etc.
# Nota: serbio es "sr", no "rs". Si pones "rs", se convierte a "sr".
TARGET_LANGUAGE = ""

BATCH_SIZE = 30
MAX_RETRIES = 5

# Auditoría
AUDIT_CONTEXT_RADIUS = 2
TARGET_SCORE_THRESHOLD = 0.55
SPANISH_SCORE_THRESHOLD = 0.52
WORD_ZIPF_THRESHOLD = 2.55
SHORT_WORD_ZIPF_THRESHOLD = 3.15
SOFT_HYPHEN = "\u00AD"

# -------------------------------------------------------------------
# Dependencias opcionales: wordfreq + lingua-language-detector
# -------------------------------------------------------------------
try:
    from wordfreq import zipf_frequency
    WORDFREQ_AVAILABLE = True
except Exception:
    WORDFREQ_AVAILABLE = False
    zipf_frequency = None

try:
    from lingua import Language, LanguageDetectorBuilder
    LINGUA_AVAILABLE = True
except Exception:
    LINGUA_AVAILABLE = False
    Language = None
    LanguageDetectorBuilder = None


# -------------------------------------------------------------------
# Normalización de códigos de idioma
# -------------------------------------------------------------------
LANG_ALIASES = {
    "rs": "sr",      # Serbia país -> serbio idioma
    "cn": "zh",      # China país -> chino idioma
    "jp": "ja",      # Japón país -> japonés idioma
    "kr": "ko",      # Corea país -> coreano idioma
    "ua": "uk",      # Ucrania país -> ucraniano idioma
    "iw": "he",      # código viejo de hebreo

    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",

    "pt-br": "pt",
    "pt-pt": "pt",

    "sr-rs": "sr",
    "sr-latn": "sr",
    "sr-cyrl": "sr",
}


def canonical_lang_code(code):
    if not code:
        return "auto"

    code = str(code).lower().strip().replace("_", "-")

    if code in LANG_ALIASES:
        return LANG_ALIASES[code]

    base = code.split("-")[0]

    if base in LANG_ALIASES:
        return LANG_ALIASES[base]

    return base


SOURCE_LANGUAGE = canonical_lang_code(SOURCE_LANGUAGE)
TARGET_LANGUAGE = canonical_lang_code(TARGET_LANGUAGE)


TARGET_LANGUAGE_LABELS = {
    "es": "español latino neutro",
    "en": "inglés natural",
    "it": "italiano natural",
    "fr": "francés natural",
    "pt": "portugués natural",
    "de": "alemán natural",
    "nl": "neerlandés natural",
    "ca": "catalán natural",
    "ro": "rumano natural",

    "zh": "chino simplificado natural",
    "ja": "japonés natural",
    "ko": "coreano natural",

    "ru": "ruso natural",
    "uk": "ucraniano natural",
    "pl": "polaco natural",
    "cs": "checo natural",
    "sk": "eslovaco natural",
    "sl": "esloveno natural",
    "hr": "croata natural",
    "sr": "serbio natural",
    "bg": "búlgaro natural",
    "mk": "macedonio natural",

    "ar": "árabe estándar moderno natural",
    "he": "hebreo natural",
    "fa": "persa natural",
    "ur": "urdu natural",

    "tr": "turco natural",
    "el": "griego natural",
    "hi": "hindi natural",
    "bn": "bengalí natural",
    "th": "tailandés natural",
    "vi": "vietnamita natural",
    "id": "indonesio natural",
    "ms": "malayo natural",
    "tl": "tagalo natural",

    "fi": "finés natural",
    "sv": "sueco natural",
    "da": "danés natural",
    "no": "noruego natural",
    "nb": "noruego bokmål natural",
    "nn": "noruego nynorsk natural",
    "is": "islandés natural",

    "hu": "húngaro natural",
    "et": "estonio natural",
    "lv": "letón natural",
    "lt": "lituano natural",
    "sq": "albanés natural",
}


def get_target_language_label(code):
    code = canonical_lang_code(code)
    return TARGET_LANGUAGE_LABELS.get(code, code)


def get_format_examples(target_lang):
    """
    Ejemplos solo para formato. Para español se conservan muy parecidos
    a los originales. Para otros idiomas se dan ejemplos breves en destino,
    para no sesgar al modelo hacia español.
    """
    lang = canonical_lang_code(target_lang)

    examples = {
        "es": """   Ejemplos válidos:
   [6] ¿Qué pasa?
   [7] No, no te preocupes.
   [8] Sí, claro, voy enseguida.

   Ejemplo con diálogo de dos líneas:
   [14] - ¿Vienes conmigo?
   - Sí, claro.

   Ejemplo con una línea larga:
   [21] No quería decirte nada hasta estar seguro.""",

        "en": """   Ejemplos válidos:
   [6] What's going on?
   [7] No, don't worry.
   [8] Yes, of course. I'll be right there.

   Ejemplo con diálogo de dos líneas:
   [14] - Are you coming with me?
   - Yes, of course.

   Ejemplo con una línea larga:
   [21] I didn't want to tell you anything until I was sure.""",

        "it": """   Ejemplos válidos:
   [6] Che succede?
   [7] No, non preoccuparti.
   [8] Sì, certo. Arrivo subito.

   Ejemplo con diálogo de dos líneas:
   [14] - Vieni con me?
   - Sì, certo.

   Ejemplo con una línea larga:
   [21] Non volevo dirti niente finché non ne fossi stato sicuro.""",

        "fr": """   Ejemplos válidos:
   [6] Qu'est-ce qui se passe ?
   [7] Non, ne t'inquiète pas.
   [8] Oui, bien sûr. J'arrive tout de suite.

   Ejemplo con diálogo de dos líneas:
   [14] - Tu viens avec moi ?
   - Oui, bien sûr.

   Ejemplo con una línea larga:
   [21] Je ne voulais rien te dire avant d'en être sûr.""",

        "pt": """   Ejemplos válidos:
   [6] O que está acontecendo?
   [7] Não, não se preocupe.
   [8] Sim, claro. Já vou.

   Ejemplo con diálogo de dos líneas:
   [14] - Você vem comigo?
   - Sim, claro.

   Ejemplo con una línea larga:
   [21] Eu não queria te dizer nada antes de ter certeza.""",

        "de": """   Ejemplos válidos:
   [6] Was ist los?
   [7] Nein, keine Sorge.
   [8] Ja, natürlich. Ich komme sofort.

   Ejemplo con diálogo de dos líneas:
   [14] - Kommst du mit mir?
   - Ja, natürlich.

   Ejemplo con una línea larga:
   [21] Ich wollte dir nichts sagen, bevor ich sicher war.""",

        "ar": """   Ejemplos válidos:
   [6] ماذا يحدث؟
   [7] لا، لا تقلق.
   [8] نعم، بالطبع. سآتي فورًا.

   Ejemplo con diálogo de dos líneas:
   [14] - هل ستأتي معي؟
   - نعم، بالطبع.

   Ejemplo con una línea larga:
   [21] لم أرد أن أخبرك بأي شيء قبل أن أتأكد.""",

        "zh": """   Ejemplos válidos:
   [6] 怎么了？
   [7] 不，别担心。
   [8] 是的，当然。我马上过去。

   Ejemplo con diálogo de dos líneas:
   [14] - 你要和我一起去吗？
   - 是的，当然。

   Ejemplo con una línea larga:
   [21] 在确定之前，我不想告诉你任何事。""",

        "ja": """   Ejemplos válidos:
   [6] どうしたの？
   [7] いや、心配しないで。
   [8] うん、もちろん。すぐ行く。

   Ejemplo con diálogo de dos líneas:
   [14] - 一緒に来る？
   - うん、もちろん。

   Ejemplo con una línea larga:
   [21] 確信が持てるまで、何も言いたくなかった。""",

        "ko": """   Ejemplos válidos:
   [6] 무슨 일이야?
   [7] 아니, 걱정하지 마.
   [8] 응, 물론이지. 바로 갈게.

   Ejemplo con diálogo de dos líneas:
   [14] - 나랑 같이 갈래?
   - 응, 물론이지.

   Ejemplo con una línea larga:
   [21] 확실해질 때까지 아무 말도 하고 싶지 않았어.""",

        "ru": """   Ejemplos válidos:
   [6] Что происходит?
   [7] Нет, не волнуйся.
   [8] Да, конечно. Я сейчас приду.

   Ejemplo con diálogo de dos líneas:
   [14] - Пойдешь со мной?
   - Да, конечно.

   Ejemplo con una línea larga:
   [21] Я не хотел ничего тебе говорить, пока не был уверен.""",

        "sr": """   Ejemplos válidos:
   [6] Шта се дешава?
   [7] Не, не брини.
   [8] Да, наравно. Одмах долазим.

   Ejemplo con diálogo de dos líneas:
   [14] - Идеш са мном?
   - Да, наравно.

   Ejemplo con una línea larga:
   [21] Нисам хтео ништа да ти кажем док не будем сигуран.""",
    }

    if lang in examples:
        return examples[lang]

    return """   Ejemplos válidos de formato:
   [6] ...
   [7] ...
   [8] ...

   Ejemplo con diálogo de dos líneas:
   [14] - ...
   - ...

   Ejemplo con una línea larga:
   [21] ..."""


# -------------------------------------------------------------------
# Lingua language detector
# -------------------------------------------------------------------
LINGUA_NAME_TO_ISO = {
    "AFRIKAANS": "af",
    "ALBANIAN": "sq",
    "ARABIC": "ar",
    "ARMENIAN": "hy",
    "AZERBAIJANI": "az",
    "BASQUE": "eu",
    "BELARUSIAN": "be",
    "BENGALI": "bn",
    "BOKMAL": "nb",
    "BOSNIAN": "bs",
    "BULGARIAN": "bg",
    "CATALAN": "ca",
    "CHINESE": "zh",
    "CROATIAN": "hr",
    "CZECH": "cs",
    "DANISH": "da",
    "DUTCH": "nl",
    "ENGLISH": "en",
    "ESPERANTO": "eo",
    "ESTONIAN": "et",
    "FINNISH": "fi",
    "FRENCH": "fr",
    "GANDA": "lg",
    "GEORGIAN": "ka",
    "GERMAN": "de",
    "GREEK": "el",
    "GUJARATI": "gu",
    "HEBREW": "he",
    "HINDI": "hi",
    "HUNGARIAN": "hu",
    "ICELANDIC": "is",
    "INDONESIAN": "id",
    "IRISH": "ga",
    "ITALIAN": "it",
    "JAPANESE": "ja",
    "KAZAKH": "kk",
    "KOREAN": "ko",
    "LATIN": "la",
    "LATVIAN": "lv",
    "LITHUANIAN": "lt",
    "MACEDONIAN": "mk",
    "MALAY": "ms",
    "MAORI": "mi",
    "MARATHI": "mr",
    "MONGOLIAN": "mn",
    "NYNORSK": "nn",
    "PERSIAN": "fa",
    "POLISH": "pl",
    "PORTUGUESE": "pt",
    "PUNJABI": "pa",
    "ROMANIAN": "ro",
    "RUSSIAN": "ru",
    "SERBIAN": "sr",
    "SHONA": "sn",
    "SLOVAK": "sk",
    "SLOVENE": "sl",
    "SOMALI": "so",
    "SOTHO": "st",
    "SPANISH": "es",
    "SWAHILI": "sw",
    "SWEDISH": "sv",
    "TAGALOG": "tl",
    "TAMIL": "ta",
    "TELUGU": "te",
    "THAI": "th",
    "TSONGA": "ts",
    "TSWANA": "tn",
    "TURKISH": "tr",
    "UKRAINIAN": "uk",
    "URDU": "ur",
    "VIETNAMESE": "vi",
    "WELSH": "cy",
    "XHOSA": "xh",
    "YORUBA": "yo",
    "ZULU": "zu",
}

ISO_TO_LINGUA_NAME = {}
for name, iso in LINGUA_NAME_TO_ISO.items():
    ISO_TO_LINGUA_NAME.setdefault(iso, name)

# Base razonable para evitar detectar contra todo si SOURCE_LANGUAGE no es auto.
LINGUA_BASE_CANDIDATES = {
    "SPANISH",
    "ENGLISH",
    "ITALIAN",
    "FRENCH",
    "PORTUGUESE",
    "GERMAN",
    "CATALAN",
    "ROMANIAN",
    "DUTCH",
    "JAPANESE",
    "CHINESE",
    "KOREAN",
    "RUSSIAN",
}

# Solo se usa completo si SOURCE_LANGUAGE="auto".
LINGUA_AUTO_EXTRA_CANDIDATES = {
    "AFRIKAANS",
    "ALBANIAN",
    "ARABIC",
    "ARMENIAN",
    "AZERBAIJANI",
    "BASQUE",
    "BELARUSIAN",
    "BENGALI",
    "BOKMAL",
    "BOSNIAN",
    "BULGARIAN",
    "CROATIAN",
    "CZECH",
    "DANISH",
    "ESPERANTO",
    "ESTONIAN",
    "FINNISH",
    "GANDA",
    "GEORGIAN",
    "GREEK",
    "GUJARATI",
    "HEBREW",
    "HINDI",
    "HUNGARIAN",
    "ICELANDIC",
    "INDONESIAN",
    "IRISH",
    "KAZAKH",
    "LATIN",
    "LATVIAN",
    "LITHUANIAN",
    "MACEDONIAN",
    "MALAY",
    "MAORI",
    "MARATHI",
    "MONGOLIAN",
    "NYNORSK",
    "PERSIAN",
    "POLISH",
    "PUNJABI",
    "SERBIAN",
    "SHONA",
    "SLOVAK",
    "SLOVENE",
    "SOMALI",
    "SOTHO",
    "SWAHILI",
    "SWEDISH",
    "TAGALOG",
    "TAMIL",
    "TELUGU",
    "THAI",
    "TSONGA",
    "TSWANA",
    "TURKISH",
    "UKRAINIAN",
    "URDU",
    "VIETNAMESE",
    "WELSH",
    "XHOSA",
    "YORUBA",
    "ZULU",
}

LINGUA_DETECTOR = None
LINGUA_LANG_TO_ISO = {}
LINGUA_ENABLED_ISOS = []


def init_lingua_detector():
    global LINGUA_LANG_TO_ISO, LINGUA_ENABLED_ISOS

    if not LINGUA_AVAILABLE:
        return None

    candidate_names = set(LINGUA_BASE_CANDIDATES)

    if SOURCE_LANGUAGE == "auto":
        candidate_names.update(LINGUA_AUTO_EXTRA_CANDIDATES)
    else:
        src_name = ISO_TO_LINGUA_NAME.get(SOURCE_LANGUAGE)
        if src_name:
            candidate_names.add(src_name)

    # Siempre incluir el idioma destino.
    target_name = ISO_TO_LINGUA_NAME.get(TARGET_LANGUAGE)
    if target_name:
        candidate_names.add(target_name)

    languages = []
    for name in sorted(candidate_names):
        lang_obj = getattr(Language, name, None)
        if lang_obj is not None:
            languages.append(lang_obj)
            LINGUA_LANG_TO_ISO[lang_obj] = LINGUA_NAME_TO_ISO.get(name)

    if len(languages) < 2:
        return None

    LINGUA_ENABLED_ISOS = sorted({
        iso for iso in LINGUA_LANG_TO_ISO.values() if iso
    })

    return LanguageDetectorBuilder.from_languages(*languages).build()


LINGUA_DETECTOR = init_lingua_detector()


def lingua_language_to_iso(lang_obj):
    if lang_obj is None:
        return None

    if lang_obj in LINGUA_LANG_TO_ISO:
        return LINGUA_LANG_TO_ISO[lang_obj]

    name = getattr(lang_obj, "name", None)
    if name and name in LINGUA_NAME_TO_ISO:
        return LINGUA_NAME_TO_ISO[name]

    text = str(lang_obj)
    if "." in text:
        name = text.split(".")[-1]
        return LINGUA_NAME_TO_ISO.get(name)

    return None


def _confidence_value(cv):
    return float(getattr(cv, "value", getattr(cv, "confidence", 0.0)))


def lingua_detect_lang(text):
    """
    Devuelve:
    {
        "iso": "es",
        "confidence": 0.83,
        "second_iso": "pt",
        "second_confidence": 0.12
    }
    o None.
    """
    if not LINGUA_DETECTOR:
        return None

    t = normalize_for_check(text)
    if not t:
        return None

    try:
        values = LINGUA_DETECTOR.compute_language_confidence_values(t)
        values = sorted(values, key=_confidence_value, reverse=True)

        if not values:
            return None

        top = values[0]
        second = values[1] if len(values) > 1 else None

        return {
            "iso": lingua_language_to_iso(top.language),
            "confidence": _confidence_value(top),
            "second_iso": lingua_language_to_iso(second.language) if second else None,
            "second_confidence": _confidence_value(second) if second else 0.0,
        }

    except Exception:
        return None


# -------------------------------------------------------------------
# Plantillas de prompts
# -------------------------------------------------------------------
CLEAN_PROMPT_TEMPLATE = """Eres un experto lingüista y corrector fonético de subtítulos en {source_lang}. 
Tu misión es corregir los errores de transcripción automática (Whisper) y el ruido, manteniendo el texto en {source_lang} sin traducir a otro idioma.

REGLAS INQUEBRANTABLES:

1) CORRECCIÓN FONÉTICA OBLIGATORIA:
   Whisper a menudo transcribe mal, creando palabras que parecen de otro idioma o no existen.
   Antes de corregir cada subtítulo, analiza si alguna palabra podría ser una alucinación fonética.
   Usa tu conocimiento profundo del idioma {source_lang} y el contexto de los subtítulos vecinos (trama, canciones, ambiente) para deducir la palabra original correcta.
   Ejemplo: si encuentras "¡RUENA!" en un entorno italiano con música, la palabra real es "Suona!" y debes escribir "Suona!" (no traducir).
   Si no puedes estar seguro pero la palabra es sospechosa, sustitúyela por el carácter soft hyphen (U+00AD).
   Corrige solo lo que sea razonablemente seguro; si dudas, no inventes contenido nuevo.

2) ALUCINACIONES CLÁSICAS (ruido obvio):
   Letras sueltas que no representen nada en el contexto, caracteres repetidos ("ECCCCC..."), fragmentos sin sentido → sustitúyelos completamente por el carácter soft hyphen (U+00AD). Solo el carácter, sin corchetes ni espacios.

3) SONIDOS SDH (entre paréntesis o corchetes): 
   - Si el texto original contiene partes entre paréntesis o corchetes que describen sonidos (ej: (risas), [música], - (noyse)), reemplaza todo ese fragmento (incluyendo paréntesis/corchetes y cualquier guión o puntuación adyacente) por un único carácter soft hyphen (U+00AD). 
    Ejemplo: 
       (noyse) → ­ 
       [people clamoring] → ­ 
        Hola (risas) mundo → Hola ­ mundo

    - Si la linea solo tiene "-" o "- ", reemplazala toda la linea de texto por el carácter soft hyphen (U+00AD) ­
     Ejemplo:
        - → ­
        
   IMPORTANTE:
   - No traduzcas estos elementos.
   - No los adaptes.
   - No los mantengas.
   - Siempre reemplázalos por U+00AD.

4) NO RESEGMENTAR:
   No fusiones, no dividas y no reordenes subtítulos.
   Conserva exactamente cada índice y cada bloque de texto.

5) FORMATO DE RESPUESTA:
   Responde exclusivamente con los subtítulos corregidos en el formato:
   [número] texto corregido en {source_lang}
   Conserva los saltos de línea originales.
   No añadas explicaciones, comentarios ni nada fuera del formato.

¡ATENCIÓN OBLIGATORIA! La respuesta debe estar EXCLUSIVAMENTE en {source_lang}. Cualquier palabra en otro idioma será rechazada.

SUBTÍTULOS A CORREGIR:
{batch_text}
"""

TRANSLATE_PROMPT_TEMPLATE = """Eres un traductor de subtítulos cinematográficos de élite, experto en el idioma {source_lang} y en {target_lang_label}.
Tu misión es traducir los siguientes subtítulos (ya limpios y corregidos) a {target_lang_label}, con naturalidad total.

REGLAS INQUEBRANTABLES:

1) TRADUCCIÓN CONTEXTUAL:
   Busca equivalencias naturales, manteniendo tono, emociones y giros idiomáticos.
   Adapta gerundios, jerga y expresiones culturales como lo haría un guionista profesional.
   Prioriza la intención y el subtexto del diálogo, no una traducción literal palabra por palabra.
   No agregues explicaciones, no amplíes el diálogo y no introduzcas matices que no estén sugeridos por el contexto.

2) VARIANTE DEL IDIOMA:
   Si el idioma de destino es español, utiliza español neutro (latino internacional).
   Evita modismos regionales específicos (España, México, Argentina, chile, etc.).
   No uses voseo (vos) ni expresiones como “vale”, “tío”, “che”, etc.

3) FORMATO DE RESPUESTA:

    - Responde exclusivamente con los subtítulos en este formato:
      [index] texto traducido

    - Conserva exactamente cada índice.
    - No resegmente subtítulos.
    - No agregues timestamps.
    - No repitas líneas de referencia, encabezados, etiquetas ni explicaciones.

    - Si un subtítulo original tiene varias líneas:
      - DEBES conservar exactamente la misma cantidad de líneas.
      - Cada línea debe mantenerse separada (usar saltos de línea).
      - No unas múltiples líneas en una sola.

    - Si el subtítulo contiene diálogo (líneas que comienzan con "-" o "—"):
      - Mantén cada intervención en una línea separada.
      - Cada línea debe comenzar con "—".
      - No fusiones intervenciones de distintos hablantes.
      
{format_examples}

¡ATENCIÓN OBLIGATORIA!
La respuesta debe estar EXCLUSIVAMENTE en {target_lang_label}.
No incluyas timestamps bajo ningún formato (por ejemplo: 00:00:00,000 --> 00:00:00,000).
Nunca generes ni inventes tiempos.
No incluyas texto fuera del formato [index] texto traducido.

SUBTÍTULOS A TRADUCIR:
{batch_text}
"""


# -------------------------------------------------------------------
# Funciones de procesamiento SRT/API
# -------------------------------------------------------------------
def parse_srt(content):
    """Extrae subtítulos del SRT preservando índice, timestamp y texto."""
    subtitles = []
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()

    blocks = re.split(r"\n\s*\n", content)

    for block in blocks:
        lines = [line.strip("\ufeff").rstrip() for line in block.split("\n") if line.strip() != ""]
        if len(lines) < 2:
            continue

        index = lines[0].strip()
        timestamp = lines[1].strip()
        text = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""

        if not re.fullmatch(r"\d+", index):
            continue

        if not re.fullmatch(
            r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}",
            timestamp
        ):
            continue

        subtitles.append({
            "index": index,
            "timestamp": timestamp,
            "text": text,
            "original_text": text,
            "batch_ok": True,
        })

    return subtitles


def _is_timestamp_like(line):
    """Detecta líneas que parecen timestamps o basura de formato SRT."""
    s = line.strip()
    if not s:
        return True

    if "-->" in s:
        return True

    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}[,.\u066B]?\d{0,3}", s):
        return True

    return False


def _parse_generic_response(response_text, batch, target_field="text"):
    """Parseo común para respuestas de la API, retorna cantidad de actualizaciones."""
    clean = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL | re.IGNORECASE)
    clean = clean.replace("\r\n", "\n").replace("\r", "\n")

    pattern = r"(?:^|\n)\s*\[?\s*(\d+)\s*\]?\s*(.*?)(?=(?:\n\s*\[?\s*\d+\s*\]?\s*)|\Z)"
    matches = re.findall(pattern, clean, flags=re.DOTALL)

    translations = {}
    for idx_str, text in matches:
        text = text.strip()

        lines = []
        for line in text.split("\n"):
            if _is_timestamp_like(line):
                continue
            lines.append(line.rstrip())

        text = "\n".join(lines).strip()
        if text:
            translations[idx_str] = text

    count = 0
    for sub in batch:
        idx = sub["index"]
        if idx in translations:
            new_text = translations[idx].replace("\u200b", "").strip()
            if new_text:
                sub[target_field] = new_text
                count += 1

    return count


def clean_batch(batch, source_lang):
    """Llamada a la API para limpiar y corregir sin traducir."""
    for sub in batch:
        sub["text"] = normalize_dashes(sub["text"])
    batch_text = "\n\n".join([f"[{sub['index']}] {sub['text']}" for sub in batch])
    prompt = CLEAN_PROMPT_TEMPLATE.format(source_lang=source_lang, batch_text=batch_text)

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8000,
        "reasoning_effort": "high"
    }

    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers=headers,
        json=data,
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"API Error (cleaning): {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def translate_batch(batch, source_lang):
    """Llamada a la API para traducir el lote ya limpio."""
    batch_text = "\n\n".join([f"[{sub['index']}] {sub['text']}" for sub in batch])

    prompt = TRANSLATE_PROMPT_TEMPLATE.format(
        source_lang=source_lang,
        target_lang_label=get_target_language_label(TARGET_LANGUAGE),
        format_examples=get_format_examples(TARGET_LANGUAGE),
        batch_text=batch_text
    )

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8000,
        "reasoning_effort": "high"
    }

    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers=headers,
        json=data,
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"API Error (translation): {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"]


# -------------------------------------------------------------------
# Utilidades para fallback y terminal
# -------------------------------------------------------------------
def split_joined_dialogue_lines(text):
    if not text:
        return text

    text = normalize_dashes(text)
    text = text.replace(" —", "\n- ").replace(" -", "\n- ")

    parts = text.split("\n")
    out = []

    for part in parts:
        s = part.strip()
        if not s:
            continue

        if s.startswith("- ") and " - " in s[2:]:
            first, rest = s[2:].split(" - ", 1)
            out.append("- " + first.strip())
            out.append("- " + rest.strip())
        else:
            out.append(s)

    return "\n".join(out)


def restore_batch_to_original(batch):
    for sub in batch:
        sub["text"] = sub.get("original_text", "")
        sub["batch_ok"] = False


def batch_index_range(batch):
    if not batch:
        return None, None
    return batch[0]["index"], batch[-1]["index"]


def print_batch_failure(batch_num, start_idx, end_idx, stage):
    print(f"\n❌ Fallo batch {batch_num} ({stage}) | del index {start_idx} al index {end_idx}")


# -------------------------------------------------------------------
# Auditoría: scripts, tokenización, wordfreq, heurísticas
# -------------------------------------------------------------------
WORD_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]+)?",
    flags=re.UNICODE
)

SCRIPT_PATTERNS = {
    "ja_kana": r"[\u3040-\u30ff]",
    "cjk": r"[\u3400-\u4dbf\u4e00-\u9fff]",
    "ko_hangul": r"[\uac00-\ud7af]",
    "cyrillic": r"[\u0400-\u04ff]",
    "arabic": r"[\u0600-\u06ff]",
    "hebrew": r"[\u0590-\u05ff]",
    "devanagari": r"[\u0900-\u097f]",
    "bengali": r"[\u0980-\u09ff]",
    "thai": r"[\u0e00-\u0e7f]",
    "greek": r"[\u0370-\u03ff]",
    "georgian": r"[\u10a0-\u10ff]",
    "armenian": r"[\u0530-\u058f]",
    "tamil": r"[\u0b80-\u0bff]",
    "telugu": r"[\u0c00-\u0c7f]",
    "gujarati": r"[\u0a80-\u0aff]",
    "gurmukhi": r"[\u0a00-\u0a7f]",
}

TARGET_ALLOWED_SCRIPTS = {
    # Latinos
    "es": {"latin"},
    "en": {"latin"},
    "it": {"latin"},
    "fr": {"latin"},
    "pt": {"latin"},
    "de": {"latin"},
    "nl": {"latin"},
    "ca": {"latin"},
    "ro": {"latin"},
    "tr": {"latin"},
    "vi": {"latin"},
    "id": {"latin"},
    "ms": {"latin"},
    "tl": {"latin"},
    "fi": {"latin"},
    "sv": {"latin"},
    "da": {"latin"},
    "no": {"latin"},
    "nb": {"latin"},
    "nn": {"latin"},
    "is": {"latin"},
    "hu": {"latin"},
    "et": {"latin"},
    "lv": {"latin"},
    "lt": {"latin"},
    "sq": {"latin"},
    "hr": {"latin"},
    "sl": {"latin"},
    "sk": {"latin"},
    "cs": {"latin"},
    "pl": {"latin"},

    # Cirílicos
    "ru": {"cyrillic"},
    "uk": {"cyrillic"},
    "bg": {"cyrillic"},
    "mk": {"cyrillic"},
    "be": {"cyrillic"},
    "kk": {"cyrillic"},
    "mn": {"cyrillic"},

    # Serbio puede ir en cirílico o latino.
    "sr": {"latin", "cyrillic"},

    # Árabe y afines
    "ar": {"arabic"},
    "fa": {"arabic"},
    "ur": {"arabic"},

    # Otros sistemas
    "he": {"hebrew"},
    "el": {"greek"},
    "hi": {"devanagari"},
    "mr": {"devanagari"},
    "bn": {"bengali"},
    "th": {"thai"},
    "ka": {"georgian"},
    "hy": {"armenian"},
    "ta": {"tamil"},
    "te": {"telugu"},
    "gu": {"gujarati"},
    "pa": {"gurmukhi"},

    # CJK
    "zh": {"cjk"},
    "ja": {"ja_kana", "cjk"},
    "ko": {"ko_hangul"},
}

ONE_LETTER_OK_BY_LANG = {
    "es": {"a", "y", "o", "u"},
    "en": {"a", "i"},
    "it": {"e", "a", "o"},
    "fr": {"à", "a", "y"},
    "pt": {"a", "e", "o"},
    "de": set(),
}


SHORT_OK_BY_LANG = {
    "es": {
        "sí", "si", "no", "ok", "okay", "vale", "claro", "bueno", "bien",
        "mal", "hola", "adiós", "gracias", "perdón", "oye", "mira",
        "vamos", "ven", "dime", "diga", "espera", "basta", "ay", "eh",
        "ah", "oh"
    },
    "en": {
        "yes", "no", "ok", "okay", "sure", "right", "hello", "hi",
        "bye", "thanks", "sorry", "please", "wait", "come", "go",
        "why", "what", "who", "where"
    },
    "it": {
        "sì", "si", "no", "ok", "certo", "bene", "ciao", "grazie",
        "scusa", "aspetta", "vieni", "vai", "cosa", "perché"
    },
    "fr": {
        "oui", "non", "ok", "d'accord", "merci", "pardon", "salut",
        "bonjour", "attends", "viens", "quoi", "pourquoi"
    },
    "pt": {
        "sim", "não", "nao", "ok", "claro", "bem", "olá", "ola",
        "obrigado", "obrigada", "desculpa", "espera", "vem"
    },
    "de": {
        "ja", "nein", "ok", "danke", "bitte", "hallo", "komm",
        "warte", "was", "warum"
    },
}


def normalize_for_check(text):
    return re.sub(r"\s+", " ", text.strip())


def strip_accents(text):
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def tokenize_words(text):
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def scripts_present(text):
    found = set()

    for name, pattern in SCRIPT_PATTERNS.items():
        if re.search(pattern, text):
            found.add(name)

    if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ſ]", text):
        found.add("latin")

    return found


def detect_disallowed_script_for_target(text, target_lang):
    """
    Devuelve el primer script no permitido para el idioma destino.
    Se permite latin como soporte para nombres propios incluso si el destino
    es árabe, chino, japonés, ruso, etc.
    """
    target_lang = canonical_lang_code(target_lang)
    present = scripts_present(text)

    allowed = TARGET_ALLOWED_SCRIPTS.get(target_lang, {"latin"})

    soft_allowed = set(allowed)
    soft_allowed.add("latin")

    for script in present:
        if script not in soft_allowed:
            return script

    return None


def has_target_script_signal(text, target_lang):
    target_lang = canonical_lang_code(target_lang)
    present = scripts_present(text)
    allowed = TARGET_ALLOWED_SCRIPTS.get(target_lang, {"latin"})
    non_latin_allowed = allowed - {"latin"}
    return bool(present.intersection(non_latin_allowed))


@lru_cache(maxsize=80000)
def language_word_frequency(token, lang):
    if not WORDFREQ_AVAILABLE:
        return 0.0

    lang = canonical_lang_code(lang)
    tok = token.lower().strip()

    if not tok:
        return 0.0

    one_letter_ok = ONE_LETTER_OK_BY_LANG.get(lang, set())
    if len(tok) == 1 and tok not in one_letter_ok:
        return 0.0

    norm = strip_accents(tok)

    try:
        if norm != tok:
            return max(zipf_frequency(tok, lang), zipf_frequency(norm, lang))
        return zipf_frequency(tok, lang)
    except Exception:
        return 0.0


def generic_target_score_details(text, lang):
    lang = canonical_lang_code(lang)
    t = normalize_for_check(text)

    if not t:
        return {
            "score": 0.0,
            "token_count": 0,
            "freq_hits": 0,
            "script_signal": False,
        }

    bad_script = detect_disallowed_script_for_target(t, lang)
    if bad_script:
        return {
            "score": 0.0,
            "token_count": 0,
            "freq_hits": 0,
            "script_signal": False,
        }

    script_signal = has_target_script_signal(t, lang)

    # Para idiomas no latinos, tener el script correcto es una señal fuerte.
    if script_signal:
        return {
            "score": 0.92,
            "token_count": 0,
            "freq_hits": 0,
            "script_signal": True,
        }

    tokens = tokenize_words(t)
    if not tokens:
        return {
            "score": 0.0,
            "token_count": 0,
            "freq_hits": 0,
            "script_signal": False,
        }

    freq_hits = 0
    for tok in tokens:
        if language_word_frequency(tok, lang) >= WORD_ZIPF_THRESHOLD:
            freq_hits += 1

    ratio = freq_hits / len(tokens)

    score = ratio

    if len(tokens) >= 3 and ratio >= 0.70:
        score += 0.10

    score = max(0.0, min(score, 1.0))

    return {
        "score": score,
        "token_count": len(tokens),
        "freq_hits": freq_hits,
        "script_signal": False,
    }


def generic_target_score(text, lang):
    return generic_target_score_details(text, lang)["score"]


# -------------------------------------------------------------------
# Heurística especial para español
# -------------------------------------------------------------------
SPANISH_COMMON_WORDS = {
    "a", "al", "algo", "alguien", "algún", "alguna", "algunas", "algunos",
    "allá", "allí", "ahí", "aquí", "ante", "antes", "aunque",
    "bajo", "bastante", "bien", "bueno", "buena", "buenos", "buenas",
    "cada", "casi", "claro", "como", "cómo", "con", "contra", "cual", "cuál",
    "cuando", "cuándo", "cuanto", "cuánto", "de", "del", "desde",
    "donde", "dónde", "durante", "e", "el", "él", "ella", "ellas", "ellos",
    "en", "entre", "era", "eran", "eras", "eres", "es", "esa", "esas",
    "ese", "esos", "esta", "está", "estaba", "estaban", "estado", "estamos",
    "están", "estar", "estas", "este", "esto", "estos", "estoy",
    "fue", "fui", "fuera", "fueron", "gracias", "ha", "había", "habían",
    "hacer", "haces", "hace", "hacemos", "hacen", "hacia", "hasta", "hay",
    "hola", "hombre", "la", "las", "le", "les", "lo", "los", "mal",
    "más", "me", "menos", "mi", "mí", "mis", "mucho", "mucha", "muchos",
    "muchas", "muy", "nada", "nadie", "ni", "no", "nos", "nosotros",
    "nuestra", "nuestro", "nuestras", "nuestros", "nunca", "o", "otra",
    "otro", "otras", "otros", "para", "pero", "poco", "por", "porque",
    "porqué", "qué", "que", "quien", "quién", "quiere", "quieres",
    "quiero", "queremos", "se", "sé", "sea", "ser", "será", "sería",
    "si", "sí", "siempre", "sin", "sobre", "soy", "su", "sus", "tal",
    "también", "tampoco", "tan", "tanto", "te", "tener", "tengo", "tenemos",
    "tendrá", "tendría", "tienes", "tiene", "tienen", "ti", "todo", "toda",
    "todos", "todas", "tras", "tu", "tú", "tus", "un", "una", "unas",
    "uno", "unos", "usted", "ustedes", "va", "vamos", "van", "vas", "voy",
    "venir", "vengo", "viene", "vienes", "ver", "verdad", "vez", "y", "ya",
    "yo",

    "adiós", "ahora", "amigo", "amiga", "amor", "año", "años", "abogado",
    "ayuda", "ayúdame", "basta", "carajo", "cariño", "casa", "chico", "chica",
    "cielo", "cosa", "cosas", "debe", "debes", "debo", "decir", "diga", "dime",
    "disculpa", "disculpe", "dios", "espera", "esperen", "fácil", "feliz",
    "gente", "hijo", "hija", "mira", "mire", "momento", "mujer",
    "necesito", "niño", "niña", "oye", "pagar", "papá", "mamá", "perdón",
    "perdone", "puedo", "puedes", "puede", "saber", "señor", "señora",
    "señorita", "solo", "sólo", "ven", "venga", "vida",
}

SPANISH_SUFFIXES = (
    "ción", "ciones", "mente", "dad", "dades", "ando", "iendo",
    "ado", "ada", "ados", "adas", "ido", "ida", "idos", "idas",
    "aré", "eré", "iré", "aba", "abas", "aban", "ía", "ías", "ían"
)


def has_spanish_accent_or_n(text):
    return bool(re.search(r"[áéíóúüñÁÉÍÓÚÜÑ]", text))


def has_inverted_spanish_punctuation(text):
    return "¿" in text or "¡" in text


@lru_cache(maxsize=50000)
def spanish_word_frequency(token):
    return language_word_frequency(token, "es")


def is_spanish_known_word(token):
    tok = token.lower()
    norm = strip_accents(tok)

    if len(tok) == 1 and tok not in {"a", "y", "o", "u"}:
        return False

    if tok in SPANISH_COMMON_WORDS or norm in SPANISH_COMMON_WORDS:
        return True

    if WORDFREQ_AVAILABLE and spanish_word_frequency(tok) >= WORD_ZIPF_THRESHOLD:
        return True

    return False


def is_probable_proper_name_only(text):
    t = text.strip()

    if has_inverted_spanish_punctuation(t) or has_spanish_accent_or_n(t):
        return False

    words_original = [m.group(0) for m in WORD_RE.finditer(t)]
    if not words_original or len(words_original) > 3:
        return False

    proper_like = 0
    for w in words_original:
        lw = w.lower()
        if lw in SPANISH_COMMON_WORDS:
            return False
        if len(w) >= 2 and w[0].isupper() and not w.isupper():
            proper_like += 1

    return proper_like == len(words_original)


def has_proper_name_token(text):
    words_original = [m.group(0) for m in WORD_RE.finditer(text)]

    for w in words_original:
        lw = w.lower()
        if lw in SPANISH_COMMON_WORDS:
            continue
        if len(w) >= 2 and w[0].isupper() and not w.isupper():
            return True

    return False


def spanish_score_details(text):
    t = normalize_for_check(text)
    tokens = tokenize_words(t)

    if not tokens:
        return {
            "score": 0.0,
            "token_count": 0,
            "common_hits": 0,
            "freq_hits": 0,
            "recognized_hits": 0,
            "suffix_hits": 0,
        }

    common_hits = 0
    freq_hits = 0
    recognized_hits = 0
    suffix_hits = 0

    for tok in tokens:
        norm = strip_accents(tok)

        is_common = tok in SPANISH_COMMON_WORDS or norm in SPANISH_COMMON_WORDS
        freq = spanish_word_frequency(tok) if WORDFREQ_AVAILABLE else 0.0
        is_freq = freq >= WORD_ZIPF_THRESHOLD

        if is_common:
            common_hits += 1
        if is_freq:
            freq_hits += 1
        if is_common or is_freq:
            recognized_hits += 1

        if len(tok) >= 5 and any(tok.endswith(suf) for suf in SPANISH_SUFFIXES):
            suffix_hits += 1

    n = len(tokens)
    common_ratio = common_hits / n
    freq_ratio = freq_hits / n
    recognized_ratio = recognized_hits / n
    suffix_ratio = suffix_hits / n

    score = 0.0

    if has_inverted_spanish_punctuation(t):
        score += 0.10

    if re.search(r"[ñÑ]", t):
        score += 0.10

    if re.search(r"[áéíóúüÁÉÍÓÚÜ]", t):
        score += 0.05

    score += common_ratio * 0.38
    score += freq_ratio * 0.40
    score += suffix_ratio * 0.10

    if n >= 3 and recognized_ratio >= 0.70:
        score += 0.10

    bad_script = detect_disallowed_script_for_target(t, "es")
    if bad_script:
        score -= 0.45

    score = max(0.0, min(score, 1.0))

    return {
        "score": score,
        "token_count": n,
        "common_hits": common_hits,
        "freq_hits": freq_hits,
        "recognized_hits": recognized_hits,
        "suffix_hits": suffix_hits,
    }


def spanish_score(text):
    return spanish_score_details(text)["score"]


def looks_spanish(text):
    t = normalize_for_check(text)

    if not t or t == SOFT_HYPHEN:
        return False

    if detect_disallowed_script_for_target(t, "es"):
        return False

    tokens = tokenize_words(t)
    if not tokens:
        return False

    details = spanish_score_details(t)
    n = details["token_count"]
    score = details["score"]
    recognized = details["recognized_hits"]
    common_hits = details["common_hits"]

    strong_signal = has_inverted_spanish_punctuation(t) or has_spanish_accent_or_n(t)

    if n == 1:
        tok = tokens[0]

        if tok in SHORT_OK_BY_LANG["es"]:
            return True

        if is_probable_proper_name_only(t):
            return False

        if strong_signal and is_spanish_known_word(tok):
            return True

        if WORDFREQ_AVAILABLE and spanish_word_frequency(tok) >= SHORT_WORD_ZIPF_THRESHOLD:
            return True

        return False

    if n == 2:
        if recognized == 2:
            return True

        if strong_signal and recognized >= 1:
            return True

        if common_hits >= 1 and has_proper_name_token(t):
            return True

        return score >= 0.65

    return score >= SPANISH_SCORE_THRESHOLD


# -------------------------------------------------------------------
# Auditoría genérica por idioma destino
# -------------------------------------------------------------------
def looks_target_language(text, target_lang):
    target_lang = canonical_lang_code(target_lang)
    t = normalize_for_check(text)

    if not t or t == SOFT_HYPHEN:
        return False

    if target_lang == "es":
        return looks_spanish(t)

    bad_script = detect_disallowed_script_for_target(t, target_lang)
    if bad_script:
        return False

    details = generic_target_score_details(t, target_lang)
    score = details["score"]
    tokens = tokenize_words(t)

    # Para idiomas no latinos: si aparece el script correcto, lo aceptamos.
    if details["script_signal"]:
        return True

    if not tokens:
        return False

    short_ok = SHORT_OK_BY_LANG.get(target_lang, set())

    if len(tokens) == 1:
        tok = tokens[0]

        if tok in short_ok:
            return True

        if WORDFREQ_AVAILABLE and language_word_frequency(tok, target_lang) >= SHORT_WORD_ZIPF_THRESHOLD:
            return True

        info = lingua_detect_lang(t)
        if info and info.get("iso") == target_lang and info.get("confidence", 0.0) >= 0.45:
            return True

        return False

    if len(tokens) == 2:
        if details["freq_hits"] == 2:
            return True

        info = lingua_detect_lang(t)
        if info and info.get("iso") == target_lang and info.get("confidence", 0.0) >= 0.40:
            return True

        return score >= 0.70

    if score >= TARGET_SCORE_THRESHOLD:
        return True

    info = lingua_detect_lang(t)
    if info and info.get("iso") == target_lang and info.get("confidence", 0.0) >= 0.35:
        return True

    return False


def target_language_score(text, target_lang):
    target_lang = canonical_lang_code(target_lang)

    if target_lang == "es":
        return spanish_score(text)

    return generic_target_score(text, target_lang)


def is_trivial_or_ambiguous(text):
    t = normalize_for_check(text)

    if not t:
        return True

    if t == SOFT_HYPHEN:
        return True

    if len(t) <= 3:
        return True

    if re.fullmatch(r"[\W_]+", t):
        return True

    return False


# Palabras distintivas para detectar si una traducción quedó en idioma fuente.
# Se usa como señal auxiliar. No debe contener palabras demasiado compartidas.
SOURCE_RESIDUE_WORDS = {
    "it": {
        "che", "non", "sono", "sei", "siamo", "siete", "cosa", "così",
        "perché", "perche", "allora", "deve", "devi", "pagare", "scusa",
        "scusami", "scusatemi", "grazie", "prego", "andiamo", "voglio",
        "vuoi", "vuole", "posso", "puoi", "può", "questo", "questa",
        "quello", "quella", "buon", "buona", "anno", "avvocato", "dica",
        "facile", "vestirmi"
    },
    "en": {
        "the", "you", "your", "yours", "are", "is", "am", "was", "were",
        "what", "why", "where", "when", "how", "who", "don't", "dont",
        "can't", "cant", "won't", "wont", "would", "could", "should",
        "have", "has", "had", "this", "that", "these", "those", "please",
        "sorry", "thanks", "thank", "hello", "goodbye"
    },
    "fr": {
        "je", "tu", "vous", "nous", "ils", "elles", "suis", "êtes",
        "etre", "être", "avec", "sans", "pourquoi", "quand", "comment",
        "quoi", "pas", "plus", "très", "tres", "monsieur", "madame",
        "merci", "pardon", "bonjour", "bonsoir"
    },
    "pt": {
        "você", "voce", "vocês", "voces", "estou", "está", "esta",
        "estamos", "estão", "estao", "não", "nao", "obrigado",
        "obrigada", "desculpa", "desculpe", "porquê", "também",
        "tambem", "agora"
    },
    "de": {
        "ich", "du", "sie", "wir", "ihr", "nicht", "kein", "keine",
        "was", "warum", "wann", "wie", "wo", "bitte", "danke",
        "entschuldigung", "guten", "hallo"
    },
    "es": {
        "qué", "que", "porque", "porqué", "estoy", "estás", "está",
        "tengo", "tienes", "tiene", "quiero", "quieres", "quiere",
        "puedo", "puedes", "puede", "vamos", "perdón", "gracias",
        "señor", "señora"
    },
}


def source_residue_score(text, source_lang):
    src = canonical_lang_code(source_lang)

    if src == "auto":
        return {
            "hits": 0,
            "token_count": 0,
            "ratio": 0.0,
            "matched": []
        }

    # Si el destino es igual a la fuente, no tiene sentido marcar residuo.
    if src == TARGET_LANGUAGE:
        return {
            "hits": 0,
            "token_count": 0,
            "ratio": 0.0,
            "matched": []
        }

    residue_words = SOURCE_RESIDUE_WORDS.get(src)

    if not residue_words:
        return {
            "hits": 0,
            "token_count": 0,
            "ratio": 0.0,
            "matched": []
        }

    tokens = tokenize_words(text)
    if not tokens:
        return {
            "hits": 0,
            "token_count": 0,
            "ratio": 0.0,
            "matched": []
        }

    matched = []
    for tok in tokens:
        norm = strip_accents(tok)
        if tok in residue_words or norm in residue_words:
            matched.append(tok)

    hits = len(matched)
    ratio = hits / len(tokens)

    return {
        "hits": hits,
        "token_count": len(tokens),
        "ratio": ratio,
        "matched": matched
    }


def get_context_window(subtitles, pos, field="original_text", radius=2):
    start = max(0, pos - radius)
    end = min(len(subtitles), pos + radius + 1)

    parts = []
    for j in range(start, end):
        txt = subtitles[j].get(field, "")
        if txt:
            parts.append(txt)

    return " ".join(parts).strip()


def get_source_lang_for_review(original, original_context=None):
    """
    Si SOURCE_LANGUAGE no es auto, se usa ese idioma.
    Si es auto, se detecta usando ventana contextual.
    """
    if SOURCE_LANGUAGE != "auto":
        return SOURCE_LANGUAGE

    text_for_detection = original_context or original
    info = lingua_detect_lang(text_for_detection)

    if info and info.get("iso"):
        return info["iso"]

    return "auto"


def review_translation(sub, original_context=None):
    original = sub["original_text"].strip()
    translated = sub["text"].strip()

    if translated == SOFT_HYPHEN or translated == "":
        return "RUIDO_O_VACIO"

    if original == translated:
        if is_trivial_or_ambiguous(translated):
            return "AMBIGUO"
        return "IDENTICO"

    source_iso = get_source_lang_for_review(original, original_context)

    bad_script = detect_disallowed_script_for_target(translated, TARGET_LANGUAGE)
    if bad_script:
        return f"POSIBLE_SIN_TRADUCIR(script={bad_script}, target={TARGET_LANGUAGE})"

    target_score = target_language_score(translated, TARGET_LANGUAGE)

    residue = source_residue_score(translated, source_iso)
    if residue["token_count"] >= 2:
        residue_strong = residue["hits"] >= 2 or residue["ratio"] >= 0.60
        if residue_strong and target_score < 0.60:
            matched = ",".join(residue["matched"][:5])
            return f"POSIBLE_SIN_TRADUCIR({source_iso}, hits={residue['hits']}, words={matched})"

    if looks_target_language(translated, TARGET_LANGUAGE):
        return "OK"

    trans_info = lingua_detect_lang(translated)
    trans_iso = trans_info["iso"] if trans_info else None
    trans_conf = trans_info["confidence"] if trans_info else 0.0

    if trans_iso == TARGET_LANGUAGE and trans_conf >= 0.35:
        return "OK"

    if is_trivial_or_ambiguous(original) or is_trivial_or_ambiguous(translated):
        return "AMBIGUO"

    return (
        f"SOSPECHOSO({source_iso}->{trans_iso}, "
        f"target={TARGET_LANGUAGE}, score={target_score:.2f}, conf={trans_conf:.2f})"
    )

def normalize_dashes(text):
    return (
        text.replace("—", "-")  # em dash
            .replace("–", "-")  # en dash
            .replace("−", "-")  # minus unicode
            .replace("-", "-")  # non-breaking hyphen
    )
# -------------------------------------------------------------------
# Ejecución principal
# -------------------------------------------------------------------
def main():
    start_time = time.time()

    print(f"📖 Leyendo: {INPUT_FILE}")
    print(f"🌍 SOURCE_LANGUAGE: {SOURCE_LANGUAGE}")
    print(f"🎯 TARGET_LANGUAGE: {TARGET_LANGUAGE} ({get_target_language_label(TARGET_LANGUAGE)})")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {INPUT_FILE}")
        return

    subtitles = parse_srt(content)
    batches = [subtitles[i:i + BATCH_SIZE] for i in range(0, len(subtitles), BATCH_SIZE)]

    print(f"✅ Total: {len(subtitles)} subtítulos | {len(batches)} lotes.")

    if not WORDFREQ_AVAILABLE:
        print("⚠️ Aviso: wordfreq no está instalado. Instala con:")
        print("   pip install wordfreq")
        print("⚠️ La auditoría por frecuencia de palabras será más débil.")

    if not LINGUA_AVAILABLE:
        print("⚠️ Aviso: lingua-language-detector no está instalado. Instala con:")
        print("   pip install lingua-language-detector")
    elif LINGUA_DETECTOR:
        print(f"🧪 Lingua activo. Idiomas candidatos: {', '.join(LINGUA_ENABLED_ISOS)}")
    else:
        print("⚠️ Lingua está instalado, pero no se pudo inicializar el detector.")

    # 1. Limpieza + traducción
    for i, batch in enumerate(batches, 1):
        batch_start_idx, batch_end_idx = batch_index_range(batch)
        batch_ok = True

        # Guardamos el original por si hay que restaurar todo el lote.
        for sub in batch:
            sub["text"] = sub.get("text", "")
            sub["batch_ok"] = True

        # ----- Paso 1: Limpieza fonética -----
        print(f"🧹 Limpiando lote {i}/{len(batches)}...", end="\r")
        success_clean = False
        retries = 0

        while retries < MAX_RETRIES and not success_clean:
            try:
                raw_clean = clean_batch(batch, SOURCE_LANGUAGE)
                raw_clean = normalize_dashes(raw_clean)
                clean_count = _parse_generic_response(raw_clean, batch, target_field='text')
                
                if clean_count == len(batch):
                    success_clean = True
                else:
                    retries += 1
                    print(f"\n⚠️ Limpieza lote {i}: {clean_count}/{len(batch)} aplicados. Reintentando...")
                    time.sleep(5)

            except Exception as e:
                print(f"\n⚠️ Error limpiando lote {i}: {e}")
                retries += 1
                time.sleep(5)

        if not success_clean:
            batch_ok = False
            print_batch_failure(i, batch_start_idx, batch_end_idx, "limpieza")
            restore_batch_to_original(batch)
            continue

        # ----- Paso 2: Traducción contextual -----
        print(f"🌐 Traduciendo lote {i}/{len(batches)} a {TARGET_LANGUAGE}...", end="\r")
        success_trans = False
        retries = 0

        while retries < MAX_RETRIES and not success_trans:
            try:
                raw_trans = translate_batch(batch, SOURCE_LANGUAGE)
                raw_trans = normalize_dashes(raw_trans)
                trans_count = _parse_generic_response(raw_trans, batch, target_field='text')

                if trans_count == len(batch):
                    success_trans = True
                else:
                    retries += 1
                    print(f"\n⚠️ Traducción lote {i}: {trans_count}/{len(batch)} aplicados. Reintentando...")
                    time.sleep(5)

            except Exception as e:
                print(f"\n⚠️ Error traduciendo lote {i}: {e}")
                retries += 1
                time.sleep(5)

        if not success_trans:
            batch_ok = False
            print_batch_failure(i, batch_start_idx, batch_end_idx, "traducción")
            restore_batch_to_original(batch)
        else:
            print(f"✅ Lote {i} completado (limpio + traducido).")

        # Si el lote quedó OK, se marca así; si no, los sub ya fueron restaurados.
        for sub in batch:
            sub["batch_ok"] = batch_ok

        time.sleep(4)

    # 2. Auditoría
    print("\n\n" + "=" * 70)
    print("🔍 REVISIÓN: SOLO CASOS DUDOSOS")
    print("=" * 70)

    issues = []
    counters = {
        "IDENTICO": 0,
        "AMBIGUO": 0,
        "RUIDO_O_VACIO": 0,
        "POSIBLE_SIN_TRADUCIR": 0,
        "SOSPECHOSO": 0,
        "OK": 0,
    }

    for pos, sub in enumerate(subtitles):
        original_context = get_context_window(
            subtitles,
            pos,
            field="original_text",
            radius=AUDIT_CONTEXT_RADIUS
        )

        state = review_translation(sub, original_context=original_context)

        if state == "OK":
            counters["OK"] += 1
            continue

        if state == "IDENTICO":
            counters["IDENTICO"] += 1
        elif state == "AMBIGUO":
            counters["AMBIGUO"] += 1
        elif state == "RUIDO_O_VACIO":
            counters["RUIDO_O_VACIO"] += 1
        elif state.startswith("POSIBLE_SIN_TRADUCIR"):
            counters["POSIBLE_SIN_TRADUCIR"] += 1
        elif state.startswith("SOSPECHOSO"):
            counters["SOSPECHOSO"] += 1
        else:
            counters["SOSPECHOSO"] += 1

        issues.append((sub["index"], state, sub["original_text"], sub["text"]))

    if not issues:
        print("✨ No hubo casos dudosos para revisar.")
    else:
        for idx, state, original, translated in issues:
            print(f"Índice: {idx} | Estado: {state}")
            print(f"  Original: {original}")
            print(f"  Traducido: {translated}")
            print("-" * 50)

    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)
    print(f"IDENTICO:               {counters['IDENTICO']}")
    print(f"OK:                     {counters['OK']}")
    print(f"AMBIGUO:                {counters['AMBIGUO']}")
    print(f"RUIDO_O_VACIO:          {counters['RUIDO_O_VACIO']}")
    print(f"POSIBLE_SIN_TRADUCIR:   {counters['POSIBLE_SIN_TRADUCIR']}")
    print(f"SOSPECHOSO:             {counters['SOSPECHOSO']}")
    print("=" * 70)

    # 3. Guardado
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles):
            if sub.get("batch_ok", True):
                final_text = sub.get("text", "")
                final_text = split_joined_dialogue_lines(final_text)
                if not final_text.strip():
                    final_text = sub.get("original_text", "")
            else:
                final_text = sub.get("original_text", "")

            f.write(f"{sub['index']}\n{sub['timestamp']}\n{final_text}\n")
            if i < len(subtitles) - 1:
                f.write("\n")

    print(f"\n💾 Archivo traducido guardado en: {OUTPUT_FILE}")

    elapsed_seconds = time.time() - start_time
    elapsed_minutes = elapsed_seconds / 60
    print(f"⏱️ Tiempo total de ejecución: {elapsed_minutes:.2f} minutos.")


if __name__ == "__main__":
    main()
