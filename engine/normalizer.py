"""
NusaVox Text Normalizer & Speech Preprocessor
Prepares raw conversational and administrative Indonesian text for flawless neural TTS synthesis.
"""

import re
from typing import Dict

# Indonesian civic & acronym phonetic dictionary
CIVIC_REPLACEMENTS: Dict[str, str] = {
    r'\bKBLI\s*56303\b': 'KBLI lima enam tiga nol tiga',
    r'\bOSS[-\s]*RBA\b': 'O S S R B A',
    r'\bOSS\b': 'O S S',
    r'\bNIB\b': 'N I B',
    r'\bSKCK\b': 'S K C K',
    r'\bMPP\b': 'M P P',
    r'\bDPMPTSP\b': 'D P M P T S P',
    r'\bBPHTB\b': 'B P H T B',
    r'\bPBB[-\s]*P2\b': 'P B B P dua',
    r'\bPBB\b': 'P B B',
    r'\bKTP[-\s]*el\b': 'K T P elektronik',
    r'\be[-\s]*KTP\b': 'K T P elektronik',
    r'\bKTP\b': 'K T P',
    r'\bKK\b': 'Kartu Keluarga',
    r'\bKIA\b': 'Kartu Identitas Anak',
    r'\bIKD\b': 'I K D',
    r'\bSIM\s*A\b': 'S I M A',
    r'\bSIM\s*C\b': 'S I M C',
    r'\bSIM\b': 'S I M',
    r'\bPNBP\b': 'P N B P',
    r'\bPB[-\s]*UMKU\b': 'P B U M K U',
    r'\bUMKM\b': 'U M K M',
    r'\bCPPOB\b': 'C P P O B',
    r'\b0\s*menit\b': 'nol menit atau langsung terbit',
    r'\b15\s*menit\b': 'lima belas menit',
    r'\bRp\.?\s*0\b': 'nol rupiah atau gratis',
    r'\bRp\.?\s*30\.?000\b': 'tiga puluh ribu rupiah',
    r'\bRp\.?\s*50\.?000\b': 'lima puluh ribu rupiah',
    r'\bRp\.?\s*100\.?000\b': 'seratus ribu rupiah',
    r'\bs\/d\b': 'sampai dengan',
    r'\bdll\b': 'dan lain-lain',
    r'\bdst\b': 'dan seterusnya',
    r'\byth\b': 'yang terhormat',
    r'\bno\.\s*([0-9]+)': r'nomor \1',
}

def clean_markdown_and_links(text: str) -> str:
    """Strip markdown formatting, URLs, code blocks, and HTML tags."""
    # Strip markdown links [Anchor](URL) -> Anchor
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Strip bare URLs
    text = re.sub(r'https?://\S+', '', text)
    # Strip bold / italic / strikethrough
    text = re.sub(r'[*#_`~]+', ' ', text)
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Strip special bullets and numbering clutter
    text = re.sub(r'^\s*[-*•]\s+', 'Poin: ', text, flags=re.MULTILINE)
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_indonesian_speech(text: str) -> str:
    """Apply phonetic corrections and formatting for natural spoken speech."""
    text = clean_markdown_and_links(text)

    for pattern, replacement in CIVIC_REPLACEMENTS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Convert remaining currency patterns Rp 1.500.000 -> rupiah 1.500.000
    text = re.sub(r'\bRp\.?\s*([0-9\.]+)\b', r'rupiah \1', text, flags=re.IGNORECASE)

    # Clean multiple punctuation
    text = re.sub(r'[\.!\?]{2,}', '.', text)
    text = re.sub(r'\s+([,\.!\?])', r'\1', text)

    # Ensure sentence ending punctuation for natural vocal cadence
    text = text.strip()
    if text and text[-1] not in ['.', '!', '?']:
        text += '.'

    # Add trailing silence cushion
    text += ' '
    return text
