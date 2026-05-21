"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence).
"""
import re
from typing import Optional


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Step 1: Normalize Unicode characters to ASCII/LaTeX equivalents
        # This must be done FIRST before any other processing
        
        # Convert Unicode minus sign (U+2212) to ASCII hyphen-minus
        answer = answer.replace('−', '-')
        
        # Convert Unicode superscripts to LaTeX ^{} notation
        superscript_map = {
            '⁰': '^0', '¹': '^1', '²': '^2', '³': '^3', '⁴': '^4',
            '⁵': '^5', '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9',
            '⁻': '^-'
        }
        for sup, replacement in superscript_map.items():
            answer = answer.replace(sup, replacement)
        
        # Convert Unicode square root to \sqrt (without trailing space)
        answer = answer.replace('√', '\\sqrt')
        
        # Convert Unicode multiplication and division signs
        answer = answer.replace('×', r'\times ')
        answer = answer.replace('÷', '/')
        
        # Convert Unicode inequality symbols to LaTeX
        answer = answer.replace('≤', '\\leq ')
        answer = answer.replace('≥', '\\geq ')
        answer = answer.replace('≠', '\\neq ')
        answer = answer.replace('≈', '\\approx ')
        
        # Convert other Unicode math symbols
        answer = answer.replace('∞', '\\infty ')
        answer = answer.replace('π', '\\pi ')
        answer = answer.replace('θ', '\\theta ')
        answer = answer.replace('φ', '\\phi ')
        answer = answer.replace('±', '\\pm ')
        
        # Remove dollar signs (LaTeX math delimiters)
        answer = answer.replace("$", "")
        
        # Remove LaTeX display/inline style commands
        answer = re.sub(r'\\displaystyle\s*', '', answer)
        answer = re.sub(r'\\textstyle\s*', '', answer)
        
        # Remove \boxed{...} command
        answer = re.sub(r'\\boxed\s*\{([^}]*)\}', r'\1', answer)
        
        # Handle \sqrt with different bracket types
        # First, convert \sqrt(...) to \sqrt{...}
        answer = re.sub(r'\\sqrt\s*\(([^()]*)\)', r'\\sqrt{\1}', answer)
        # Then convert plain sqrt(...) to \sqrt{...}
        answer = re.sub(r'(?<!\\)sqrt\s*\(([^()]*)\)', r'\\sqrt{\1}', answer)
        
        # Remove all \text{...} (not just enclosing ones)
        answer = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', answer)
        
        # Remove \mathrm{...}
        answer = re.sub(r'\\mathrm\s*\{([^}]*)\}', r'\1', answer)
        
        # Remove LaTeX spacing commands
        answer = answer.replace('\\,', '')
        answer = answer.replace('\\;', '')
        answer = answer.replace('\\:', '')
        answer = answer.replace('\\!', '')
        answer = answer.replace('\\quad', '')
        answer = answer.replace('\\qquad', '')
        
        # Remove inline math delimiters \( \)
        answer = answer.replace('\\(', '').replace('\\)', '')
        
        # Remove display math delimiters \[ \]
        answer = answer.replace('\\[', '').replace('\\]', '')
        
        # Remove enclosing `\text{}` (legacy check, mostly redundant now)
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        
        # Normalize scientific notation (before removing units)
        answer = _normalize_scientific_notation(answer)
        
        # Remove common units (after scientific notation normalization)
        answer = _remove_units(answer)
        
        # Remove trailing punctuation EXCEPT factorial symbols
        # Be careful: '!' is used for factorial, '!!' for double factorial
        # Only remove trailing punctuation if it's not part of factorial notation
        if not re.search(r'!+$', answer):  # If doesn't end with factorials
            answer = answer.rstrip('.,;!?')
        else:
            # Remove other punctuation but preserve factorials
            answer = answer.rstrip('.,;?')
        
        return _strip_string(answer)
    except:
        return answer


def _remove_units(string):
    """
    Remove common units from the answer string.
    Examples:
        "1.57 cm" -> "1.57"
        "4.5e33 erg/s" -> "4.5e33"
        "10 kg" -> "10"
    """
    # Remove LaTeX wrappers (should already be done in normalize_answer, but just in case)
    string = re.sub(r'\\mathrm\s*\{([^}]*)\}', r'\1', string)
    
    # Common units patterns (must match word boundaries or end of string)
    # Order matters: longer/more specific patterns first
    # Note: \s* makes spaces optional around slashes and exponents
    unit_patterns = [
        # Angular velocity units (MUST be before single 's', 'rad')
        r'(?:\s*)rad\s*/\s*s(?:\s|$)',
        r'(?:\s*)deg\s*/\s*s(?:\s|$)',
        
        # Complex units with slashes and exponents (handle atoms/m^2, atoms/m2, etc.)
        r'(?:\s*)atoms\s*/\s*m\s*\^?\s*2(?:\s|$)',
        r'(?:\s*)atoms\s*/\s*m(?:\s|$)',
        # Other complex units with slashes
        r'(?:\s*)erg\s*/\s*s(?:\s|$)',
        r'(?:\s*)m\s*/\s*s\s*\^\s*2(?:\s|$)',
        r'(?:\s*)m\s*/\s*s(?:\s|$)',
        r'(?:\s*)km\s*/\s*h(?:\s|$)',
        r'(?:\s*)J\s*/\s*s(?:\s|$)',
        r'(?:\s*)(?:kg|g)\s*/\s*mol(?:\s|$)',
        # Compound units with exponents
        r'(?:\s*)cm\s*\^\s*2\s*/\s*s(?:\s|$)',
        r'(?:\s*)cm\s*\^\s*\{?\s*-?\s*3\s*\}?(?:\s|$)',
        r'(?:\s*)m\s*\^\s*2(?:\s|$)',
        r'\s*s\s*\^\s*\{?\s*-?\s*1\s*\}?(?:\s|$)',
        # Length
        r'\s*(?:nano|micro|milli|centi|kilo)?meters?(?:\s|$)',
        r'\s*\b(?:nm|mm|cm|km|m)\b(?:\s|$)',
        # Mass
        r'\s*(?:nano|micro|milli|kilo)?grams?(?:\s|$)',
        r'\s*\b(?:ng|μg|mg|g|kg)\b(?:\s|$)',
        # Time
        r'\s*seconds?(?:\s|$)',
        r'\s*minutes?(?:\s|$)',
        r'\s*hours?(?:\s|$)',
        r'\s*days?(?:\s|$)',
        r'\s*\b(?:s|min|hr|h)\b(?:\s|$)',
        # Energy/Power
        r'\s*ergs?(?:\s|$)',
        r'\s*joules?(?:\s|$)',
        r'\s*watts?(?:\s|$)',
        r'\s*\b(?:erg|J|W)\b(?:\s|$)',
        r'\s*\beV\b(?:\s|$)',
        r'\s*\bkJ\b(?:\s|$)',
        # Temperature
        r'\s*(?:kelvin|celsius|fahrenheit)(?:\s|$)',
        r'\s*\b(?:K|°C|°F)\b(?:\s|$)',
        r'\s*°\s*C(?:\s|$)',
        # Angles
        r'\s*(?:degrees?|radians?|rad)(?:\s|$)',
        r'\s*\b(?:rad|deg)\b(?:\s|$)',
        r'\s*\barcsec\b(?:\s|$)',
        r'\s*°(?:\s|$)',
        # Astronomy/Physics
        r'\s*(?:magnitudes?|mag)(?:\s|$)',
        r'\s*\batoms\b(?:\s|$)',
        # Frequency
        r'\s*\b(?:Hz|kHz|MHz|GHz)\b(?:\s|$)',
        # Pressure
        r'\s*\b(?:Pa|kPa|MPa|GPa)\b(?:\s|$)',
        # Voltage
        r'\s*\bV\b(?:\s|$)',
        # Other
        r'\s*(?:liters?|moles?|hertz|pascals?)(?:\s|$)',
        r'\s*\b(?:L|mol)\b(?:\s|$)',
    ]
    
    for pattern in unit_patterns:
        string = re.sub(pattern, '', string, flags=re.IGNORECASE)
    
    return string.strip()


def _remove_units_before_space_removal(string):
    """
    Remove units before spaces are stripped. This function handles all unit patterns
    that require spaces or word boundaries to match correctly.
    
    Should be called BEFORE string.replace(" ", "") in _strip_string.
    """
    # First, handle complex units with exponents and fractions
    # These need to be handled before simple units to avoid partial matches
    string = re.sub(r'atoms/m\^?2(?:\s|$)', '', string, flags=re.IGNORECASE)
    string = re.sub(r'atoms/m(?:\s|$)', '', string, flags=re.IGNORECASE)
    string = re.sub(r'kg/mol(?:\s|$)', '', string, flags=re.IGNORECASE)
    string = re.sub(r'g/mol(?:\s|$)', '', string, flags=re.IGNORECASE)
    string = re.sub(r'cm\^?2/s(?:\s|$)', '', string, flags=re.IGNORECASE)
    string = re.sub(r'cm\^\{-3\}(?:\s|$)', '', string, flags=re.IGNORECASE)
    string = re.sub(r'm/s\^?2(?:\s|$)', '', string, flags=re.IGNORECASE)
    string = re.sub(r'm\^?2(?:\s|$)', '', string, flags=re.IGNORECASE)
    string = re.sub(r's\^\{-1\}(?:\s|$)', '', string, flags=re.IGNORECASE)
    
    # Now handle simple units
    # Important: Don't use \b at the start - it doesn't work between digit and letter
    # Use \b only at the end to ensure we match whole unit names
    # Match with optional leading space: (?:\s*)
    # Use negative lookbehind (?<![a-zA-Z]) to avoid matching parts of LaTeX commands
    # like \times, \sin, \deg, etc.
    
    unit_patterns = [
        # Angular velocity units (rad/s, deg/s, etc.) - MUST be before 'rad' and 's'
        r'(?:\s*)rad\s*/\s*s(?:\s|$)',
        r'(?:\s*)deg\s*/\s*s(?:\s|$)',
        
        # Energy units
        r'(?<![a-zA-Z])(?:\s*)eV\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)kJ\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)erg\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)J\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)W\b(?:\s|$)',
        
        # Angle units
        r'(?<![a-zA-Z])(?:\s*)rad\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)deg\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)arcsec\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)°(?:\s|$)',
        
        # Frequency units
        r'(?<![a-zA-Z])(?:\s*)GHz\b(?:\s|$)',  # Handle longer units first
        r'(?<![a-zA-Z])(?:\s*)MHz\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)kHz\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)Hz\b(?:\s|$)',
        
        # Pressure units
        r'(?<![a-zA-Z])(?:\s*)GPa\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)MPa\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)kPa\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)Pa\b(?:\s|$)',
        
        # Length/distance units
        r'(?<![a-zA-Z])(?:\s*)km\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)cm\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)mm\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)m\b(?:\s|$)',  # After km, cm, mm
        
        # Mass units
        r'(?<![a-zA-Z])(?:\s*)kg\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)mg\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)g\b(?:\s|$)',  # After kg, mg
        
        # Length/wavelength units (Angstrom)
        r'(?<![a-zA-Z])(?:\s*)Å(?:\s|$)',
        
        # Astronomy/time units
        r'(?<![a-zA-Z])(?:\s*)Gyr\b(?:\s|$)',  # Gigayear
        r'(?<![a-zA-Z])(?:\s*)Myr\b(?:\s|$)',  # Megayear
        
        # Wavelength units
        r'(?<![a-zA-Z])(?:\s*)nm\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)pm\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)μm\b(?:\s|$)',
        
        # Time units (CRITICAL: these can conflict with LaTeX commands)
        r'(?<![a-zA-Z])(?:\s*)min\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)s\b(?:\s|$)',  # Don't match \times, \sin, etc.
        r'(?<![a-zA-Z])(?:\s*)h\b(?:\s|$)',
        
        # Temperature units
        r'(?<![a-zA-Z])(?:\s*)K\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)°C(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)°F(?:\s|$)',
        
        # Electrical units (single letters - be careful)
        r'(?<![a-zA-Z])(?:\s*)V\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)A\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)Ω(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)F\b(?:\s|$)',  # Farad
        
        # Wavenumber and inverse units (cm^{-1}, m^{-1}, etc.)
        r'(?<![a-zA-Z])(?:\s*)cm\s*\^\s*\{?\s*-\s*1\s*\}?(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)m\s*\^\s*\{?\s*-\s*1\s*\}?(?:\s|$)',
        
        # Other common units
        r'(?<![a-zA-Z])(?:\s*)atoms\b(?:\s|$)',
        r'(?<![a-zA-Z])(?:\s*)mag\b(?:\s|$)',
    ]
    
    for pattern in unit_patterns:
        string = re.sub(pattern, '', string, flags=re.IGNORECASE)
    
    return string.strip()


def _normalize_scientific_notation(string):
    """
    Normalize scientific notation to a standard format.
    Examples:
        "4.5\\times10^{33}" -> "4.5e33"
        "4.5 \\times 10^{33}" -> "4.5e33"
        "4.5*10^{33}" -> "4.5e33"
        "\\(4.5\\times10^{33}\\)" -> "4.5e33"
    """
    # Remove LaTeX delimiters like \( \) and escaped spaces
    string = re.sub(r'\\\(|\\\)|\\\\|\\ ', '', string)
    
    # Remove \mathrm{} wrappers
    string = re.sub(r'\\mathrm\s*\{([^}]*)\}', r'\1', string)
    
    # Pattern: (number) \times 10^{exponent} or (number) * 10^{exponent}
    # Matches: 4.5\times10^{33}, 4.5 \times 10^{33}, 4.5*10^{33}, etc.
    pattern = r'(-?\d+\.?\d*)\s*(?:\\times|\*|×)\s*10\s*\^\s*\{?(-?\d+)\}?'
    
    def replace_scientific(match):
        mantissa = match.group(1)
        exponent = match.group(2)
        return f"{mantissa}e{exponent}"
    
    string = re.sub(pattern, replace_scientific, string)
    
    return string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _standardize_scientific_notation(string):
    """
    Standardize scientific notation to ensure numerical equivalence.
    Examples:
        "5.6e2" -> "560" (if result is reasonable integer)
        "1.0e-8" -> "1e-8" (remove unnecessary .0)
        "2.0e3" -> "2000" or "2e3" (keep in scientific if large)
    """
    # Match scientific notation: number e exponent
    pattern = r'^(-?\d+\.?\d*)e([+-]?\d+)$'
    match = re.match(pattern, string, re.IGNORECASE)
    
    if not match:
        return string
    
    try:
        # Try to convert to float and back
        num = float(string)
        
        # For very small or very large numbers, keep in scientific notation
        # but standardize the format
        if abs(num) < 1e-4 or abs(num) >= 1e6:
            # Keep in scientific notation, but standardize format
            # Remove unnecessary ".0" in coefficient
            coefficient = match.group(1)
            exponent = match.group(2)
            
            # Simplify coefficient: "1.0" -> "1", "2.0" -> "2"
            if '.' in coefficient:
                coef_float = float(coefficient)
                if coef_float == int(coef_float):
                    coefficient = str(int(coef_float))
            
            # Simplify exponent: remove leading + and 0s
            exp_int = int(exponent)
            
            return f"{coefficient}e{exp_int}"
        else:
            # For reasonable numbers, convert to integer or decimal
            if num == int(num):
                return str(int(num))
            else:
                # Keep reasonable precision
                return str(num).rstrip('0').rstrip('.')
    except (ValueError, OverflowError):
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right) - legacy function
    string = _remove_right_units(string)
    
    # IMPORTANT: Remove units BEFORE removing spaces
    # This allows patterns like "12.4 eV" to be matched
    string = _remove_units_before_space_removal(string)
    
    # Remove common English words that appear after numbers (animals, time periods, etc.)
    # Must be done AFTER scientific notation normalization and BEFORE space removal
    word_patterns = [
        r'\s+ferrets?(?:\s|$)',
        r'\s+parsecs?(?:\s|$)',
        r'\s+years?(?:\s|$)',
        r'\s+days?(?:\s|$)',
        r'\s+bits?\s+per\s+second(?:\s|$)',
        r'\s+atoms?(?:\s|$)',
    ]
    for pattern in word_patterns:
        string = re.sub(pattern, '', string, flags=re.IGNORECASE)

    # Convert percentages to decimals (e.g., "19.2%" -> "0.192")
    # Match patterns like: 19.2%, 19.2\%, \(19.2\%\), etc.
    # Must be done BEFORE removing percentage signs
    percentage_pattern = r'(\d+\.?\d*)\s*\\?%'
    def convert_percentage(match):
        value = float(match.group(1))
        return str(value / 100)
    string = re.sub(percentage_pattern, convert_percentage, string)
    
    # remove percentage signs that weren't converted (cleanup)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)
    
    # Convert remaining simple fractions to \frac form
    # Pattern matches: letter/letter, number/number, \command/anything, anything/\command
    # This catches cases like y/z -> \frac{y}{z}, i\pi/3 -> i\frac{\pi}{3}
    string = re.sub(r'(\\[a-zA-Z]+|[a-zA-Z]|\d+\.?\d*)/(\\[a-zA-Z]+|[a-zA-Z]|\d+\.?\d*)', r'\\frac{\1}{\2}', string)
    
    # Normalize position of imaginary unit 'i' in complex numbers
    # Convert patterns like \frac{\pi}{3}i to i\frac{\pi}{3}
    # Must be done AFTER _fix_fracs so that a/b is already converted to \frac{a}{b}
    # Pattern: (expression)i -> i(expression)
    # Be careful not to match variable names like 'radius' or 'sin'
    # Only match standalone 'i' after numbers, fractions, pi, etc.
    string = re.sub(r'(\\frac\{[^}]+\}\{[^}]+\})i\b', r'i\1', string)
    string = re.sub(r'(\\pi)i\b', r'i\1', string)
    string = re.sub(r'(\d+\.?\d*)i\b', r'i\1', string)
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)
    
    # Standardize scientific notation: convert to actual number if it's a simple case
    # This handles cases like "5.6e2" -> "560" and "1.0e-8" -> "1e-8"
    string = _standardize_scientific_notation(string)

    return string