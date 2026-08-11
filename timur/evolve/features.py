"""
timur/evolve/features.py
═════════════════════════════════════════════════════════════════════════════
MAP-Elites davranış tanımlayıcısı (behavior descriptor) çıkarıcı.

Bir aday denklemi (TIMURModel.discovery_result.equation_str veya bir SymPy
ifadesi) alıp MAP-Elites ızgarasındaki hücre koordinatına eşler:
(complexity_bin, depth_bin, operator_family_bin) — her biri {0,1,2,3},
yani archive.py'deki 4×4×4 ızgarayı besler.

NOT: TIMURModel'in döndürdüğü equation_str, Buckingham Pi aktifken PySR'ın
ham SymPy çıktısı değil, değişken adları "display" isimlere (Pi(...) grupları,
Unicode üst simgeler) dönüştürülmüş bir string'dir (bkz. timur/__init__.py
fit()). Bu modül o string'i SymPy'a geri sokabilmek için en-iyi-çaba (best
effort) bir temizleme uygular; başarısız olursa (spec gereği) en yüksek
complexity binine düşer.

EKSEN KARARI (cardinality → nesting depth): İlk tasarımda 2. eksen
"cardinality" (ifadedeki benzersiz serbest değişken sayısı) idi. Buckingham
Pi aktif tek-Pi-grubu veri setlerinde (ör. Planck/DS1: in_features=1) bu
eksen MATEMATİKSEL OLARAK dejeneredir — her aday tek bir Pi-değişkeni
kullanabilir, başka değişken YOKTUR, dolayısıyla eksen daima aynı bin'e
düşer ve sıfır ayırt edicilik sağlar (bkz. evolve_demo_planck.py tanı
koşusu: düzeltmeden sonra 4/4 hücre cardinality_bin=0). Bunun yerine ifade
AĞACININ DERİNLİĞİ (nesting depth) kullanılıyor: complexity_bin (düğüm
SAYISI = "büyüklük") ile ORTOGONAL bir "şekil" bilgisi taşır — ör. düz-geniş
"a·x0+b·x1+c·x2+d" (11 düğüm) ile dar-derin "sin(sin(exp(-c/x)))" aynı
complexity_bin'e düşebilirken derinlikte keskin biçimde ayrışır. Cardinality
tamamen atılmadı (Seçenek 1 değil) çünkü çok-Pi-gruplu veri setlerinde
(potansiyel olarak DS3/DS4 gibi) hâlâ bilgi taşıyabilir — derinlik daha genel
ve hiçbir zaman yapısal olarak dejenere olamayan bir eksen olduğu için tercih
edildi.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Sequence, Tuple, Union

import sympy as sp

_log = logging.getLogger(__name__)

# ─── Izgara sınırları (archive.py'deki GRID_SHAPE=(4,4,4) ile eşleşir) ────────

# Üst sınırlar; i'inci eşiği aşmayan değer bin=i alır, hepsini aşan son bine düşer.
COMPLEXITY_BINS: Tuple[int, ...] = (3, 6, 10)     # ≤3→0, 4-6→1, 7-10→2, 11+→3
# İfade ağacı derinliği (nesting depth) — sınırlar gerçek aday örnekleri
# üzerinde kalibre edildi (bkz. modül docstring'indeki eksen kararı notu):
# düz/basit formlar (≤3), orta-derin (4-5), derin iç-içe (6-7), çok derin (8+).
DEPTH_BINS: Tuple[int, ...] = (3, 5, 7)           # ≤3→0, 4-5→1, 6-7→2, 8+→3

_EXP_LOG_FUNCS = {sp.exp, sp.log}
_TRIG_FUNCS = {sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.tanh,
               sp.asin, sp.acos, sp.atan}

_SYMPIFY_LOCALS = {
    "exp": sp.exp, "log": sp.log, "sin": sp.sin, "cos": sp.cos,
    "sqrt": sp.sqrt, "inv": lambda x: sp.Integer(1) / x,
}

# Parse başarısız olduğunda dönecek varsayılan (en yüksek complexity, nötr diğerleri)
_PARSE_FAIL_DESCRIPTOR: Tuple[int, int, int] = (len(COMPLEXITY_BINS), 0, 3)

_SUP_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_PI_BLOCK_RE = re.compile(r"Pi\([^()]*\)")          # "Pi(a·b⁻¹)" → tek opak sembol
_BRACKET_RE = re.compile(r"\[[^\]]*\]")             # "[Saf Sabit]" gibi ekleri at
_SUP_RUN_RE = re.compile(r"([A-Za-z0-9_\)])([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)")


def _sanitize_equation_string(eq: str) -> str:
    """equation_str'i SymPy'a sokulabilir hale getirir (en-iyi-çaba).

    - "y = " önekini ve "[...]" açıklama eklerini atar
    - "·" çarpım noktasını "*" yapar
    - "Pi(...)" bloklarını (Buckingham Pi grup isimleri) tek bir opak sembole
      indirger — içeriklerini yeniden parse etmeye çalışmaz, çünkü bunlar
      zaten tek bir Pi-grubu değişkenini temsil eder (yapısal olarak monomial,
      operatör ailesi tespitini etkilemez). AYNI "Pi(...)" METNİ ifadede birden
      çok kez geçerse (ör. "Pi(x)*sin(exp(1/Pi(x)))"), HEPSİ AYNI sembole
      eşlenir — aksi halde tek bir Pi-grubunun tekrar kullanımı, cardinality'yi
      (farklı değişken sayısını) yapay olarak şişirir (bkz. commit notu: bu
      Faz-1 demo koşusunda gözlenen gerçek bir ölçüm hatasıydı).
    - Kalan Unicode üst simgeleri ("sicaklik⁴") "**n" biçimine çevirir.
    """
    eq = eq.strip()
    if eq.startswith("y = "):
        eq = eq[4:]
    eq = _BRACKET_RE.sub("", eq).strip()
    eq = eq.replace("·", "*")

    seen: Dict[str, str] = {}
    counter = {"n": 0}

    def _pi_sub(m: re.Match) -> str:
        content = m.group(0)          # tam "Pi(...)" metni — tekilleştirme anahtarı
        if content not in seen:
            counter["n"] += 1
            seen[content] = f"PIVAR{counter['n']}"
        return seen[content]

    eq = _PI_BLOCK_RE.sub(_pi_sub, eq)

    def _sup_sub(m: re.Match) -> str:
        base, sup = m.group(1), m.group(2)
        return f"{base}**{sup.translate(_SUP_MAP)}"

    eq = _SUP_RUN_RE.sub(_sup_sub, eq)
    return eq


def _node_count(expr: sp.Basic) -> int:
    return sum(1 for _ in sp.preorder_traversal(expr))


def _expr_depth(expr: sp.Basic) -> int:
    """İfade ağacının en derin dalının uzunluğu (yaprak=1)."""
    if not expr.args:
        return 1
    return 1 + max(_expr_depth(arg) for arg in expr.args)


def _operator_family_bin(expr: sp.Basic) -> int:
    families = set()
    for node in sp.preorder_traversal(expr):
        func = getattr(node, "func", None)
        if func in _EXP_LOG_FUNCS:
            families.add("explog")
        elif func in _TRIG_FUNCS:
            families.add("trig")
    if not families:
        return 0                                    # sadece polinom (+,-,*,/,üs)
    if len(families) == 1:
        return 1 if "explog" in families else 2
    return 3                                         # karışık (2+ farklı aile)


def _bin_index(value: int, upper_bounds: Tuple[int, ...]) -> int:
    for i, upper in enumerate(upper_bounds):
        if value <= upper:
            return i
    return len(upper_bounds)


def extract_descriptor(
    expr_or_str: Union[str, sp.Basic],
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[int, int, int]:
    """MAP-Elites davranış tanımlayıcısını döndürür: (complexity_bin,
    depth_bin, operator_family_bin).

    feature_names şu an binlemeyi etkilemiyor; ileride özellik/sabit ayrımı
    yapan varyantlar için imzada tutuluyor.

    SymPy parse hatasına karşı dayanıklıdır: parse edilemezse en yüksek
    complexity binine düşer ve uyarı loglar.
    """
    try:
        if isinstance(expr_or_str, str):
            eq = _sanitize_equation_string(expr_or_str)
            expr = sp.sympify(eq, locals=dict(_SYMPIFY_LOCALS))
        else:
            expr = expr_or_str

        n_nodes = _node_count(expr)
        depth = _expr_depth(expr)
        op_bin = _operator_family_bin(expr)

        return (
            _bin_index(n_nodes, COMPLEXITY_BINS),
            _bin_index(depth, DEPTH_BINS),
            op_bin,
        )
    except Exception as exc:
        _log.warning(
            "extract_descriptor: parse hatası (%s) — en yüksek complexity "
            "binine atanıyor. ifade=%r", exc, expr_or_str,
        )
        return _PARSE_FAIL_DESCRIPTOR
