"""
timur/evolve/judge.py
═════════════════════════════════════════════════════════════════════════════
Yargıç (Judge) — Faz 3: fiziksel kısıt testleri tamamlandı.

fitness = validation R² (Faz 1'den beri aynı tanım). Faz 1'de accept() sadece
R²≥0.0 kontrol ediyordu (pratikte hiçbir şeyi elemiyordu). evaluate() artık
BEŞ bağımsız, açılıp-kapanabilir testi TEK bir akışta birleştirir:

    1. R² eşiği          — varsayılan 0.5 (parametrik)
    2. Sonluluk testi     — girdi alanının bir örnekleminde NaN/Inf var mı?
    3. Limit davranışı    — opsiyonel LimitConstraint listesi (None-safe):
                            monotonluk + x→0/x→∞'da sonluluk
    4. Simetri            — opsiyonel SymmetryConstraint listesi (None-safe):
                            çift/tek fonksiyon ya da ölçek simetrisi
    5. Korunum/toplam     — opsiyonel ConservationConstraint listesi
                            (None-safe): integralin sabit kalması ya da
                            bilinen bir değere eşit olması

Her testin sonucu candidate.metadata["judge_report"]'a yazılır (şeffaflık —
TIMUR'un XAI ilkesiyle uyumlu: neden kabul/red edildiği izlenebilir olmalı).

Testler candidate.frozen_fn ÜZERİNDEN DEĞİL, candidate.metadata["model"]
(fit edilmiş TIMURModel) üzerinden predict_symbolic() ile NUMERİK çalışır —
bkz. mutate.py'deki frozen_fn/predict_symbolic ayrımı notu (frozen_fn
Buckingham Pi aktifken boyutsuz Pi-uzayında çalışır, predict_symbolic ise
SI-uzayına ters-dönüşümü de uygular).

Boyutsal tutarlılık için ayrı bir test YOKTUR — bu kısıt TIMURModel'in
Pi-uzayı araması ile zaten yapısal olarak sağlanıyor (bkz.
timur/symbolic/dimensions.py).

GERİYE UYUMLULUK: 4. ve 5. testler için constraint listesi verilmezse (None,
varsayılan) evaluate()'in davranışı Faz 2 ile BİREBİR AYNI kalır — bu testler
sessizce "geçti" sayılıp atlanır (aşağıdaki test_symmetry/test_conservation
docstring'lerine bakın).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import r2_score

from timur.evolve.archive import Candidate

DEFAULT_ACCEPT_THRESHOLD = 0.5      # Faz 1'deki 0.0'dan yükseltildi — gerçekten eler
DEFAULT_FINITENESS_SAMPLES = 200    # sonluluk testi için rastgele örnek sayısı
DEFAULT_ZERO_FRACTION = 0.01        # "x→0" testi: gözlenen minimumun bu katsayı kadarı
DEFAULT_INF_MULTIPLIER = 100.0      # "x→∞" testi: gözlenen maksimumun kaç katı
DEFAULT_MONOTONIC_SAMPLES = 20      # monotonluk testi için sıralı örnek sayısı
DEFAULT_SYMMETRY_SAMPLES = 10       # simetri testi için nokta sayısı
DEFAULT_SYMMETRY_TOL = 0.05         # simetri testi bağıl toleransı
DEFAULT_CONSERVATION_POINTS = 50    # korunum integrali için trapz örnek sayısı
DEFAULT_CONSERVATION_TOL = 0.10     # korunum testi bağıl toleransı


def score(candidate: Candidate, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """fitness = validation R² (Faz 1 ile aynı).

    candidate.metadata["model"] (fit edilmiş TIMURModel) üzerinden
    predict_symbolic() çağrılır — candidate.frozen_fn DEĞİL (yukarıdaki
    modül notuna bakın).
    """
    model = candidate.metadata.get("model")
    if model is None:
        return float("-inf")
    try:
        y_pred = model.predict_symbolic(X_val)
    except Exception:
        return float("-inf")
    if not np.all(np.isfinite(y_pred)):
        return float("-inf")
    return float(r2_score(y_val, y_pred))


def accept(candidate: Candidate, threshold: float = DEFAULT_ACCEPT_THRESHOLD) -> bool:
    """Geriye-dönük uyumlu basit kontrol: candidate.fitness (zaten score() ile
    hesaplanmış olmalı) eşiğin altındaysa reddeder. Sonluluk/limit testlerini
    ÇALIŞTIRMAZ — tam test paketi için evaluate() kullanın."""
    return candidate.fitness >= threshold


# ─── Test: Sonluluk / geçerlilik ──────────────────────────────────────────────

def test_finiteness(
    candidate: Candidate,
    X_domain: np.ndarray,
    n_samples: int = DEFAULT_FINITENESS_SAMPLES,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Girdi alanının (X_domain'in gözlenen min-max aralığı İÇİNDE — dışına
    taşmadan) rastgele bir örnekleminde adayın NaN/Inf ürettiği yer var mı?"""
    model = candidate.metadata.get("model")
    if model is None:
        return False, {"reason": "model yok"}

    rng = rng or np.random.default_rng()
    lo, hi = X_domain.min(axis=0), X_domain.max(axis=0)
    sample = rng.uniform(lo, hi, size=(n_samples, X_domain.shape[1]))

    try:
        y_sample = model.predict_symbolic(sample)
    except Exception as exc:
        return False, {"reason": f"predict_symbolic hata: {exc}"}

    n_bad = int(np.sum(~np.isfinite(y_sample)))
    passed = n_bad == 0
    return passed, {"n_samples": n_samples, "n_nonfinite": n_bad}


# ─── Test: Limit davranışı (opsiyonel, None-safe) ────────────────────────────

@dataclass
class LimitConstraint:
    """Bir özellik (SI-uzayı X sütunu) için beklenen sınır/monotonluk davranışı.

    feature_index  : X_domain'deki (SI-uzayı) sütun indeksi
    behavior       : 'finite_at_zero' | 'finite_at_inf' |
                      'monotonic_increasing' | 'monotonic_decreasing'
    fixed_others   : {sütun_indeksi: değer} — diğer sütunlar bu değerlerde
                      sabitlenir; verilmezse X_domain'in sütun ortalamaları
                      kullanılır
    zero_fraction  : 'finite_at_zero' testi X_domain'in o sütundaki gözlenen
                      minimumunun bu kesri kadar bir değerde değerlendirilir
                      (mutlak epsilon DEĞİL — birim ölçeğinden bağımsız olsun
                      diye; ör. dalga boyu metre cinsinden 1e-7 mertebesinde,
                      sabit bir 1e-6 epsilon "sıfıra yakın" anlamına gelmez)
    inf_multiplier : 'finite_at_inf' testi gözlenen maksimumun bu katında
                      değerlendirilir
    """
    feature_index: int
    behavior: str
    fixed_others: Optional[Dict[int, float]] = None
    zero_fraction: float = DEFAULT_ZERO_FRACTION
    inf_multiplier: float = DEFAULT_INF_MULTIPLIER


def test_limit_behavior(
    candidate: Candidate,
    X_domain: np.ndarray,
    constraints: Optional[List[LimitConstraint]],
) -> Tuple[bool, Dict[str, Any]]:
    """constraints=None (veya boş liste) ise test ATLANIR ve geçti sayılır
    (None-safe). Her constraint numerik olarak değerlendirilir:
      - finite_at_zero / finite_at_inf : ilgili sütun gözlenen min'e yakın /
        maks'ın katına taşınır, diğer sütunlar sabit tutulur, çıktı sonlu mu?
      - monotonic_increasing/decreasing: ilgili sütun gözlenen aralıkta
        artan değerlerle taranır, diğerleri sabit tutulur, çıktı o yönde
        monoton mu (küçük sayısal tolerans ile)?
    """
    if not constraints:
        return True, {"reason": "kısıt verilmedi, atlandı"}

    model = candidate.metadata.get("model")
    if model is None:
        return False, {"reason": "model yok"}

    means = X_domain.mean(axis=0)
    lo, hi = X_domain.min(axis=0), X_domain.max(axis=0)

    details: Dict[str, Any] = {}
    all_passed = True

    for c in constraints:
        base = means.copy()
        if c.fixed_others:
            for idx, val in c.fixed_others.items():
                base[idx] = val

        key = f"feat{c.feature_index}_{c.behavior}"
        try:
            if c.behavior == "finite_at_zero":
                row = base.copy()
                row[c.feature_index] = lo[c.feature_index] * c.zero_fraction
                y = model.predict_symbolic(row.reshape(1, -1))
                ok = bool(np.all(np.isfinite(y)))

            elif c.behavior == "finite_at_inf":
                row = base.copy()
                row[c.feature_index] = hi[c.feature_index] * c.inf_multiplier
                y = model.predict_symbolic(row.reshape(1, -1))
                ok = bool(np.all(np.isfinite(y)))

            elif c.behavior in ("monotonic_increasing", "monotonic_decreasing"):
                xs = np.linspace(lo[c.feature_index], hi[c.feature_index],
                                  DEFAULT_MONOTONIC_SAMPLES)
                rows = np.tile(base, (len(xs), 1))
                rows[:, c.feature_index] = xs
                ys = model.predict_symbolic(rows)
                if not np.all(np.isfinite(ys)):
                    ok = False
                else:
                    diffs = np.diff(ys)
                    ok = (np.all(diffs >= -1e-9) if c.behavior == "monotonic_increasing"
                          else np.all(diffs <= 1e-9))
            else:
                details[key] = {"passed": False, "reason": f"bilinmeyen davranış: {c.behavior}"}
                all_passed = False
                continue

            details[key] = {"passed": ok}
            all_passed = all_passed and ok
        except Exception as exc:
            details[key] = {"passed": False, "reason": str(exc)}
            all_passed = False

    return all_passed, details


# ─── Test: Simetri (opsiyonel, None-safe) ────────────────────────────────────

@dataclass
class SymmetryConstraint:
    """Bir simetri kısıtı — adayın SI-uzayı çıktısının belirli bir dönüşüm
    altında beklenen davranışı sağladığını numerik doğrular.

    kind           : 'even'  → f(x) = f(-x)
                      'odd'   → f(x) = -f(-x)
                      'scale' → f(a·x) = a^k · f(x)  (ölçek/homojenlik simetrisi;
                                ör. Stokes kuvveti F=6πηrv, η'de derece-1
                                homojendir: F(a·η,r,v)=a·F(η,r,v), k=1)
    feature_index  : test edilecek sütun
    fixed_others   : {sütun_indeksi: değer} — diğer sütunlar bu değerlerde
                      sabitlenir; verilmezse X_domain'in sütun ortalamaları
                      kullanılır
    n_samples      : kaç nokta test edilecek
    scale_factor   : 'scale' testi için çarpan a (varsayılan 2.0)
    scale_exponent : 'scale' testi için beklenen üs k — ZORUNLU ('scale' için)
    tol            : bağıl tolerans
    """
    kind: str
    feature_index: int
    fixed_others: Optional[Dict[int, float]] = None
    n_samples: int = DEFAULT_SYMMETRY_SAMPLES
    scale_factor: float = 2.0
    scale_exponent: Optional[float] = None
    tol: float = DEFAULT_SYMMETRY_TOL


def test_symmetry(
    candidate: Candidate,
    X_domain: np.ndarray,
    constraints: Optional[List[SymmetryConstraint]],
) -> Tuple[bool, Dict[str, Any]]:
    """constraints=None (veya boş liste) ise test ATLANIR ve geçti sayılır
    (None-safe — Faz 2 davranışıyla geriye uyumluluk)."""
    if not constraints:
        return True, {"reason": "kısıt verilmedi, atlandı"}

    model = candidate.metadata.get("model")
    if model is None:
        return False, {"reason": "model yok"}

    means = X_domain.mean(axis=0)
    lo, hi = X_domain.min(axis=0), X_domain.max(axis=0)

    details: Dict[str, Any] = {}
    all_passed = True

    for c in constraints:
        base = means.copy()
        if c.fixed_others:
            for idx, val in c.fixed_others.items():
                base[idx] = val

        key = f"feat{c.feature_index}_{c.kind}"
        try:
            xs = np.linspace(lo[c.feature_index], hi[c.feature_index], c.n_samples)

            if c.kind in ("even", "odd"):
                xs = xs[np.abs(xs) > 1e-12]                     # x=0 tekilliğini atla
                rows_pos = np.tile(base, (len(xs), 1)); rows_pos[:, c.feature_index] = xs
                rows_neg = np.tile(base, (len(xs), 1)); rows_neg[:, c.feature_index] = -xs
                y_pos = model.predict_symbolic(rows_pos)
                y_neg = model.predict_symbolic(rows_neg)
                if not (np.all(np.isfinite(y_pos)) and np.all(np.isfinite(y_neg))):
                    ok = False
                else:
                    expected = y_pos if c.kind == "even" else -y_pos
                    denom = np.maximum(np.abs(expected), 1e-12)
                    ok = bool(np.all(np.abs(y_neg - expected) / denom <= c.tol))

            elif c.kind == "scale":
                if c.scale_exponent is None:
                    details[key] = {"passed": False, "reason": "scale_exponent verilmedi"}
                    all_passed = False
                    continue
                rows_base = np.tile(base, (len(xs), 1)); rows_base[:, c.feature_index] = xs
                rows_scaled = rows_base.copy()
                rows_scaled[:, c.feature_index] = xs * c.scale_factor
                y_base = model.predict_symbolic(rows_base)
                y_scaled = model.predict_symbolic(rows_scaled)
                if not (np.all(np.isfinite(y_base)) and np.all(np.isfinite(y_scaled))):
                    ok = False
                else:
                    expected = y_base * (c.scale_factor ** c.scale_exponent)
                    denom = np.maximum(np.abs(expected), 1e-12)
                    ok = bool(np.all(np.abs(y_scaled - expected) / denom <= c.tol))
            else:
                details[key] = {"passed": False, "reason": f"bilinmeyen kind: {c.kind}"}
                all_passed = False
                continue

            details[key] = {"passed": ok}
            all_passed = all_passed and ok
        except Exception as exc:
            details[key] = {"passed": False, "reason": str(exc)}
            all_passed = False

    return all_passed, details


# ─── Test: Korunum / toplam (opsiyonel, None-safe) ───────────────────────────

@dataclass
class ConservationConstraint:
    """Bir korunum/toplam kısıtı — adayın SI-uzayı çıktısının bir eksen
    üzerindeki integralinin (trapz) sabit kaldığını ya da bilinen bir değere
    eşit olduğunu numerik doğrular. İKİ moddan biri kullanılır:

      (a) expected_value verilir              → mutlak korunum testi
      (b) invariance_feature_index verilir    → bağımsızlık/korunum testi:
          integral, o sütunun gözlenen min VE maks değerlerinde hesaplanır;
          ikisi arasındaki bağıl fark tol'u aşarsa reddedilir (yani integral
          bu değişkenden BAĞIMSIZ olmalı demektir — ör. ∫B(λ,T)dλ'nın T'den
          bağımsız bir SABİT olması BEKLENMEZ ama bir normalizasyon
          büyüklüğünün belirli bir eksenden bağımsız kalması gerektiği
          durumlarda kullanışlıdır).

    integrate_over    : integrali alınacak sütun indeksi
    integration_range : (low, high); None ise X_domain'in o sütundaki gözlenen
                         min-max'ı kullanılır
    fixed_others      : diğer sütunlar için sabit değerler (None ise ortalama)
    n_points          : integral için trapz örnek sayısı
    expected_value    : biliniyorsa beklenen integral değeri
    invariance_feature_index : verilirse (a) yerine (b) modu kullanılır
    tol               : bağıl tolerans
    """
    integrate_over: int
    integration_range: Optional[Tuple[float, float]] = None
    fixed_others: Optional[Dict[int, float]] = None
    n_points: int = DEFAULT_CONSERVATION_POINTS
    expected_value: Optional[float] = None
    invariance_feature_index: Optional[int] = None
    tol: float = DEFAULT_CONSERVATION_TOL


_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.x: trapz -> trapezoid


def _trapz_integral(model, base_row: np.ndarray, integrate_over: int,
                     rng_lo: float, rng_hi: float, n_points: int) -> Optional[float]:
    xs = np.linspace(rng_lo, rng_hi, n_points)
    rows = np.tile(base_row, (n_points, 1))
    rows[:, integrate_over] = xs
    ys = model.predict_symbolic(rows)
    if not np.all(np.isfinite(ys)):
        return None
    return float(_trapezoid(ys, xs))


def test_conservation(
    candidate: Candidate,
    X_domain: np.ndarray,
    constraints: Optional[List[ConservationConstraint]],
) -> Tuple[bool, Dict[str, Any]]:
    """constraints=None (veya boş liste) ise test ATLANIR ve geçti sayılır
    (None-safe — Faz 2 davranışıyla geriye uyumluluk)."""
    if not constraints:
        return True, {"reason": "kısıt verilmedi, atlandı"}

    model = candidate.metadata.get("model")
    if model is None:
        return False, {"reason": "model yok"}

    means = X_domain.mean(axis=0)
    lo, hi = X_domain.min(axis=0), X_domain.max(axis=0)

    details: Dict[str, Any] = {}
    all_passed = True

    for c in constraints:
        base = means.copy()
        if c.fixed_others:
            for idx, val in c.fixed_others.items():
                base[idx] = val

        rng_lo, rng_hi = c.integration_range or (
            float(lo[c.integrate_over]), float(hi[c.integrate_over])
        )
        key = f"integral_feat{c.integrate_over}"
        try:
            integral = _trapz_integral(model, base, c.integrate_over, rng_lo, rng_hi, c.n_points)
            if integral is None:
                details[key] = {"passed": False, "reason": "integral hesaplanırken NaN/Inf"}
                all_passed = False
                continue

            if c.expected_value is not None:
                denom = max(abs(c.expected_value), 1e-12)
                ok = abs(integral - c.expected_value) / denom <= c.tol
                details[key] = {"passed": ok, "integral": integral, "expected": c.expected_value}

            elif c.invariance_feature_index is not None:
                idx = c.invariance_feature_index
                base_lo = base.copy(); base_lo[idx] = lo[idx]
                base_hi = base.copy(); base_hi[idx] = hi[idx]
                integral_lo = _trapz_integral(model, base_lo, c.integrate_over, rng_lo, rng_hi, c.n_points)
                integral_hi = _trapz_integral(model, base_hi, c.integrate_over, rng_lo, rng_hi, c.n_points)
                if integral_lo is None or integral_hi is None:
                    ok = False
                else:
                    denom = max(abs(integral_lo), 1e-12)
                    ok = abs(integral_hi - integral_lo) / denom <= c.tol
                details[key] = {"passed": ok, "integral_lo": integral_lo, "integral_hi": integral_hi}

            else:
                details[key] = {"passed": False,
                                 "reason": "expected_value veya invariance_feature_index gerekli"}
                ok = False

            all_passed = all_passed and details[key]["passed"]
        except Exception as exc:
            details[key] = {"passed": False, "reason": str(exc)}
            all_passed = False

    return all_passed, details


# ─── Ana kabul akışı — score + tüm testleri TEK yargıya birleştirir ──────────

def evaluate(
    candidate: Candidate,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_domain: Optional[np.ndarray] = None,
    r2_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    limit_constraints: Optional[List[LimitConstraint]] = None,
    symmetry_constraints: Optional[List[SymmetryConstraint]] = None,
    conservation_constraints: Optional[List[ConservationConstraint]] = None,
    check_finiteness: bool = True,
    n_finiteness_samples: int = DEFAULT_FINITENESS_SAMPLES,
    rng: Optional[np.random.Generator] = None,
) -> bool:
    """candidate.fitness'ı hesaplar (score()) VE tüm aktif testleri çalıştırıp
    tek bir kabul/red kararı döndürür. Her testin sonucu şeffaflık için
    candidate.metadata['judge_report']'a yazılır.

    X_domain verilmezse X_val kullanılır (sonluluk/limit/simetri/korunum
    testleri için örnekleme alanı — genelde daha geniş olan X_train
    verilmesi önerilir).

    symmetry_constraints / conservation_constraints verilmezse (None,
    varsayılan) bu iki test sessizce atlanır — evaluate()'in davranışı bu
    parametreler eklenmeden önceki (Faz 2) haliyle BİREBİR AYNI kalır.
    """
    domain = X_domain if X_domain is not None else X_val

    candidate.fitness = score(candidate, X_val, y_val)
    report: Dict[str, Any] = {"r2": candidate.fitness, "r2_threshold": r2_threshold}

    r2_ok = candidate.fitness >= r2_threshold
    report["r2_passed"] = r2_ok

    if check_finiteness:
        fin_ok, fin_detail = test_finiteness(candidate, domain, n_finiteness_samples, rng)
    else:
        fin_ok, fin_detail = True, {"reason": "devre dışı"}
    report["finiteness"] = {"passed": fin_ok, **fin_detail}

    limit_ok, limit_detail = test_limit_behavior(candidate, domain, limit_constraints)
    report["limit_behavior"] = {"passed": limit_ok, **limit_detail}

    sym_ok, sym_detail = test_symmetry(candidate, domain, symmetry_constraints)
    report["symmetry"] = {"passed": sym_ok, **sym_detail}

    cons_ok, cons_detail = test_conservation(candidate, domain, conservation_constraints)
    report["conservation"] = {"passed": cons_ok, **cons_detail}

    accepted = r2_ok and fin_ok and limit_ok and sym_ok and cons_ok
    report["accepted"] = accepted
    candidate.metadata["judge_report"] = report
    return accepted
