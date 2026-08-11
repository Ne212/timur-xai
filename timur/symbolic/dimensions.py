# timur/symbolic/dimensions.py

import numpy as np
import sympy as sp

class DimensionalAnalyzer:
    """
    Sıfırıncı Bakış Açısı: Buckingham Pi Teoremi Motoru
    Ham veriyi boyutsuz Pi gruplarına dönüştürür.
    """
    def __init__(self, feature_dims, target_dim, constant_dims=None):
        self.base_units = ['m', 'kg', 's', 'A', 'K', 'mol', 'cd']
        self.feature_dims = feature_dims
        self.target_dim = target_dim
        self.constant_dims = constant_dims or []
        
        self.all_dims = self.feature_dims + self.constant_dims + [self.target_dim]
        self.dim_matrix = self._build_dimensional_matrix()
        self.pi_exponents = self._find_null_space()

    def _build_dimensional_matrix(self):
        matrix = np.zeros((len(self.base_units), len(self.all_dims)))
        for col_idx, dim_dict in enumerate(self.all_dims):
            for row_idx, unit in enumerate(self.base_units):
                matrix[row_idx, col_idx] = dim_dict.get(unit, 0)
        return matrix

    def _find_null_space(self):
        # SVD yerine SymPy RREF kullanarak hedef değişkenini (y) kusursuzca izole eder.
        # Sıfır uzayı vektörleri SymPy tarafından tam rasyonel olarak döner;
        # float'a çevirmeden önce en küçük ortak payda ile tamsayıya dönüştürülür
        # (birim analizi doğruluğu için).
        M = sp.Matrix(self.dim_matrix)
        ns = M.nullspace()

        pi_vectors = []
        for v in ns:
            # Vektörün elemanlarını rasyonel sayı olarak al ve ortak paydayı bul
            rationals = [sp.Rational(x).limit_denominator(100) for x in v]
            lcm_denom = 1
            for r in rationals:
                lcm_denom = sp.lcm(lcm_denom, r.q)
            # Tamsayıya ölçekle, ardından float'a çevir
            scaled = [float(r * lcm_denom) for r in rationals]
            pi_vectors.append(np.array(scaled, dtype=float))

        pi_matrix = np.column_stack(pi_vectors)

        # Hedef değişkenin (son satır) bulunduğu Pi grubunu bul ve
        # hedefin üssünü tam olarak 1.0 yapacak şekilde normalize et.
        # Bu, formülün y = ... şeklinde çıkmasını garanti eder.
        target_row_idx = len(self.all_dims) - 1
        for j in range(pi_matrix.shape[1]):
            if abs(pi_matrix[target_row_idx, j]) > 1e-5:
                pi_matrix[:, j] /= pi_matrix[target_row_idx, j]

        # Kayan nokta gürültüsünü temizle: çok küçük değerleri sıfırla,
        # integere yakın değerleri yuvarlayarak birim analizini dengele
        pi_matrix = np.where(np.abs(pi_matrix) < 1e-10, 0.0, pi_matrix)
        rounded   = np.round(pi_matrix)
        close_int = np.abs(pi_matrix - rounded) < 1e-8
        pi_matrix = np.where(close_int, rounded, pi_matrix)

        return pi_matrix

    def transform_to_pi(self, X, y, constants_dict, feature_names=None):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_vars = len(self.all_dims)
        
        V = np.zeros((n_samples, n_vars))
        V[:, :n_features] = X
        
        const_vals = list(constants_dict.values())
        for i, val in enumerate(const_vals):
            V[:, n_features + i] = val
            
        V[:, -1] = y
        
        n_pi_groups = self.pi_exponents.shape[1]
        pi_matrix = np.ones((n_samples, n_pi_groups))
        target_row_idx = n_vars - 1
        
        for j in range(n_pi_groups):
            for i in range(n_vars):
                power = self.pi_exponents[i, j]
                if abs(power) > 1e-5:
                    # Eğer bu hedef değişkense ve üssü 1 ise, gürültülü işaretleri (negatif) koru
                    if i == target_row_idx and abs(power - 1.0) < 1e-5:
                        pi_matrix[:, j] *= V[:, i]
                    else:
                        # Diğer fiziksel değişkenler (veya sabitler) için negatif gürültüden 
                        # kaynaklanacak NaN hatalarını önlemek adına mutlak değer al
                        pi_matrix[:, j] *= np.power(np.abs(V[:, i]), power)
                        
        target_pi_idx = -1
        feature_pi_indices = []
        
        for j in range(n_pi_groups):
            if abs(self.pi_exponents[target_row_idx, j]) > 1e-5:
                target_pi_idx = j
            else:
                feature_pi_indices.append(j)
                
        if target_pi_idx == -1:
             raise ValueError("Hedef değişkeni (y) boyutsuzlaştıran bir Pi grubu bulunamadı.")
             
        X_pi = pi_matrix[:, feature_pi_indices]
        y_pi = pi_matrix[:, target_pi_idx]
        
        # İki isim seti üret: display (insan-okunabilir) + pysr (alphanumeric-safe)
        all_var_names = self._build_var_names(n_features, constants_dict, feature_names_override=feature_names)
        display_names, pysr_names = self._make_pi_names(feature_pi_indices, all_var_names)
        
        # İsim setlerini sonraki erişim için instance'ta sakla
        self._pysr_feature_names     = pysr_names
        self._display_feature_names  = display_names

        return X_pi, y_pi, display_names

    # ── İsim yardımcıları ────────────────────────────────────────────────────

    def _build_var_names(self, n_features, constants_dict, feature_names_override=None):
        """
        Tüm değişkenlerin (özellikler + sabitler + hedef) isim listesini döndürür.
        feature_names_override geçilirse özellik isimleri oradan alınır;
        aksi hâlde x0, x1, ... kullanılır.
        """
        if feature_names_override is not None:
            feat_names = list(feature_names_override)
        else:
            feat_names = [f"x{i}" for i in range(n_features)]
        
        const_names = list(constants_dict.keys()) if constants_dict else []
        return feat_names + const_names + ["y"]

    def _make_pi_names(self, pi_col_indices, all_var_names):
        """
        pi_exponents matrisinin belirtilen sütunları için iki isim seti döndürür:
          - display_names : İnsan-okunabilir, Unicode üst simgeli  → rapor/grafik
          - pysr_names    : PySR uyumlu, sadece [A-Za-z0-9_]       → sembolik keşif

        Display üs formatı:
            1.0   →  var
            -1.0  →  var⁻¹
            2.0   →  var²
            0.5   →  var^0.5

        PySR üs formatı:
            1.0   →  var
            -1.0  →  var_n1   (neg = n, sayı devam)
            2.0   →  var_p2
            0.5   →  var_p0d5 (nokta → d)
        """
        superscripts = {
            2: "²", 3: "³", 4: "⁴", -1: "⁻¹", -2: "⁻²", -3: "⁻³", -4: "⁻⁴",
        }

        def fmt_display(name, p):
            p_round = round(p, 6)
            p_int   = int(round(p_round))
            if abs(p_round - 1.0) < 1e-5:
                return name
            if abs(p_round - p_int) < 1e-5 and p_int in superscripts:
                return f"{name}{superscripts[p_int]}"
            return f"{name}^{p_round:.3g}"

        def fmt_pysr(name, p):
            """PySR için [A-Za-z0-9_] sınırını aşmayan üs kodu üret."""
            p_round = round(p, 6)
            p_int   = int(round(p_round))
            if abs(p_round - 1.0) < 1e-5:
                return name
            if abs(p_round - p_int) < 1e-5:
                sign = "n" if p_int < 0 else "p"
                return f"{name}_{sign}{abs(p_int)}"
            # Kesirli üs: 0.5 → p0d5, -0.5 → n0d5
            p_str = f"{abs(p_round):.3g}".replace(".", "d")
            sign = "n" if p_round < 0 else "p"
            return f"{name}_{sign}{p_str}"

        display_names = []
        pysr_names    = []

        for i, col_idx in enumerate(pi_col_indices):
            display_parts = []
            pysr_parts    = []
            for var_idx, var_name in enumerate(all_var_names[:-1]):  # hedefi (y) dışla
                p = self.pi_exponents[var_idx, col_idx]
                if abs(p) > 1e-5:
                    display_parts.append(fmt_display(var_name, p))
                    pysr_parts.append(fmt_pysr(var_name, p))

            if display_parts:
                display_names.append("Pi(" + "·".join(display_parts) + ")")
                pysr_names.append("Pi_" + "_".join(pysr_parts))
            else:
                label = f"Pi_{i+1}"
                display_names.append(label)
                pysr_names.append(label)

        return display_names, pysr_names
    
    def inverse_transform_target(self, X, y_pi, constants_dict):
        """Tahmin edilen boyutsuz y_pi değerini, orijinal fiziksel y değerine (SI) geri yansıtır."""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_vars = len(self.all_dims)

        V = np.zeros((n_samples, n_vars))
        V[:, :n_features] = X

        const_vals = list(constants_dict.values())
        for i, val in enumerate(const_vals):
            V[:, n_features + i] = val

        # Hedef Pi grubunu bul
        target_row_idx = n_vars - 1
        target_pi_idx = -1
        for j in range(self.pi_exponents.shape[1]):
            if abs(self.pi_exponents[target_row_idx, j]) > 1e-5:
                target_pi_idx = j
                break

        # Çarpanı hesapla (hedef değişken hariç diğerlerinin üssü)
        multiplier = np.ones(n_samples)
        for i in range(n_vars - 1):
            power = self.pi_exponents[i, target_pi_idx]
            if abs(power) > 1e-5:
                multiplier *= np.power(np.abs(V[:, i]), power)

        # Pi uzayından klasik SI dünyasına çıkış: y = y_pi / çarpan
        return y_pi / multiplier

    # ── Fiziksel Denklem Geri-Dönüşümü ─────────────────────────────────────────

    def get_physical_equation(self, pysr_eq_str, feature_names, constants_dict):
        """
        PySR'ın Pi-grup uzayında bulduğu denklemi fiziksel değişkenlere dönüştürür.

        Adım 1: Her Pi grubunu orijinal değişkenlerle (λ, T, h, c, kB) ifade eder.
        Adım 2: Boyutsal çarpanı hesaplar: y = Π_target / ∏(var^üs)
        Adım 3: SymPy ile tam ikameyi dener; başarısız olursa parametrik form döner.

        Parametreler
        -----------
        pysr_eq_str   : PySR-safe isimler içeren denklem string'i (display öncesi)
        feature_names : orijinal özellik isimleri listesi
        constants_dict: {isim: (değer, birim_dict)} sözlüğü

        Dönüş
        -----
        (sympy_expr | None, display_str)
        """
        import sympy as sp

        feat_names  = list(feature_names) if feature_names else [
            f"x{i}" for i in range(len(self.feature_dims))
        ]
        const_names = list(constants_dict.keys()) if constants_dict else []
        all_var_names = feat_names + const_names + ["y"]

        # SymPy sembolleri (pozitif varsay — fiziksel büyüklükler)
        sym_map = {name: sp.Symbol(name, positive=True) for name in all_var_names[:-1]}

        target_row_idx = len(all_var_names) - 1
        feature_pi_cols, target_pi_col = [], -1
        for j in range(self.pi_exponents.shape[1]):
            if abs(self.pi_exponents[target_row_idx, j]) > 1e-5:
                target_pi_col = j
            else:
                feature_pi_cols.append(j)

        if target_pi_col == -1:
            return None, pysr_eq_str

        pysr_names = getattr(self, '_pysr_feature_names', []) or []

        # Her özellik Pi grubunun SymPy ifadesi
        pi_sympy = {}
        pi_display_formulas = {}
        for k, (j, pname) in enumerate(zip(feature_pi_cols, pysr_names)):
            expr = sp.Integer(1)
            numer_parts, denom_parts = [], []
            for i, name in enumerate(all_var_names[:-1]):
                p = float(self.pi_exponents[i, j])
                if abs(p) < 1e-5:
                    continue
                pr = sp.Rational(p).limit_denominator(12)
                expr *= sym_map[name] ** pr
                if p > 0:
                    numer_parts.append(self._fmt_sup(name, p))
                else:
                    denom_parts.append(self._fmt_sup(name, -p))
            pi_sympy[pname] = expr
            if denom_parts:
                pi_display_formulas[pname] = (
                    ("·".join(numer_parts) if numer_parts else "1")
                    + " / (" + "·".join(denom_parts) + ")"
                )
            else:
                pi_display_formulas[pname] = "·".join(numer_parts) or "1"

        # Boyutsal çarpan: y = Π_target / ∏(var^p)
        dim_factor = sp.Integer(1)
        y_numer_parts, y_denom_parts = [], []
        for i, name in enumerate(all_var_names[:-1]):
            p = float(self.pi_exponents[i, target_pi_col])
            if abs(p) < 1e-5:
                continue
            pr = sp.Rational(p).limit_denominator(12)
            dim_factor *= sym_map[name] ** pr
            # y = Π_target / ∏(var^p) → var^p pozitifse payda, negatifse pay
            if p > 0:
                y_denom_parts.append(self._fmt_sup(name, p))
            else:
                y_numer_parts.append(self._fmt_sup(name, -p))

        if y_denom_parts:
            dim_factor_str = ("·".join(y_numer_parts) if y_numer_parts else "1") \
                             + " / (" + "·".join(y_denom_parts) + ")"
        else:
            dim_factor_str = "·".join(y_numer_parts) or "1"

        # ── SymPy ile tam ikame denemesi ────────────────────────────────────
        eq_str = pysr_eq_str[4:] if pysr_eq_str.startswith("y = ") else pysr_eq_str

        local_dict = {
            "exp": sp.exp, "log": sp.log, "sin": sp.sin,
            "cos": sp.cos, "sqrt": sp.sqrt,
            "inv": lambda x: sp.Integer(1) / x,
        }
        placeholder_map = {}
        for k, pname in enumerate(sorted(pi_sympy.keys(), key=len, reverse=True)):
            ph = sp.Symbol(f"_p{k}_", positive=True)
            placeholder_map[pname] = ph
            local_dict[f"_p{k}_"] = ph
            eq_str = eq_str.replace(pname, f"_p{k}_")

        try:
            parsed = sp.sympify(eq_str, locals=local_dict)
            for pname, ph in placeholder_map.items():
                parsed = parsed.subs(ph, pi_sympy[pname])
            y_expr = sp.simplify(parsed / dim_factor)
            # Okunabilir string: SymPy'nın varsayılan repr'ini kullan ama
            # sayısal katsayıları 6 anlamlı basamakla yuvarla
            y_str  = self._format_sympy_expr(y_expr)
            return y_expr, f"y = {y_str}"
        except Exception:
            pass

        # ── Fallback: Parametrik gösterim ───────────────────────────────────
        disp_names = getattr(self, '_display_feature_names', pysr_names) or pysr_names
        pi_def_lines = []
        for k, pname in enumerate(pysr_names):
            label = disp_names[k] if k < len(disp_names) else f"Π_{k+1}"
            pi_def_lines.append(f"  {label} = {pi_display_formulas.get(pname, '?')}")

        # PySR denklemindeki PySR isimlerini display isimleriyle değiştir
        eq_display = pysr_eq_str
        for pname, label in zip(
            sorted(pysr_names, key=len, reverse=True),
            [disp_names[i] if i < len(disp_names) else f"Π_{i+1}"
             for i in range(len(pysr_names))]
        ):
            eq_display = eq_display.replace(pname, label)

        lines = [eq_display]
        lines.append(f"  y = Π_hedef · {dim_factor_str}")
        lines += pi_def_lines
        return None, "\n  ".join(lines)

    @staticmethod
    def _format_sympy_expr(expr) -> str:
        """
        SymPy ifadesini okunabilir string'e dönüştürür.
        Float katsayıları 5 anlamlı rakamla gösterir; saf sembolik kısımlar değişmez.
        """
        import sympy as sp
        import re

        raw = str(expr)

        def _round_float(m):
            val = float(m.group(0))
            # Tam integerse int göster
            if abs(val - round(val)) < 1e-9:
                return str(int(round(val)))
            return f"{val:.5g}"

        # Float sayıları yuvarlama (ör. 2.1263325 → 2.1263, 0.60697615 → 0.60698)
        rounded = re.sub(r'\d+\.\d+(?:[eE][+-]?\d+)?', _round_float, raw)
        # Gereksiz "1*" çarpımlarını temizle (ör. "1*c*h" → "c*h")
        rounded = re.sub(r'(?<![0-9])1\*(?=[a-zA-Z_(])', '', rounded)
        return rounded

    @staticmethod
    def _fmt_sup(name: str, p: float) -> str:
        """Değişken ismi ve üssü okunabilir biçimde birleştirir. Örn: 'lambda', 5 → 'lambda⁵'"""
        sups = {2:"²",3:"³",4:"⁴",5:"⁵",6:"⁶",-1:"⁻¹",-2:"⁻²",-3:"⁻³",-4:"⁻⁴",-5:"⁻⁵"}
        p_r  = round(p, 6)
        p_i  = int(round(p_r))
        if abs(p_r - 1.0) < 1e-5:
            return name
        if abs(p_r - p_i) < 1e-5 and p_i in sups:
            return f"{name}{sups[p_i]}"
        return f"{name}^{p_r:.3g}"