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
        # SVD yerine SymPy RREF kullanarak hedef değişkenini (y) kusursuzca izole eder
        M = sp.Matrix(self.dim_matrix)
        ns = M.nullspace()
        
        pi_vectors = []
        for v in ns:
            pi_vectors.append(np.array(v).astype(float).flatten())
            
        pi_matrix = np.column_stack(pi_vectors)
        
        # Hedef değişkenin (son satır) bulunduğu Pi grubunu bul ve
        # hedefin üssünü tam olarak 1.0 yapacak şekilde normalize et.
        # Bu, formülün y = ... şeklinde çıkmasını garanti eder.
        target_row_idx = len(self.all_dims) - 1
        for j in range(pi_matrix.shape[1]):
            if abs(pi_matrix[target_row_idx, j]) > 1e-5:
                pi_matrix[:, j] /= pi_matrix[target_row_idx, j]
                
        return pi_matrix

    def transform_to_pi(self, X, y, constants_dict):
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
        
        # Anlamlı boyutsuz isimler üret
        # Yeni Güvenli Satır:
        pi_names = [f"Pi_{i+1}" for i in range(len(feature_pi_indices))]
        
        return X_pi, y_pi, pi_names
    
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