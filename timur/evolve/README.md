# timur/evolve/ — Özyinelemeli Keşif Katmanı (MAP-Elites)

`TIMURModel.fit()` tek-geçişlidir: `discover()` bir kez çalışır, tek bir
`DiscoveryResult` döner, sonuç hiçbir yerde arşivlenmez ya da geri beslenmez
(bkz. `timur/symbolic/router.py`). Bu paket, **çekirdek TIMUR kodunu hiç
değiştirmeden** (`timur/__init__.py`, `timur/symbolic/`, `timur/pinn/`
dokunulmaz), `TIMURModel`'i kara-kutu bir "aday üretici" olarak tekrar tekrar
çağırıp sonuçları bir **MAP-Elites arşivinde** biriktiren, skorlayan ve
varyasyonla besleyen dış bir katmandır.

## Mimari

```
                ┌─────────────────────────────────────────────┐
                │              evolve()  (loop.py)             │
                │                                               │
   SEED ────────┤  TIMURModel(**kwargs).fit(bootstrap(X,y))     │
   (N kez)      │        │                                      │
                │        ▼                                      │
   ┌──────────┐ │  ┌───────────┐   ┌────────────┐   ┌─────────┐│
   │ ÜRETİCİ  │ │  │ features  │──▶│   judge    │──▶│ archive ││
   │ (TIMUR,  │◀┼─▶│ .py       │   │ .evaluate()│   │  .add() ││
   │ kara-kutu│ │  │ descriptor│   │ R²+testler │   │MAP-Elites││
   └──────────┘ │  └───────────┘   └────────────┘   └────┬────┘│
                │                                          │     │
   LOOP ────────┤  mutate(elite, X, y, kwargs) ◀───────────┘     │
   (M kez)      │        │  (archive.sample_elites)               │
                │        └──────────────▶ (yeniden SEED gibi)     │
                └─────────────────────────────────────────────┘
```

- **Üretici = TIMUR** (`mutate.py`): `TIMURModel`'i kara-kutu olarak yeniden
  çağırır. `TIMURModel` dışarıya `random_state` açmadığı için stokastiklik
  bootstrap yeniden-örnekleme + eşik pertürbasyonuyla sağlanır; bu, PySR'ın
  kendi iç rastgeleliğiyle birleşince gerçek çeşitlilik üretir (bkz. Planck
  tanı koşuları).
- **Davranış tanımlayıcısı** (`features.py`): her adayı 3 eksenli bir ızgara
  hücresine eşler — `(complexity, depth, operator_family)`, her biri 4 bin
  → 4×4×4=64 hücre. Derinlik ekseni, cardinality yerine tercih edildi çünkü
  tek-Pi-grubu veri setlerinde (Planck gibi) cardinality matematiksel olarak
  dejenere kalıyordu.
- **Doğrulayıcı = judge** (`judge.py`): `evaluate()` beş bağımsız testi
  birleştirir — R² eşiği, sonluluk, limit/monotonluk, simetri, korunum/toplam.
  Boyutsal tutarlılık ayrı bir test DEĞİLDİR: `TIMURModel` zaten Buckingham
  Pi uzayında arama yaptığı için her aday yapısal olarak boyutsal tutarlıdır.
- **Arşiv = MAP-Elites** (`archive.py`): her hücrede sadece o davranış
  bölgesinin en iyi (en yüksek fitness) tek adayı tutulur — TIMUR'un
  "tek-geçiş → tek sonuç" davranışının yerini alan çeşitlilik-koruyan bellek.
- **Döngü** (`loop.py`): `evolve()` = SEED (bootstrap ile N ilk aday) + LOOP
  (M iterasyon: elite örnekle → mutate → judge.evaluate → kabul edilirse
  arşive ekle). Durma kriteri: `n_iterations` dolması VEYA son `patience`
  iterasyonda en iyi fitness'ta `epsilon`'dan az iyileşme (plato).

## Nasıl koşulur

```python
from timur.evolve.loop import evolve
from timur.evolve.judge import LimitConstraint, SymmetryConstraint, ConservationConstraint

archive = evolve(
    X_train, y_train, X_val, y_val,
    feature_names=[...], feature_dims=[...], target_dim={...},
    n_iterations=15, seed_runs=3,
    model_kwargs=dict(constants={...}, lambda_sym=0.5, epochs=20, ...),
    r2_threshold=0.5,
    limit_constraints=[LimitConstraint(feature_index=1, behavior="monotonic_increasing")],
    symmetry_constraints=[SymmetryConstraint(kind="scale", feature_index=0, scale_exponent=1.0)],
    conservation_constraints=None,   # opsiyonel, None-safe
    save_path="archive.json",
)
print(archive.best().expr_str, archive.coverage())
```

Uçtan uca çalışan örnekler: `evolve_demo.py` (Stefan-Boltzmann, hızlı/saf-sabit
yol), `evolve_demo_planck.py` (Planck, gerçek PySR araması, tanı amaçlı
izlemeli), `evolve_benchmark.py` (beş yasanın tamamı).

## Hangi kısıtlar verilebilir (hepsi opsiyonel, None-safe)

| Parametre | Ne test eder | Örnek |
|---|---|---|
| `r2_threshold` | Validation R² eşiği (varsayılan 0.5) | `r2_threshold=0.7` |
| `check_finiteness` | Girdi alanı örnekleminde NaN/Inf var mı | `True`/`False` |
| `LimitConstraint` | `finite_at_zero` / `finite_at_inf` / `monotonic_increasing` / `monotonic_decreasing` | sıcaklık artınca ışıma artmalı |
| `SymmetryConstraint` | `even` / `odd` / `scale` (f(a·x)=a^k·f(x)) | Stokes kuvveti viskozitede derece-1 homojen |
| `ConservationConstraint` | integralin (trapz) sabit bir değere eşitliği YA DA bir eksenden bağımsızlığı | ∫f dx = beklenen değer |

Kısıt verilmezse ilgili test sessizce atlanır ve geçti sayılır — hiçbir
parametre vermeden `evaluate()` çağırmak yalnızca R² eşiği + sonluluk testini
uygular.

## Bilinen sınırlamalar (dürüstçe)

- `TIMURModel` `random_state` açmadığı için mutasyon çeşitliliği veri
  pertürbasyonuna dayanır — doğrudan parametre-uzayı araması değildir.
- Buckingham Pi analizi bir veri setini tam olarak belirlerse (`in_features=0`,
  "saf-sabit" yol), PySR hiç çalışmaz ve MAP-Elites arşivi tek hücrede kalır.
  Bu bir kusur değil, o fizik yasasının Pi teoremi tarafından tamamen
  çözülmüş olmasının doğrudan sonucudur (Stefan-Boltzmann, Stokes, Wien gibi
  tek-Pi-gruplu yasalar).
- `test_symmetry`/`test_conservation` numerik örneklemeye dayanır (sembolik
  kanıt değil) — büyük toleranslarla (`tol`) yanlış-pozitif, küçük
  toleranslarla yanlış-red riski taşır; varsayılan değerler ampirik olarak
  makul seçilmiştir, üretim kullanımında veri setine göre ayarlanmalıdır.
