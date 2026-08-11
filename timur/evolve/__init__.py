"""
timur/evolve/
═════════════════════════════════════════════════════════════════════════════
TIMUR XAI — Özyinelemeli Keşif Katmanı (MAP-Elites)  [Faz 2: gerçek yargıç + derinlik ekseni]

timur.TIMURModel.fit() → discover() tek-geçişlidir (bkz. timur/symbolic/router.py):
bir kez çalışır, bir DiscoveryResult döndürür, sonuç hiçbir yerde arşivlenmez ya da
geri beslenmez. Bu paket, çekirdek kodu (timur/__init__.py, timur/symbolic/,
timur/pinn/) HİÇ DEĞİŞTİRMEDEN, TIMURModel'i kara-kutu bir "aday üretici" olarak
tekrar tekrar çağırıp sonuçları bir MAP-Elites arşivinde biriktiren, skorlayan ve
varyasyonla besleyen üst (dış) katmandır.

Modüller
--------
    features.py  — bir aday ifadeden (equation_str) MAP-Elites davranış
                    tanımlayıcısı (complexity × depth × operator_family) çıkarır
    archive.py   — MAP-Elites ızgarası: Candidate veri yapısı + Archive sınıfı
    judge.py     — Faz 2: R² eşiği + sonluluk testi + opsiyonel limit-davranışı
                    testleri (test_symmetry/test_conservation hâlâ Faz 3 TODO)
    mutate.py    — bir elite'ten yola çıkarak TIMURModel'i yeniden çağırıp
                    varyasyon üretir (pluggable strateji)
    loop.py      — evolve(): SEED + LOOP ana özyinelemeli döngü

Kullanım
--------
    from timur.evolve.loop import evolve
    archive = evolve(X, y, X_val, y_val, feature_dims=..., target_dim=..., ...)
"""
