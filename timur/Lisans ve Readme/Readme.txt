# TIMUR XAI
*Teorik İlkelerle Makine Usulü Regresyon*

TIMUR, kara kutu sinir ağlarını (Black Box) parçalayıp, verinin arkasındaki matematiksel ve fiziksel gerçekliği sembolik denklemlerle çıkaran ve ağı bu denklemlere sabitleyen bir PINN (Fiziksel Bilgili Sinir Ağı) regülarizasyon çerçevesidir.

## Kurulum
\`\`\`bash
pip install timur-xai
\`\`\`

## Kullanım
\`\`\`python
from timur import TIMURModel
model = TIMURModel(lambda_sym=0.1)
model.fit(X_train, y_train)
print(model.equation)
\`\`\`

## ⚖️ Lisanslama ve Ticari Kullanım
Bu proje **Çift Lisanslıdır (Dual-Licensed)**.
* Akademik ve açık kaynaklı projeler için **GPLv3** lisansı altında ücretsizdir.
* TIMUR XAI'yi kapalı kaynaklı kurumsal projelerde, ticari altyapılarda veya şirket içi üretim (production) ortamlarında kullanmak için **Ticari Lisans** satın alınması zorunludur. İletişim: iletisim@eposta.com