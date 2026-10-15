# AI/ML Python məşqləri

scikit-learn ilə ML əsaslarını addım-addım öyrəndiyim layihə.

## Qurulum

    pip install scikit-learn pandas numpy

## Fayllar

| Fayl | Mövzu |
|------|-------|
| `overfitting.py` | `max_depth` dəyişdikcə train/test fərqi (overfitting) |
| `grid_search.py` | GridSearchCV ilə hiperparametr seçimi |
| `random_forest.py` | Random Forest və feature importance |
| `confusion_matrix.py` | Confusion matrix, precision, recall |
| `threshold.py` | Qərar həddinin recall/precision balansına təsiri |
| `scaling.py` | StandardScaler və Pipeline (KNN) |
| `grid_pipeline.py` | Pipeline + GridSearchCV |
| `compare_models.py` | 5 modelin eyni CV bölgüsündə müqayisəsi |
| `real_data.py` | Titanic: boş qiymətlər, kateqoriyalı sütunlar, baseline |
| `feature_engineering.py` | `family_size` və boşluq göstəricisi (fayda vermədi) |
| `title_features.py` | Addan titul çıxarmaq (Logistic Regression üçün fayda verdi) |
| `repeated_cv.py` | Təkrarlanan CV ilə title təsirinin yoxlanması |

## Əsas müşahidələr

- Tək train/test bölgüsünə güvənmək olmaz. Cross-validation və `±` ilə baxmaq lazımdır.
- Hiperparametrlər train hissəsində seçilməlidir. Test hissəsinə yalnız sonda baxılmalıdır.
- Breast cancer: miqyaslandıqdan sonra SVM və Logistic Regression ən yaxşı nəticəni verdi (~0.975), ağac isə sonuncu oldu (~0.92).
- Tibbi məsələdə accuracy kifayət deyil. Recall və qərar həddi seçimi vacibdir.
- Titanic: baseline 0.618, modellər ~0.79-0.80. Addan çıxarılan `title` xüsusiyyəti Logistic Regression-u ~0.786-dan ~0.807-yə qaldırdı (50 CV bölgüsünün 45-də daha yaxşı).
- Hər xüsusiyyət kömək etmir: `family_size` və `is_alone` nəticəni yaxşılaşdırmadı.
