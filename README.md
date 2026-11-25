# Дендропарк, який видно з космосу

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yevheniikhan/lubny-dendropark-ai/blob/main/notebooks/dendropark_ai_gis.ipynb)

Навчальний проєкт для студентів Лубенського лісотехнічного фахового коледжу: що відкриті супутникові дані і штучний інтелект розповідають про дендропарк коледжу імені В. Д. Байтали.

У 1952 році під час обговорення проєкту парку хтось заперечив, що зірку в його центрі «може бути видно лише з літака». Сьогодні її бачить супутник, а дані аналізує ШІ.

## Що всередині

| Крок | Що робимо | Дані |
|---|---|---|
| 1 | Карта парку | Esri World Imagery (лише перегляд) |
| 2 | Висота крон і 3D-модель | Meta і WRI, Global Canopy Height 1 m |
| 3 | Як парк «дихає» протягом року (NDVI) | Copernicus Sentinel-2 L2A, маска хмар Cloud Score+ |
| 4 | Зони парку, знайдені ШІ без підказок | AlphaEarth Foundations Satellite Embedding, k-середніх |
| 5 | Окремі крони (за бажанням, потрібен GPU) | SAM / samgeo |

Ноутбук: [`notebooks/dendropark_ai_gis.ipynb`](notebooks/dendropark_ai_gis.ipynb). Потрібні обліковий запис Google і безкоштовний проєкт Google Earth Engine: у першій клітинці замініть `your-ee-project` на ідентифікатор свого проєкту.

## Перші результати (центральна частина парку, ≈19,5 га)

* Медіанна висота крон – 15 м, 90-й процентиль – 19 м; кронами понад 10 м вкрито 71 % площі (сітка 5 м).
* Середній NDVI 2024 року: 0,18 у січні, пік 0,89 у червні; у лютому і грудні безхмарних знімків не було.
* Кластеризація ембедингів AlphaEarth виділила 5 зон: відкриті ділянки, молоді й розріджені насадження, середньовисокі та два типи високих насаджень, які ще треба розпізнати в полі.

Це власні розрахунки за наведеними відкритими даними. Межа аналізу – ручна оцифровка центральної частини парку (≈19,5 га). За буклетом коледжу площа парку – 32 га (у 1952 році виділили 35 га); дендрофлора налічує близько 150 видів і форм, близько 13 тис. дерев і кущів. У центрі – п'ятикутна зірка з екзотами, навколо 14 кварталів лісових культур.

## Завдання для студентів

1. **Тип А чи тип Б?** Визначте породи у двох зонах високих насаджень, які розрізнив ШІ, і поясніть різницю.
2. **Далекомір проти супутника.** Виміряйте висоту 20 дерев і порівняйте з картою висот крон.
3. **Шукаємо зірку.** Прив'яжіть картосхему парку з буклета коледжу до знімка (QGIS, Georeferencer) і знайдіть промені зірки та межі 14 кварталів на карті висот і NDVI.
4. **Календар листя.** Порівняйте, коли розпускаються екзоти зірки і дуб звичайний.

ШІ рахує, а висновки робить людина: кожен результат треба перевірити в полі.

## Структура

* `notebooks/` – ноутбук Colab;
* `data/park.npz` – сітки в EPSG:32636: висота крон (5 м), місячний NDVI 2024 (20 м), зони ШІ (10 м);
* `tools/make_notebook.py` – генерує ноутбук; `tools/decode.py` – розпаковує експорт сіток; `tools/media.py` – анімації для презентації (знімки Esri треба отримати окремо, вони не поширюються в цьому репозиторії).

## Джерела

* Історія дендропарку, буклет «Дендропарк імені В. Д. Байтали. 60 років» і картосхема: [lltk.edu.ua/pro-koledzh/dendropark](https://lltk.edu.ua/pro-koledzh/dendropark).
* Tolan J. et al. (2024). Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar. *Remote Sensing of Environment* 300, 113888.
* Pasquarella V. J. et al. (2023). Comprehensive quality assessment of optical satellite imagery using weakly supervised video learning. CVPR Workshops (Cloud Score+).
* Brown C. F. et al. (2025). AlphaEarth Foundations: an embedding field model for accurate and efficient global mapping from sparse label data. arXiv:2507.22291.
* Copernicus Sentinel-2 data (2024), ESA.

Код – MIT. Дані третіх сторін – на умовах їхніх ліцензій.
