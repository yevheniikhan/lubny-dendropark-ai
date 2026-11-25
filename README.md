# Дендропарк коледжу очима супутника і штучного інтелекту

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yevheniikhan/lubny-dendropark-ai/blob/main/notebooks/dendropark_ai_gis.ipynb)

Профорієнтаційний проєкт НУБіП України для Лубенського лісотехнічного фахового коледжу: як супутникові дані, ГІС і штучний інтелект описують дендропарк коледжу імені В. Д. Байтали.

Автор: Євгеній Хань, Національний університет біоресурсів і природокористування України.

## Що всередині

| Крок | Що робимо | Дані |
|---|---|---|
| 1 | Межа парку: картосхема з буклета коледжу, прив'язана до знімка | Esri World Imagery, картосхема коледжу |
| 2 | Висота крон і 3D-модель | Meta і WRI, Global Canopy Height 1 m |
| 3 | Сезонна динаміка NDVI | Copernicus Sentinel-2 L2A, маска хмар Cloud Score+ |
| 4 | Зони парку, знайдені ШІ без підказок | AlphaEarth Foundations Satellite Embedding, k-середніх |
| 5 | Окремі крони (за бажанням, потрібен GPU) | SAM / samgeo |

Ноутбук: [`notebooks/dendropark_ai_gis.ipynb`](notebooks/dendropark_ai_gis.ipynb). Потрібен безкоштовний проєкт Google Earth Engine: у першій клітинці замініть `your-ee-project` на свій ідентифікатор.

## Межа парку

Картосхему з буклета «Дендропарк імені В. Д. Байтали. 60 років» прив'язано до знімка Esri сплайном тонкої пластини через 17 опорних точок (межа лісового масиву, дорога, центральна галявина; `data/lltk/gcp.json`). Отримана межа – 30,4 га (за буклетом площа парку 32 га). Вона використана в усіх розрахунках (`data/park_boundary.json`, EPSG:32636).

## Результати (власний розрахунок)

* Деревами, вищими за 5 м, вкрито 69 % площі; медіанна висота крон дерев – 15 м, 90-й процентиль – 19 м (сітка 5 м).
* Середній NDVI 2024 року: 0,16 у січні, пік 0,85 у червні; у лютому і грудні безхмарних знімків не було.
* Кластеризація ембедингів AlphaEarth після згладжування виділила чотири типи ділянок: високі насадження (12,0 га), насадження середньої висоти (10,9 га), розсадник і відкриті ділянки (4,9 га), узлісся вздовж забудови (2,6 га). Назви зонам дано за висотою крон.

## Структура

* `notebooks/` – ноутбук Colab;
* `data/park2.npz` – сітки в EPSG:32636 з початком 498900 / 5537360: висота крон (5 м), місячний NDVI 2024 (20 м), зони ШІ (10 м); `data/park_stats.json`, `data/zones_area.json` – підсумки;
* `tools/boundary.py` – межа з прив'язаної картосхеми; `tools/grids.py` – читання й маскування сіток; `tools/media2.py` – анімації для презентації; `tools/make_notebook.py` – генерує ноутбук.

Знімки Esri і зображення картосхеми в репозиторії не поширюються: їх треба отримати з першоджерел.

## Навчання

Методам дистанційного моніторингу лісів, лісової таксації, лісовпорядкування й аналізу лісової інформації навчають у Навчально-науковому інституті лісового і садово-паркового господарства НУБіП України: [nubip.edu.ua](https://nubip.edu.ua).

## Джерела

* Дендропарк, буклет і картосхема: [lltk.edu.ua/pro-koledzh/dendropark](https://lltk.edu.ua/pro-koledzh/dendropark).
* Tolan J. et al. (2024). Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar. *Remote Sensing of Environment* 300, 113888.
* Pasquarella V. J. et al. (2023). Comprehensive quality assessment of optical satellite imagery using weakly supervised video learning. CVPR Workshops.
* Brown C. F. et al. (2025). AlphaEarth Foundations: an embedding field model for accurate and efficient global mapping from sparse label data. arXiv:2507.22291.
* Copernicus Sentinel-2 data (2024), ESA.

Код – MIT. Дані третіх сторін – на умовах їхніх ліцензій.
