# Дендропарк коледжу очима супутника і штучного інтелекту

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yevheniikhan/lubny-dendropark-ai/blob/main/notebooks/dendropark_ai_gis.ipynb)

Профорієнтаційний проєкт НУБіП України для Лубенського лісотехнічного фахового коледжу: як супутникові дані, ГІС і штучний інтелект описують дендропарк коледжу імені В. Д. Байтали.

Автор: Євгеній Хань, Національний університет біоресурсів і природокористування України.

## Що всередині

| Крок | Що робимо | Дані |
|---|---|---|
| 1 | Межа парку: 68 кварталів, оцифровані зі схеми й прив'язані до знімка | Esri World Imagery, схема кварталів коледжу |
| 2 | Висота крон і 3D-модель | Meta і WRI, Global Canopy Height 1 m |
| 3 | Сезонна динаміка NDVI | Copernicus Sentinel-2 L2A, маска хмар Cloud Score+ |
| 4 | Зони парку, знайдені ШІ без підказок | AlphaEarth Foundations Satellite Embedding, k-середніх |
| 5 | Окремі крони (за бажанням, потрібен GPU) | SAM / samgeo |

Ноутбук: [`notebooks/dendropark_ai_gis.ipynb`](notebooks/dendropark_ai_gis.ipynb). Потрібен безкоштовний проєкт Google Earth Engine: у першій клітинці замініть `your-ee-project` на свій ідентифікатор.

## Межа парку і квартали

Схему 68 кварталів дендропарку оцифровано у векторні полігони зі спільними межами (`tools/quarters_digitize.py`): номери кварталів правлять за зерна, межі виділено вододілом по лініях схеми, контури спрощено зі збереженням спільних ребер. Основну частину парку (квартали 1–56) прив'язано до знімка Esri афінним перетворенням за 12 опорними точками (`data/lltk/gcp.json`), тож прямі лінії лишаються прямими. Південну смугу кварталів 57–68 на схемі намальовано не в масштабі, тому її прив'язано окремо проєктивним перетворенням за 4 точками на межі лісу (`data/lltk/q/gcp_tail.json`) і обрізано по межі основної частини (`tools/quarters_georef.py`). Зовнішню межу парку проведено прямими відрізками по краю парку на знімку (`tools/user_boundary.py`), а крайні квартали доведено до неї: кожну непокриту ділянку віддано найближчому кварталу, тож внутрішні лінії просто продовжуються до межі.

Результат: `data/park_quarters_utm36n.geojson` (EPSG:32636) і `data/park_quarters_wgs84.geojson`, межа парку – `data/park_boundary.json`. Площа – 32,39 га (за буклетом коледжу – 32 га).

## Результати (власний розрахунок)

* Деревами, вищими за 5 м, вкрито 74 % площі; медіанна висота крон дерев – 15 м, 90-й процентиль – 19 м (сітка 5 м).
* Середній NDVI 2024 року: 0,17 у січні, пік 0,87 у червні; у лютому і грудні безхмарних знімків не було.
* Кластеризація ембедингів AlphaEarth після згладжування виділила три типи ділянок: зімкнені лісові насадження (19,6 га), узлісся, галявини й молоді насадження (9,4 га) та інші землі (3,4 га, переважно квартали 49–54). Назви зонам дано за знімком і висотою крон.

## Структура

* `notebooks/` – ноутбук Colab;
* `data/park3.npz` – сітки в EPSG:32636 з початком 498880 / 5537360: висота крон (5 м), місячний NDVI 2024 (20 м), зони ШІ (10 м); `data/park_stats.json`, `data/zones_area.json` – підсумки;
* `tools/quarters_digitize.py`, `tools/quarters_georef.py` – оцифрування і прив'язка кварталів; `tools/boundary.py` – афінне перетворення за опорними точками; `tools/stats.py` – підсумки; `tools/grids.py` – читання й маскування сіток; `tools/media2.py` – анімації для презентації; `tools/make_notebook.py` – генерує ноутбук.

Знімки Esri, буклет і растрова схема кварталів у репозиторії не поширюються: їх треба отримати з першоджерел.

## Навчання

У Навчально-науковому інституті лісового і садово-паркового господарства НУБіП України навчають за спеціальностями H4 «Лісове господарство», H3 «Садово-паркове господарство» і G14 «Деревообробні та меблеві технології». Кафедра таксації лісу та лісового менеджменту викладає, зокрема, дисципліни «ШІ-рішення в лісовому секторі», «Дрони для моніторингу довкілля», «Дистанційне зондування Землі» і «ГІС-технології в лісовому господарстві». Вступ на бакалаврат після коледжу – за результатами НМТ, див. Правила прийому на [nubip.edu.ua](https://nubip.edu.ua).

## Джерела

* Дендропарк і буклет: [lltk.edu.ua/pro-koledzh/dendropark](https://lltk.edu.ua/pro-koledzh/dendropark).
* Tolan J. et al. (2024). Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar. *Remote Sensing of Environment* 300, 113888.
* Pasquarella V. J. et al. (2023). Comprehensive quality assessment of optical satellite imagery using weakly supervised video learning. CVPR Workshops.
* Brown C. F. et al. (2025). AlphaEarth Foundations: an embedding field model for accurate and efficient global mapping from sparse label data. arXiv:2507.22291.
* Copernicus Sentinel-2 data (2024), ESA.

Код – MIT. Дані третіх сторін – на умовах їхніх ліцензій.
