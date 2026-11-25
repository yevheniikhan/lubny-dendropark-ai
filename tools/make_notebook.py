"""Build notebooks/dendropark_ai_gis.ipynb (run: python tools/make_notebook.py)."""
import nbformat as nbf
from pathlib import Path

REPO = "https://github.com/yevheniikhan/lubny-dendropark-ai"
NB = "dendropark_ai_gis.ipynb"
BADGE = (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         f"(https://colab.research.google.com/github/yevheniikhan/lubny-dendropark-ai/blob/main/notebooks/{NB})")

cells = [
"md:" + f"""# Дендропарк імені Василя Байтали очима супутника і штучного інтелекту

{BADGE}

Лубенський лісотехнічний фаховий коледж · практичне заняття з ГІС і ДЗЗ

За 20 хвилин ви:

1. побачите свій дендропарк на карті з роздільною здатністю менше метра;
2. дізнаєтеся висоту кожної ділянки намету, виміряну з космосу;
3. подивитеся, як парк «дихає» протягом року за знімками Sentinel-2;
4. дасте штучному інтелекту знайти зони парку без жодної підказки;
5. порахуєте окремі крони дерев (бонус із GPU).

Потрібен лише обліковий запис Google. Перед стартом: **Середовище виконання → Змінити тип → CPU** (для кроку 5 – GPU).""",

"md:## 0. Підготовка\n\nВкажіть свій проєкт Google Cloud з увімкненим Earth Engine (безкоштовно для навчання: [earthengine.google.com/noncommercial](https://earthengine.google.com/noncommercial/)). Вікно авторизації підтверджуєте ви.",
"""!pip -q install geemap pyproj scikit-learn
import ee, geemap, numpy as np, requests, matplotlib.pyplot as plt
PROJECT = 'your-ee-project'          # <- замініть на свій проєкт
try:
    ee.Initialize(project=PROJECT)
except Exception:
    ee.Authenticate(); ee.Initialize(project=PROJECT)

# Дендропарк імені В. Д. Байтали, Лубни. Межа – картосхема з буклета коледжу, прив'язана до знімка
# сплайном тонкої пластини (опорні точки: data/lltk/gcp.json у репозиторії).
PARK = ee.Geometry.Polygon(PARK_COORDS)
CENTER = PARK.centroid(1).coordinates().getInfo()[::-1]
UTM = 'EPSG:32636'                    # метрична проєкція для зони 36N
print('Площа контуру, га:', round(PARK.area(1).getInfo() / 1e4, 1))""",

"md:## 1. Дендропарк з висоти 500 м\n\nПідкладка – мозаїка аерофото- і супутникових знімків Esri з роздільною здатністю до 0,5 м. Увімкніть шари в правому верхньому куті карти. Інструментом малювання можна обвести власний контур парку: він стане доступним як `m.user_roi`.",
"""m = geemap.Map(center=CENTER, zoom=16, basemap='Esri.WorldImagery')
m.addLayer(ee.Image().paint(PARK, 0, 3), {'palette': ['ffd400']}, 'Контур дендропарку')
m""",

"md:## 2. Висота дерев, виміряна з космосу\n\nКарта висоти намету Meta і World Resources Institute (роздільна здатність 1 м) отримана нейромережею, навченою на лідарних вимірах. Подивимося, де ростуть найвищі дерева парку.",
"""col = ee.ImageCollection('projects/meta-forest-monitoring-okw37/assets/CanopyHeight').filterBounds(PARK)
chm = col.mosaic().select(0).rename('h').setDefaultProjection(col.first().projection())
vis_h = {'min': 0, 'max': 30, 'palette': ['ffffff', 'e8d9a8', 'b9cf96', '7fae86', '3f7d66', '14524a', '0a2f2a']}
m = geemap.Map(center=CENTER, zoom=16, basemap='Esri.WorldImagery')
m.addLayer(chm.updateMask(chm.gt(2)).clip(PARK), vis_h, 'Висота намету, м')
m.add_colorbar(vis_h, label='Висота намету, м')
m""",
"""# Статистика висот у межах парку
st = chm.reduceRegion(ee.Reducer.percentile([50, 90, 99]).combine(ee.Reducer.max(), sharedInputs=True), PARK, 1, maxPixels=1e9).getInfo()
print({k: round(v, 1) for k, v in st.items()})
box = PARK.buffer(60).bounds(1, ee.Projection(UTM))
h2 = chm.unmask(0).reduceResolution(ee.Reducer.mean(), maxPixels=16).reproject(crs=UTM, scale=2)
arr = np.array(h2.sampleRectangle(region=box, defaultValue=0).get('h').getInfo(), float)
plt.figure(figsize=(6, 3)); plt.hist(arr[arr > 2].ravel(), bins=40, color='#14524A')
plt.xlabel('висота, м'); plt.ylabel('кількість пікселів 2 × 2 м'); plt.title('Розподіл висот намету в дендропарку'); plt.show()""",
"md:### 2a. Дендропарк у 3D\n\nПеретворимо карту висот на тривимірну модель і обертатимемо її.",
"""from matplotlib import animation
from IPython.display import HTML
z = arr[::2, ::2].astype(float); z[z < 0] = 0
y, x = np.mgrid[0:z.shape[0], 0:z.shape[1]]
fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(projection='3d')
ax.plot_surface(x, -y, z, cmap='YlGn', linewidth=0, antialiased=False, vmin=0, vmax=30)
ax.set_zlim(0, 60); ax.axis('off')
anim = animation.FuncAnimation(fig, lambda a: ax.view_init(35, a), frames=range(0, 360, 4), interval=60)
plt.close(); HTML(anim.to_jshtml())""",

"md:## 3. Парк дихає: рік очима Sentinel-2\n\nNDVI – індекс, що показує активність зеленого листя. Порахуємо його для кожного місяця 2024 року і зробимо анімацію.",
"""csp = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')     # маска хмар Cloud Score+
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(PARK).filterDate('2024-01-01', '2025-01-01')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80)).linkCollection(csp, ['cs_cdf'])
      .map(lambda i: i.updateMask(i.select('cs_cdf').gte(0.6))))
def monthly(mn):
    c = s2.filter(ee.Filter.calendarRange(mn, mn, 'month'))
    nd = c.median().normalizedDifference(['B8', 'B4']).rename('NDVI')
    return ee.Image(ee.Algorithms.If(c.size().gt(0), nd, ee.Image.constant(0).rename('NDVI').updateMask(0))).set('month', mn)
months = ee.ImageCollection([monthly(mn) for mn in range(1, 13)])
curve = [months.filter(ee.Filter.eq('month', mn)).first().reduceRegion(ee.Reducer.mean(), PARK, 10).get('NDVI').getInfo() for mn in range(1, 13)]
plt.figure(figsize=(6, 3)); plt.plot(range(1, 13), curve, 'o-', color='#14524A')
plt.xticks(range(1, 13), ['січ', 'лют', 'бер', 'кві', 'тра', 'чер', 'лип', 'сер', 'вер', 'жов', 'лис', 'гру'])
plt.ylabel('середній NDVI'); plt.title('Річний цикл дендропарку, 2024'); plt.grid(alpha=.3); plt.show()""",
"""region = PARK.buffer(150).bounds(1, ee.Projection(UTM))
vis_n = {'min': 0, 'max': 0.9, 'palette': ['8c510a', 'd8b365', 'f6e8c3', 'c7eae5', '5ab4ac', '01665e']}
frames = months.map(lambda i: i.visualize(**vis_n))
url = frames.getVideoThumbURL({'region': region, 'dimensions': 480, 'crs': UTM, 'framesPerSecond': 2})
from IPython.display import Image
Image(url=url)""",

"md:## 4. Штучний інтелект знаходить зони парку сам\n\nМодель AlphaEarth Foundations (Google DeepMind) стиснула річні спостереження різних супутників у 64 числа для кожного пікселя 10 м. Ми не даємо моделі жодних назв порід: алгоритм k-середніх лише групує подібні пікселі. Порівняйте зони з тим, що знаєте про парк.",
"""emb = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
       .filterDate('2024-01-01', '2025-01-01').filterBounds(PARK).mosaic())
sample = emb.sample(region=PARK, scale=10, numPixels=3000, seed=1)
K = 5                                  # спробуйте 3, 5, 8
clusterer = ee.Clusterer.wekaKMeans(K, seed=1).train(sample)
zones = emb.cluster(clusterer).clip(PARK)
m = geemap.Map(center=CENTER, zoom=16, basemap='Esri.WorldImagery')
m.addLayer(zones.randomVisualizer(), {}, f'{K} зон, знайдених без учителя', opacity=0.7)
m""",

"md:## 5. Бонус: штучний інтелект рахує крони (потрібен GPU)\n\nМодель сегментації SAM за текстовою підказкою «tree» виділяє окремі крони на знімку високої роздільності. Результат перевіряйте очима: модель навчена на звичайних фото, тому може помилятися в густому наметі.",
"""# Змініть середовище на GPU і виконайте:
# !pip -q install segment-geospatial groundingdino-py
# from samgeo import tms_to_geotiff
# from samgeo.text_sam import LangSAM
# b = PARK.bounds().coordinates().getInfo()[0]
# bbox = [b[0][0], b[0][1], b[2][0], b[2][1]]
# tms_to_geotiff(output='park.tif', bbox=bbox, zoom=19, source='Satellite', overwrite=True)
# sam = LangSAM()
# sam.predict('park.tif', 'tree', box_threshold=0.24, text_threshold=0.24)
# sam.show_anns(cmap='Greens', add_boxes=False, alpha=0.6, title='Крони, знайдені SAM')
# sam.raster_to_vector('park_mask.tif', 'tree_crowns.gpkg')   # полігони для QGIS""",

"md:## Далі\n\nУ цьому ноутбуці ви пройшли шлях фахівця з дистанційного моніторингу лісів: межа парку, висота крон, сезонна динаміка і зони, знайдені штучним інтелектом. Цих методів навчають у Навчально-науковому інституті лісового і садово-паркового господарства НУБіП України (nubip.edu.ua).\n\n**Джерела даних:** Esri World Imagery; Meta & WRI Global Canopy Height (Tolan et al., 2024); Copernicus Sentinel-2; Cloud Score+ (Pasquarella et al., 2023); AlphaEarth Foundations Satellite Embedding (Brown et al., 2025); SAM (Meta AI). Межа парку – картосхема з буклета коледжу (lltk.edu.ua/pro-koledzh/dendropark).",

]


def build(park_coords):
    n = nbf.v4.new_notebook()
    n.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                  "colab": {"provenance": []}}
    out = []
    for c in cells:
        if c.startswith("md:"):
            out.append(nbf.v4.new_markdown_cell(c[3:]))
        else:
            out.append(nbf.v4.new_code_cell(c.replace("PARK_COORDS", park_coords)))
    n.cells = out
    p = Path(__file__).resolve().parents[1] / "notebooks" / NB
    nbf.write(n, p)
    return p


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1] if len(sys.argv) > 1 else "[[32.98, 49.98], [32.99, 49.98], [32.99, 49.99], [32.98, 49.99]]"))
