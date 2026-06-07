import ee
import json

try:
    with open("config/credentials.json", "r") as f:
        creds_data = json.load(f)
        project_id = creds_data.get("project_id")
    
    import google.oauth2.service_account
    creds = google.oauth2.service_account.Credentials.from_service_account_file(
        "config/credentials.json",
        scopes=["https://www.googleapis.com/auth/earthengine", "https://www.googleapis.com/auth/cloud-platform"]
    )
    ee.Initialize(credentials=creds, project=project_id)
except Exception as e:
    print(f"Gagal: {e}")
    exit(1)

# Ambil sample wilayah Bandung
bandung = ee.FeatureCollection("FAO/GAUL/2015/level2") \
            .filter(ee.Filter.eq('ADM2_NAME', 'Bandung'))

start_date = ee.Date("2026-06-01")
era5_hourly = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
                .filterBounds(bandung) \
                .filterDate(start_date, start_date.advance(1, 'day')) \
                .select('skin_temperature')

daily_mean = era5_hourly.mean().subtract(273.15).rename('LST_Day')

stats = daily_mean.reduceRegions(
    collection=bandung,
    reducer=ee.Reducer.mean() \
             .combine(ee.Reducer.max(), None, True) \
             .combine(ee.Reducer.percentile([95]), None, True),
    scale=9000
)

# Cetak properti dari feature pertama
feature = stats.first().getInfo()
print("Properti yang dihasilkan oleh reduceRegions di Python GEE:")
print(json.dumps(feature['properties'], indent=4))
