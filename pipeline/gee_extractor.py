import ee
import json
import pandas as pd
from google.oauth2.service_account import Credentials

class GEEExtractor:
    def __init__(self, credentials_path="config/credentials.json"):
        self.credentials_path = credentials_path
        self.project_id = None
        self.jabar_regions = None
        self.initialize_gee()

    def initialize_gee(self):
        """Inisialisasi koneksi ke Google Earth Engine menggunakan Service Account."""
        try:
            with open(self.credentials_path, "r") as f:
                creds_data = json.load(f)
                self.project_id = creds_data.get("project_id")
            
            scopes = ["https://www.googleapis.com/auth/earthengine", "https://www.googleapis.com/auth/cloud-platform"]
            creds = Credentials.from_service_account_file(self.credentials_path, scopes=scopes)
            ee.Initialize(credentials=creds, project=self.project_id)
            print(f"[GEE] Earth Engine berhasil diinisialisasi pada project GCP: '{self.project_id}'")
            
            # Definisikan area studi (Batas Resmi Jabar)
            self.jabar_regions = ee.FeatureCollection("FAO/GAUL/2015/level2") \
                                   .filter(ee.Filter.eq('ADM1_NAME', 'Jawa Barat'))
        except Exception as e:
            raise RuntimeError(f"Gagal menginisialisasi Earth Engine: {e}")

    def clean_kabupaten_name(self, name):
        """Menstandarkan nama kabupaten agar cocok dengan pipeline lokal."""
        return str(name).lower().strip().replace('kabupaten ', '').replace('kota ', '')

    def extract_era5(self, target_date_str):
        """
        Mengekstrak data suhu permukaan (ERA5 LST) harian untuk satu tanggal tertentu.
        Format target_date_str: 'YYYY-MM-DD'
        """
        print(f"[GEE] Mengekstrak ERA5 LST untuk tanggal: {target_date_str}")
        start_date = ee.Date(target_date_str)
        end_date = start_date.advance(1, 'day')
        
        # Load Koleksi ERA5-Land Hourly
        era5_hourly = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
                        .filterBounds(self.jabar_regions) \
                        .filterDate(start_date, end_date) \
                        .select('skin_temperature')
        
        size = era5_hourly.size().getInfo()
        if size == 0:
            print(f"[GEE] ⚠ Tidak ada data ERA5 untuk tanggal {target_date_str}")
            return pd.DataFrame()

        # Rata-rata harian dan konversi Kelvin ke Celsius
        daily_mean = era5_hourly.mean().subtract(273.15).rename('LST_Day')
        image_processed = daily_mean.addBands(ee.Image.pixelLonLat())
        
        # Hitung statistik spasial per kabupaten
        stats = image_processed.select('LST_Day').reduceRegions(
            collection=self.jabar_regions,
            reducer=ee.Reducer.mean() \
                     .combine(ee.Reducer.max(), None, True) \
                     .combine(ee.Reducer.percentile([95]), None, True),
            scale=9000
        )
        
        # Ambil koordinat piksel terpanas
        def get_formatted_features(f):
            kab_name = f.get('ADM2_NAME')
            lst_mean = f.get('LST_Day_mean')
            lst_max = f.get('LST_Day_max')
            lst_p95 = f.get('LST_Day_p95')
            
            # Sub-query koordinat piksel terpanas
            max_pixel_mask = image_processed.select('LST_Day').geq(ee.Image.constant(lst_max).subtract(0.001))
            max_pixel_coords = image_processed.select(['longitude', 'latitude']).updateMask(max_pixel_mask)
            
            coord_stats = max_pixel_coords.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=f.geometry(),
                scale=9000,
                maxPixels=1e9
            )
            
            return ee.Feature(None, {
                'date': target_date_str,
                'Kabupaten': kab_name,
                'ERA5_LST_Mean': lst_mean,
                'ERA5_LST_Max': lst_max,
                'ERA5_LST_Percentile95': lst_p95,
                'ERA5_Cloud_Cover_Percentage': 0, # ERA5 Bebas awan
                'ERA5_Max_Lon': coord_stats.get('longitude'),
                'ERA5_Max_Lat': coord_stats.get('latitude')
            })

        formatted_stats = stats.map(get_formatted_features)
        
        # Pindahkan data dari GEE server ke client local pandas
        features = formatted_stats.getInfo().get('features', [])
        data_list = [feat['properties'] for feat in features]
        
        df = pd.DataFrame(data_list)
        if not df.empty:
            df['Kabupaten'] = df['Kabupaten'].apply(self.clean_kabupaten_name)
        return df

    def extract_soil_moisture(self, target_date_str):
        """Mengekstrak Soil Moisture harian dari GLDAS."""
        print(f"[GEE] Mengekstrak GLDAS Soil Moisture untuk tanggal: {target_date_str}")
        start_date = ee.Date(target_date_str)
        end_date = start_date.advance(1, 'day')
        
        gldas_coll = ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H') \
                       .filterBounds(self.jabar_regions) \
                       .filterDate(start_date, end_date) \
                       .select('SoilMoi0_10cm_inst')
        
        if gldas_coll.size().getInfo() == 0:
            print(f"[GEE] ⚠ Tidak ada data GLDAS untuk tanggal {target_date_str}")
            return pd.DataFrame()

        mean_img = gldas_coll.mean().rename('Soil_Moisture_Daily')
        
        stats = mean_img.reduceRegions(
            collection=self.jabar_regions,
            reducer=ee.Reducer.mean(),
            scale=27830
        )
        
        features = stats.getInfo().get('features', [])
        data_list = []
        for feat in features:
            props = feat['properties']
            data_list.append({
                'date': target_date_str,
                'Kabupaten': self.clean_kabupaten_name(props.get('ADM2_NAME')),
                'SoilMoisture_Daily_Mean': props.get('mean', None)
            })
            
        return pd.DataFrame(data_list)

    def extract_ndvi(self, target_date_str):
        """Mengekstrak NDVI (MODIS 8-day composite) yang melingkupi tanggal target."""
        print(f"[GEE] Mengekstrak MODIS NDVI untuk tanggal: {target_date_str}")
        target_date = ee.Date(target_date_str)
        
        # Kita cari NDVI terdekat (MODIS 8-day terupdate)
        ndvi_coll = ee.ImageCollection('MODIS/061/MOD09A1') \
                      .filterBounds(self.jabar_regions) \
                      .filterDate(target_date.advance(-8, 'day'), target_date.advance(1, 'day'))
        
        if ndvi_coll.size().getInfo() == 0:
            print(f"[GEE] ⚠ Tidak ada data MODIS NDVI untuk tanggal {target_date_str}")
            return pd.DataFrame()
            
        # Ambil citra paling baru dalam rentang waktu tersebut
        latest_ndvi_img = ndvi_coll.sort('system:time_start', False).first()
        
        # Hitung NDVI formula: (NIR - RED) / (NIR + RED)
        nir = latest_ndvi_img.select('sur_refl_b02')
        red = latest_ndvi_img.select('sur_refl_b01')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI_8Day')
        
        stats = ndvi.reduceRegions(
            collection=self.jabar_regions,
            reducer=ee.Reducer.mean(),
            scale=500
        )
        
        features = stats.getInfo().get('features', [])
        data_list = []
        for feat in features:
            props = feat['properties']
            data_list.append({
                'date': target_date_str,
                'Kabupaten': self.clean_kabupaten_name(props.get('ADM2_NAME')),
                'NDVI_8Day_Mean': props.get('mean', None)
            })
            
        return pd.DataFrame(data_list)

    def extract_landsat8(self, target_date_str):
        """Mengekstrak LST Landsat 8 jika satelit melintas pada tanggal tersebut."""
        print(f"[GEE] Mencoba mengekstrak Landsat 8 untuk tanggal: {target_date_str}")
        start_date = ee.Date(target_date_str)
        end_date = start_date.advance(1, 'day')
        
        landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                     .filterBounds(self.jabar_regions) \
                     .filterDate(start_date, end_date)
        
        if landsat8.size().getInfo() == 0:
            print(f"[GEE] Landsat 8 tidak melintas di Jawa Barat pada tanggal {target_date_str}")
            # Kembalikan dataframe kosong dengan nama kabupaten saja agar bisa di-merge dengan bfill/ffill nanti
            empty_list = []
            # Ambil list kabupaten dari jabar_regions
            features = self.jabar_regions.getInfo().get('features', [])
            for feat in features:
                kab_name = feat['properties']['ADM2_NAME']
                empty_list.append({
                    'date': target_date_str,
                    'Kabupaten': self.clean_kabupaten_name(kab_name),
                    'LST_Mean': None,
                    'LST_Max': None,
                    'LST_Percentile95': None,
                    'Cloud_Cover_Percentage': None
                })
            return pd.DataFrame(empty_list)

        # Preprocessing Landsat (Masking awan & Celsius)
        def preprocess(image):
            qa = image.select('QA_PIXEL')
            cloud_bit = (1 << 3)
            shadow_bit = (1 << 4)
            is_cloud = qa.bitwiseAnd(cloud_bit).neq(0).Or(qa.bitwiseAnd(shadow_bit).neq(0))
            
            lst_celsius = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
            return lst_celsius.updateMask(is_cloud.Not()).addBands(is_cloud.rename('Cloud_Mask'))

        processed = landsat8.map(preprocess).mean() # Rata-rata jika ada multiple overlapping tiles
        
        stats = processed.select(['LST', 'Cloud_Mask']).reduceRegions(
            collection=self.jabar_regions,
            reducer=ee.Reducer.mean() \
                     .combine(ee.Reducer.max(), None, True) \
                     .combine(ee.Reducer.percentile([95]), None, True),
            scale=30
        )
        
        features = stats.getInfo().get('features', [])
        data_list = []
        for feat in features:
            props = feat['properties']
            cloud_frac = props.get('Cloud_Mask_mean', None)
            cloud_pct = cloud_frac * 100 if cloud_frac is not None else None
            
            data_list.append({
                'date': target_date_str,
                'Kabupaten': self.clean_kabupaten_name(props.get('ADM2_NAME')),
                'LST_Mean': props.get('LST_mean', None),
                'LST_Max': props.get('LST_max', None),
                'LST_Percentile95': props.get('LST_p95', None),
                'Cloud_Cover_Percentage': cloud_pct
            })
            
        return pd.DataFrame(data_list)
