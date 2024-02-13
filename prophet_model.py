import pandas as pd
from prophet import Prophet

# Veri setini yükle
df = pd.read_csv("train.csv")
res = pd.read_csv("sample_submission.csv")

# Dolar kuru verilerini yükle
dolar_rates = pd.read_csv("dolar_rates.csv")

# 'Ay' sütununu datetime formatına çevir
dolar_rates['date'] = pd.to_datetime(dolar_rates['date'], format='%Y-%m')

# 'month_id' sütununu datetime formatına çevir
df['month_id'] = pd.to_datetime(df['month_id'], format='%Y%m')

# Verileri tarihe göre birleştir
df_merged = pd.merge(df, dolar_rates, how='left', left_on='month_id', right_on='date')

# Eksik dolar kuru değerlerini doldur (isteğe bağlı)
df_merged['Ortalama Dolar Kuru'].fillna(method='ffill', inplace=True)

# Aylık enflasyon veri setini oluştur
inflation_data = {
    'date': ['01-2024', '12-2023', '11-2023', '10-2023', '09-2023', '08-2023', '07-2023', '06-2023', '05-2023', '04-2023', '03-2023', '02-2023', '01-2023', '12-2022', '11-2022', '10-2022', '09-2022', '08-2022', '07-2022', '06-2022', '05-2022', '04-2022', '03-2022', '02-2022', '01-2022', '12-2021', '11-2021', '10-2021', '09-2021', '08-2021', '07-2021', '06-2021', '05-2021', '04-2021', '03-2021', '02-2021', '01-2021', '12-2020', '11-2020', '10-2020', '09-2020', '08-2020', '07-2020', '06-2020', '05-2020', '04-2020', '03-2020', '02-2020', '01-2020'],
    'monthly_inflation': [13.40, 5.86, 6.56, 6.86, 9.50, 18.18, 18.98, 7.84, 0.08, 4.78, 4.58, 6.30, 13.30, 2.36, 5.76, 7.08, 6.16, 2.92, 4.74, 9.90, 5.96, 9.62, 10.92, 9.62, 9.62, 15.58, 27.16, 5.78, 4.78, 5.54, 3.88, 3.88, 1.78, 3.36, 2.16, 1.82, 3.36, 2.50, 4.60, 4.26, 1.94, 1.72, 1.16, 2.26, 2.72, 1.70, 1.14, 0.70, 2.70]
}

# 'ds' sütununu datetime formatına çevir
inflation_data['ds'] = pd.to_datetime(inflation_data['date'], format='%m-%Y')

# İstenmeyen sütunları (date) düşür
del inflation_data['date']

# Ana veri seti ile aylık enflasyon veri setini birleştir
df_merged = pd.merge(df_merged, pd.DataFrame(inflation_data), how='left', left_on='month_id', right_on='ds')

# Eksik değerleri doldur (isteğe bağlı)
df_merged['monthly_inflation'].fillna(0, inplace=True)

# Submission dosyasını işle
res[['month_id', 'merchant_id']] = res['id'].str.extract(r'(\d{6})(merchant_\d+)')

# DataFrame'i 'merchant_id' ve 'month_id' sütunlarına göre sırala
df_sorted = df.sort_values(by=['merchant_id', 'month_id'], ascending=[True, False])

# 'merchant_id' gruplarına göre DataFrame'i grupla ve her grup için ilk satırı al
latest_data = df_sorted.groupby('merchant_id').first().reset_index()

# Submission ve son verileri birleştir
result = pd.merge(res, latest_data, on='merchant_id', how='left')

# Küçük değerleri ve ayrılan müşterileri yok say
result.loc[(result['net_payment_count_y'] < 5) | (result['month_id_y'] < pd.to_datetime('2023-09-01')), 'net_payment_count_x'] = 0

# Temel çizgi için en son veriyi al
result.loc[~((result['net_payment_count_y'] < 5) | (result['month_id_y'] < pd.to_datetime('2023-09-01'))), 'net_payment_count_x'] = result['net_payment_count_y']

# Modeli uygulayacak müşteri kimliklerini al
ids = result[result['net_payment_count_x'] != 0]['merchant_id'].unique()

# Veriyi zaman serisi formatına getir
df.set_index('month_id', inplace=True)

# Prophet modelini kullanarak tahminler yap
for merchant_id in ids:
    df_t = df[df['merchant_id'] == merchant_id]

    # Modeli eğitmek için yeterli veri var mı kontrol et
    if len(df_t) > 3:
        try:
            # Prophet modelini oluştur ve parametreleri ayarla
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                yearly_seasonality=False,  # Yıllık mevsimselliği devre dışı bırak
                weekly_seasonality=False   # Haftalık mevsimselliği devre dışı bırak
            )

            # Trend ve tatil bileşenlerini ekleyin
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_country_holidays(country_name='TR')

            # Dolar kuru verisini ekleyin
            model.add_regressor('Ortalama Dolar Kuru')
            model.add_regressor('monthly_inflation')

            # Veriyi formatlayın ve modele uygulayın
            df_t_prophet = df_t.reset_index().rename(columns={'month_id': 'ds', 'net_payment_count': 'y'})
            model.fit(df_t_prophet)

            # Gelecek 3 ay için tahmin yap
            future = model.make_future_dataframe(periods=3, freq='M')
            future['Ortalama Dolar Kuru'] = df_merged[df_merged['merchant_id'] == merchant_id]['Ortalama Dolar Kuru'].values
            future['monthly_inflation'] = df_merged[df_merged['merchant_id'] == merchant_id]['monthly_inflation'].values
            forecast = model.predict(future)

            # Tahminleri sonuç DataFrame'ine ekle
            result.loc[result['merchant_id'] == merchant_id, 'net_payment_count_x'] = forecast['yhat'].tail(3).values
        except Exception as e:
            print(f"Error fitting model for merchant_id {merchant_id}: {e}")

# 'net_payment_count_x' sütununu tut
columns_to_keep = ['id', 'net_payment_count_x']
result = result[columns_to_keep]

# 'net_payment_count_x' sütununu 'net_payment_count' olarak yeniden adlandır
result = result.rename(columns={'net_payment_count_x': 'net_payment_count'})

# CSV'ye kaydet
result.to_csv('submission_prophetinf_updated.csv', index=False)