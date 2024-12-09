import pickle
import streamlit as st
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def load_data():
    df = pd.read_csv('passing-grade.csv')
    df = df.dropna(subset=['RATAAN', 'S.BAKU', 'MIN'])
    return df

# Memuat model prediksi passing grade dari file .sav
try:
    model = pickle.load(open('Lasso_Regression.sav', 'rb'))
    poly = pickle.load(open('polynomial_features.sav', 'rb'))
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan file tersebut ada.")
    model = None
    poly = None

# Membaca file CSV untuk dataset
try:
    df_passing_grade = pd.read_csv('passing-grade.csv')
    df_passing_grade.reset_index(inplace=True)
    df_passing_grade.dropna(subset=['RATAAN', 'S.BAKU'], inplace=True)
except FileNotFoundError:
    st.error("File 'passing-grade.csv' tidak ditemukan. Pastikan file tersebut ada.")
    df_passing_grade = None

# Sidebar untuk navigasi
st.sidebar.header("Navigasi")
menu_option = st.sidebar.selectbox(
    "Pilih menu:",
    ["Home","Lihat Dataset", "Tampilkan Grafik", "Prediksi Passing Grade", "Modelling"]
)

# Fungsi untuk halaman utama / About Us
if menu_option == "Home":
    st.image("fotoptn.jpg", caption="Aplikasi Prediksi Passing Grade")
    st.header("📚 Selamat datang di aplikasi *Prediksi Passing Grade*!")
    st.markdown(
        """
        Aplikasi ini memprediksi nilai *Passing Grade (MIN)* berdasarkan nilai *RATAAN* dan *S.BAKU*.
        Gunakan menu di sidebar untuk memilih antara melihat dataset, grafik, atau melakukan prediksi.

        ### Kenapa Aplikasi Ini Dibuat?

        Salah satu tantangan terbesar dalam memilih program studi adalah memahami nilai passing grade. Nilai ini mencerminkan standar minimum 
        yang biasanya diperlukan untuk diterima di sebuah program studi tertentu. Dengan mengetahui nilai ini, calon mahasiswa dapat memiliki gambaran 
        realistis tentang peluang mereka dan mempersiapkan diri dengan lebih baik. Aplikasi ini dibuat untuk membantu calon mahasiswa yang sedang bersiap-siap untuk melanjutkan pendidikan ke jenjang perguruan tinggi. 
        Kami memahami bahwa proses memilih program studi dan universitas tidaklah mudah. Banyak pertimbangan yang harus diambil, 
        seperti kemampuan akademik, minat, cita-cita, dan tentunya peluang untuk diterima di program studi yang diinginkan.

        ### Apa yang Bisa Dilakukan dengan Aplikasi Ini?

        - *Melihat Dataset Passing Grade*  
          Aplikasi ini menyediakan data nilai passing grade untuk berbagai program studi di Perguruan Tinggi Negeri (PTN). Anda bisa menjelajahi data ini 
          untuk memahami program studi mana yang sesuai dengan kemampuan akademik Anda.

        - *Melihat Grafik dan Visualisasi Data*  
          Dengan visualisasi data, Anda dapat melihat pola-pola menarik dari nilai rata-rata (RATAAN), standar deviasi (S.BAKU), 
          serta nilai minimum dan maksimum dari passing grade. Ini akan membantu Anda dalam memahami bagaimana data-data ini saling terkait.

        - *Prediksi Passing Grade*  
          Jika Anda ingin mengetahui peluang Anda untuk diterima, fitur prediksi ini sangat bermanfaat. Dengan memasukkan nilai rata-rata dan standar deviasi hasil belajar Anda, 
          aplikasi akan memprediksi nilai minimum (passing grade) yang diperlukan untuk program studi tertentu.

        - *Rekomendasi Program Studi*  
          Selain memprediksi passing grade, aplikasi ini juga memberikan rekomendasi program studi dan universitas yang memiliki standar nilai mendekati kemampuan Anda. 
          Ini sangat membantu jika Anda ingin mencari alternatif yang tetap sesuai dengan minat dan potensi Anda.

        ### Siapa yang Cocok Menggunakan Aplikasi Ini?

        Aplikasi ini dirancang untuk calon mahasiswa, guru BK (Bimbingan Konseling), orang tua, atau siapa saja yang terlibat dalam proses bimbingan pendidikan. 
        Dengan aplikasi ini, pengguna dapat membuat keputusan yang lebih matang dan strategis terkait pilihan pendidikan.

        ### Keunggulan Aplikasi Ini:

        - Antarmuka yang sederhana dan mudah digunakan.
        - Data yang relevan dan bermanfaat.
        - Fitur prediksi yang canggih untuk membantu Anda membuat keputusan berdasarkan data.

        Dengan menggunakan aplikasi ini, kami berharap Anda dapat lebih percaya diri dan terarah dalam memilih program studi yang sesuai dengan kemampuan dan impian Anda. Selamat menggunakan dan semoga sukses! 🎓✨
        """
    )

    st.write("---")
    st.subheader("👥 Tim Pengembang")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.image("team_icon.png", width=250)
    with col2:
        st.markdown(
            """
            - *Nama*: Kelompok 1 TI 3C  
            - *Lokasi*: Politeknik Negeri Madiun  
            - *Bidang*: Sistem Informasi & Teknologi Informasi  
            """
        )
    st.write("---")
    st.markdown(
        """
        🌟 *Terima kasih telah menggunakan aplikasi ini. Jangan ragu untuk menghubungi kami melalui halaman kontak. Semoga aplikasi ini bermanfaat!
        """)

# Fungsi untuk menampilkan dataset
elif menu_option == "Lihat Dataset":
    if df_passing_grade is not None:
        st.header("📊 Dataset Passing Grade")
        st.write("### Data Passing Grade")
        st.markdown("""
        Dataset Passing Grade adalah kumpulan data yang berisi informasi nilai minimum (passing grade) yang diperlukan untuk diterima di program studi (prodi) tertentu di Perguruan Tinggi Negeri (PTN).
        1. *RATAAN*: Nilai rata-rata siswa dari hasil ujian.
        2. *S.BAKU*: Standar deviasi nilai, menggambarkan penyebaran data nilai dari rata-rata.
        3. *MIN*: Nilai minimum diterima di prodi tertentu.
        4. *MAX*: Nilai maksimum diterima.
        5. *PTN*: Nama Perguruan Tinggi Negeri.
        6. *NAMA PRODI*: Nama program studi.
        """)
        st.dataframe(df_passing_grade)
    else:
        st.warning("Dataset tidak tersedia.")

# Fungsi untuk menampilkan grafik
elif menu_option == "Tampilkan Grafik":
    if df_passing_grade is not None:
        # Grafik RATAAN
        st.write("### Grafik Nilai RATAAN")
        chart_rataan = alt.Chart(df_passing_grade).mark_line().encode(
            x=alt.X('index:Q', title='Index'),
            y=alt.Y('RATAAN:Q', title='Nilai RATAAN'),
            tooltip=['index', 'RATAAN']
        ).properties(title='Perubahan Nilai RATAAN')
        st.altair_chart(chart_rataan, use_container_width=True)
        st.write("Sumbu X (Index) : Menunjukkan index atau urutan data, mulai dari 0 hingga sekitar 480.",
        "Setiap nilai pada sumbu ini merepresentasikan data yang berurutan dalam dataset.")
        st.write("Sumbu Y (Nilai RATAAN) : Menunjukkan nilai rata-rata, dengan skala mulai dari sekitar 600 hingga 800.",
        "Nilai ini merepresentasikan rata-rata nilai passing grade dari  data yang dianalisis.")
        
        # Grafik Hubungan RATAAN, MIN, dan MAX
        st.write("### Grafik Hubungan antara RATAAN, MIN, dan MAX")
        line_chart = alt.Chart(df_passing_grade).mark_line().encode(
            x=alt.X('RATAAN:Q', title='Nilai RATAAN'),
            y=alt.Y('MIN:Q', title='Nilai MIN'),
            tooltip=['RATAAN', 'MIN']
        ).properties(title='Hubungan antara RATAAN dan Nilai MIN')
        line_chart_max = alt.Chart(df_passing_grade).mark_line(color='red').encode(
            x=alt.X('RATAAN:Q', title='Nilai RATAAN'),
            y=alt.Y('MAX:Q', title='Nilai MAX'),
            tooltip=['RATAAN', 'MAX']
        )
        combined_chart = line_chart + line_chart_max
        st.altair_chart(combined_chart, use_container_width=True)
        st.write("Sumbu X (Nilai RATAAN) menunjukkan nilai rata-rata passing grade yang dihitung dari data passing grade suatu program studi atau institusi.", 
        "Ini merepresentasikan rata-rata nilai yang dicapai atau diperlukan oleh calon mahasiswa untuk memenuhi syarat lulus atau diterima di program studi tersebut.")
        st.write("Sumbu Y (Nilai MIN dan MAX) menunjukkan nilai passing grade minimum (yang ditunjukkan oleh garis biru) adalah nilai minimum yang harus dicapai oleh calon mahasiswa untuk dapat diterima di suatu program studi atau universitas.",
        "Nilai ini digunakan untuk mencerminkan tingkat persaingan yang diperlukan untuk lulus. Sedangkan nilai passing grade maksimum (yang ditunjukkan oleh garis merah) adalah nilai maksimum ini mewakili nilai tertinggi yang dicapai", 
        "oleh peserta yang diterima dalam seleksi atau nilai terbaik yang diterima di suatu program studi atau universitas.  )")

        # Grafik 10 Prodi dengan Passing Grade Tertinggi
        st.write("### Sebaran 10 Prodi dengan Nilai Passing Grade Tertinggi")
        if df_passing_grade is not None and not df_passing_grade.empty:
            top_10_best_programs = df_passing_grade.nlargest(10, 'MIN')
            top_10_best_programs['PRODI_PTN'] = top_10_best_programs['NAMA PRODI'] + " - " + top_10_best_programs['PTN']
            
            best_programs_chart = alt.Chart(top_10_best_programs).mark_bar().encode(
                x=alt.X('PRODI_PTN:N', title='Nama Prodi - PTN', sort='-y'),
                y=alt.Y('MIN:Q', title='Nilai Passing Grade (MIN)'),
                color=alt.Color('PTN:N', title='PTN', legend=alt.Legend(orient='bottom')),
                tooltip=['NAMA PRODI', 'PTN', alt.Tooltip('MIN:Q', title='Passing Grade (MIN)', format='.2f')]
            ).properties(
                title='Sebaran 10 Prodi dengan Nilai Passing Grade Tertinggi',
                width='container',
                height=500
            ).configure_axisX(
                labelAngle=-45
            ).configure_title(
                fontSize=16,
                anchor='start'
            )
            st.altair_chart(best_programs_chart, use_container_width=True)
            st.write("Grafik ini memberikan gambaran program studi yang memiliki nilai passing grade terbaik di beberapa PTN. Dari grafik ini dapat disimpulkan:")
            st.write("- **Kompetisi Tinggi:** Program studi dengan nilai passing grade tinggi menandakan adanya persaingan yang ketat untuk mendapatkan program studi tersebut. Hal ini dipengaruhi oleh reputasi program studi, peluang kerja lulusan, serta minat calon mahasiswa terhadap program studi tersebut.") 
            st.write("- **Variasi Antar PTN:** Meskipun program studi sama, nilai passing gradenya berbeda-beda di setiap PTN. Perbedaan ini dipengaruhi oleh beberapa faktor seperti reputasi PTN, fasilitas yang dimiliki, lokasi PTN, serta kebijakan penerimaan mahasiswa baru yang diterapkan oleh masing-masing PTN.") 
            st.write("- **Dinamika Nilai Passing Grade:** Nilai passing grade ini bukan hal yang statis, melainkan dapat berubah dari waktu ke waktu. Perubahan ini dipengaruhi oleh berbagai faktor seperti jumlah pendaftar, kebijakan pemerintah terkait pendidikan tinggi, serta perkembangan program studi tersebut.")
        else:
            st.error("Data Passing Grade tidak tersedia atau kosong.")
            
        # Grafik 10 Prodi dengan Passing Grade Terendah
        st.write("### Sebaran 10 Prodi dengan Nilai Passing Grade Terendah")
        if df_passing_grade is not None and not df_passing_grade.empty:
            top_10_lowest_programs = df_passing_grade.nsmallest(10, 'MIN')
            top_10_lowest_programs['PRODI_PTN'] = top_10_lowest_programs['NAMA PRODI'] + " - " + top_10_lowest_programs['PTN']
            
            lowest_programs_chart = alt.Chart(top_10_lowest_programs).mark_bar().encode(
                x=alt.X('PRODI_PTN:N', title='Nama Prodi - PTN', sort='-y'),
                y=alt.Y('MIN:Q', title='Nilai Passing Grade (MIN)'),
                color=alt.Color('PTN:N', title='PTN', legend=alt.Legend(orient='bottom')),
                tooltip=['NAMA PRODI', 'PTN', alt.Tooltip('MIN:Q', title='Passing Grade (MIN)', format='.2f')]
            ).properties(
                title='Sebaran 10 Prodi dengan Nilai Passing Grade Terendah',
                width='container',
                height=500
            ).configure_axisX(
                labelAngle=-45
            ).configure_title(
                fontSize=16,
                anchor='start'
            )
            st.altair_chart(lowest_programs_chart, use_container_width=True)
            st.write("Grafik ini memberikan gambaran program studi yang memiliki nilai passing grade terendah di beberapa PTN. Dari grafik ini dapat disimpulkan:")
            st.write("- **Persaingan Rendah:** Program studi dengan nilai passing grade rendah cenderung memiliki persaingan yang lebih sedikit dibandingkan program studi dengan passing grade tinggi. Hal ini mungkin karena minat yang lebih rendah dari calon mahasiswa atau karakteristik program studi tersebut.") 
            st.write("- **Variasi Antar PTN:** Seperti pada nilai passing grade tinggi, terdapat variasi nilai passing grade yang rendah antar PTN. Faktor-faktor seperti reputasi PTN dan daya tarik lokasi PTN masih berpengaruh pada nilai ini.") 
            st.write("- **Kesempatan Lebih Tinggi:** Dengan nilai passing grade yang lebih rendah, calon mahasiswa memiliki peluang yang lebih besar untuk diterima, sehingga program-program ini bisa menjadi alternatif bagi mereka yang mencari peluang pendidikan tinggi yang lebih terbuka.")
        else:
            st.error("Data Passing Grade tidak tersedia atau kosong.")


# Fungsi untuk melakukan prediksi
elif menu_option == "Prediksi Passing Grade":
    st.header("🔮 Prediksi Passing Grade")
    
    st.write("### Masukkan Data untuk Prediksi")
    rataan = st.number_input('Masukkan Nilai RATAAN', min_value=0, max_value=800, value=0)
    sbaku = st.number_input('Masukkan Nilai S.BAKU', min_value=0, max_value=30, value=0)
    
    st.write("### Informasi PTN dan Prodi")
    selected_ptn = st.text_input("Masukkan PTN (misal: Universitas Indonesia)")
    selected_prodi = st.text_input("Masukkan Prodi (misal: Teknik Informatika)")
    
    if st.button('Prediksi'):
        if rataan <= 0:
            st.error("Nilai RATAAN tidak boleh 0 atau kosong!")
        elif sbaku <= 0:
            st.error("Nilai S.BAKU tidak boleh 0 atau kosong!")
        elif not selected_ptn.strip():
            st.error("PTN wajib diisi!")
        elif not selected_prodi.strip():
            st.error("Prodi wajib diisi!")
        else:
            if model and poly:
                try:
                    input_features = poly.transform([[rataan, sbaku]])
                    passing_grade_prediction = model.predict(input_features)
                    predicted_min = float(passing_grade_prediction[0])

                    st.write(f"**Prediksi Passing Grade (MIN)** untuk PTN **{selected_ptn.upper()}** dan Prodi **{selected_prodi.upper()}**:")
                    st.success(f"**{predicted_min:.2f}**")
                    st.write("""
                    Nilai ini menunjukkan nilai minimum yang diprediksi agar Anda lulus.
                    Pastikan untuk mempersiapkan dengan baik!
                    """)

                    # Mencari 5 PTN dan Prodi dengan passing grade terdekat
                    st.subheader("🎓 Rekomendasi PTN dan Prodi")
                    st.write("""
                    Berikut merupakan rekomendasi Perguruan Tinggi Negeri dan Program Studi,
                    berdasarkan Prediksi Passing Grade (MIN) :
                    """)
                    if df_passing_grade is not None:
                        df_passing_grade['selisih'] = abs(df_passing_grade['MIN'] - predicted_min)
                        top_recommendations = df_passing_grade.nsmallest(5, 'selisih')
                        top_recommendations_table = top_recommendations[['PTN', 'NAMA PRODI', 'MIN']]
                        top_recommendations_table['Passing Grade (MIN) Terdekat'] = top_recommendations_table['MIN'].apply(lambda x: f"{x:.2f}")
                        st.table(top_recommendations_table[['PTN', 'NAMA PRODI', 'Passing Grade (MIN) Terdekat']])

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            else:
                st.error("Model atau PolynomialFeatures tidak tersedia. Pastikan file berhasil dimuat.")

# Fungsi untuk melatih model
elif menu_option == "Modelling":
    st.header("🛠️ Modelling")
    st.write("### Melatih Model Prediksi Passing Grade")
    
    st.write("""
    Dalam proses pelatihan model ini, kami menggunakan beberapa algoritma regresi untuk memprediksi nilai **Passing Grade (MIN)** berdasarkan nilai **RATAAN** dan **S.BAKU**. Berikut adalah model yang digunakan:

    Bayangkan kamu sedang menghitung nilai Passing Grade untuk sebuah program studi berdasarkan nilai rata-rata dan deviasi standar dari nilai-nilai yang ada. Setiap model regresi ini adalah cara berbeda untuk memastikan kamu mendapatkan hasil yang akurat:

    - **Linear Regression**: Ini seperti menghitung nilai Passing Grade dengan cara yang paling sederhana. Kamu mengambil nilai RATAAN dan menggunakannya langsung untuk menentukan MIN. Model ini mencari garis lurus yang paling sesuai dengan data, sehingga hasilnya adalah prediksi yang langsung berdasarkan rata-rata.

    - **Ridge Regression**: Sekarang, bayangkan kamu ingin memastikan bahwa perhitunganmu tidak terlalu dipengaruhi oleh nilai-nilai ekstrem. Kamu menambahkan sedikit "aturan" untuk menjaga agar hasilmu tetap stabil, dengan memberikan penalti pada nilai yang terlalu besar. Ini membantu menghindari overfitting dan memastikan hasil yang lebih konsisten.

    - **Lasso Regression**: Dalam perhitungan ini, kamu mungkin menyadari bahwa beberapa nilai tidak terlalu penting untuk menentukan Passing Grade. Jadi, kamu memutuskan untuk hanya fokus pada nilai-nilai yang paling relevan dan mengabaikan yang tidak perlu. Lasso membantu kamu memilih fitur yang paling penting dan mengurangi beberapa koefisien menjadi nol, sehingga hasilnya lebih sederhana.

    - **Random Forest Regressor**: Ini seperti meminta pendapat dari sekelompok teman tentang nilai Passing Grade yang seharusnya. Setiap teman memberikan saran berdasarkan nilai RATAAN dan S.BAKU mereka, dan kamu menggabungkan semua saran tersebut untuk mendapatkan prediksi yang lebih akurat. Model ini menggabungkan banyak "suara" dari pohon keputusan untuk memberikan hasil yang lebih baik.

    - **Gradient Boosting Regressor**: Bayangkan kamu belajar dari setiap perhitungan yang kamu lakukan. Setiap kali kamu membuat kesalahan dalam menentukan Passing Grade, kamu mencatatnya dan berusaha untuk tidak mengulanginya di perhitungan berikutnya. Model ini membangun pohon keputusan satu per satu, di mana setiap pohon baru berusaha memperbaiki kesalahan dari pohon sebelumnya.

    Dengan cara ini, setiap model regresi memiliki pendekatan unik untuk membantu memastikan bahwa prediksi Passing Grade yang dihasilkan adalah yang terbaik.
    """)

    if st.button('Latih Model'):
        df = load_data()
        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(df[['RATAAN', 'S.BAKU']])
        y = df['MIN']
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        mse_results = {}
        best_model_name = None
        best_model = None
        best_mse = float('inf')

        with st.spinner('Melatih model...'):
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_results[model_name] = mse

                if mse < best_mse:
                    best_mse = mse
                    best_model_name = model_name
                    best_model = model

        st.write("\n### MSE Results:")
        cols = st.columns(len(models))
        for i, (model_name, mse) in enumerate(mse_results.items()):
            with cols[i]:
                st.metric(label=model_name, value=f"{mse:.2f}", delta=None)

        if best_model is not None:
            with open('passing_grade_model.sav', 'wb') as f:
                pickle.dump(best_model, f)
            with open('polynomial_features.sav', 'wb') as f:
                pickle.dump(poly, f)
            st.success(f"Model terbaik '{best_model_name}' berhasil dilatih dan disimpan.")
        else:
            st.error("Tidak ada model yang berhasil dilatih.")
