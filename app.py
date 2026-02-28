import streamlit as st
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime
import re


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(page_title="EggRow AI System", layout="wide")
st.title("🐔 EggRow - Smart Poultry Decision Support System")

os.makedirs("database", exist_ok=True)
db_path = "database/produktivitias_db.csv"

# =====================================================
# GEMINI CONFIG
# =====================================================

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# =====================================================
# NILAI STANDAR GLOBAL
# =====================================================

HDP_OPTIMAL = (90, 96)
HDP_ALERT = 85

FCR_OPTIMAL = (1.9, 2.2)
FCR_ALERT = 2.3

# =====================================================
# HELPER FUNCTION
# =====================================================

def format_rupiah(value):
    return f"Rp {value:,.0f}"

def clean_numeric(x):
    x = re.sub(r"[^\d.]", "", str(x))
    return pd.to_numeric(x, errors="coerce")

# =====================================================
# SIDEBAR
# =====================================================

menu = st.sidebar.selectbox(
    "Menu",
    ["Dashboard", "Produktivitas", "Visualisasi Data", "Summary"]
)

# =====================================================
# DASHBOARD
# =====================================================

if menu == "Dashboard":

    st.header("📊 Dashboard Monitoring")

    uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()

        numeric_cols = [
            "jumlah ternak", "jumlah telur",
            "berat telur rata-rata", "konsumsi pakan",
            "harga pakan", "harga telur",
            "suhu", "kelembapan"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric)

        df = df.dropna()

        st.session_state["dashboard_df"] = df

        st.dataframe(df)

        if "suhu" in df.columns and "kelembapan" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Grafik Suhu")
                st.line_chart(df.set_index("tanggal")["suhu"])
            with col2:
                st.subheader("Grafik Kelembapan")
                st.line_chart(df.set_index("tanggal")["kelembapan"])

        selected_date = st.selectbox("Pilih Tanggal", df["tanggal"])
        row = df[df["tanggal"] == selected_date].iloc[0]

        jumlah_ternak = row["jumlah ternak"]
        jumlah_telur = row["jumlah telur"]
        berat_telur = row["berat telur rata-rata"]
        konsumsi_pakan = row["konsumsi pakan"]
        harga_pakan = row["harga pakan"]
        harga_telur = row["harga telur"]

        if jumlah_telur > 0 and berat_telur > 0:

            fcr = konsumsi_pakan / (jumlah_telur * berat_telur)
            hdp = (jumlah_telur / jumlah_ternak) * 100
            feed_cost = (konsumsi_pakan * harga_pakan) / (jumlah_telur * berat_telur)
            revenue = (jumlah_telur * berat_telur) * harga_telur
            total_feed_cost = konsumsi_pakan * harga_pakan
            profit = revenue - total_feed_cost

            col1, col2, col3 = st.columns(3)
            col1.metric("FCR", round(fcr, 3))
            col2.metric("HDP (%)", round(hdp, 2))
            col3.metric("Feed Cost / kg egg", format_rupiah(feed_cost))
            st.metric("Profit", format_rupiah(profit))

            # Warning system
            if hdp < HDP_ALERT:
                st.error("⚠ HDP di bawah standar!")
            elif HDP_OPTIMAL[0] <= hdp <= HDP_OPTIMAL[1]:
                st.success("✅ HDP optimal")

            if fcr > FCR_ALERT:
                st.error("⚠ FCR melebihi standar!")
            elif FCR_OPTIMAL[0] <= fcr <= FCR_OPTIMAL[1]:
                st.success("✅ FCR optimal")

            

            if st.button("🤖 Analisis AI Dashboard"):
                prompt = f"""
                FCR: {fcr}
                HDP: {hdp}
                Feed Cost: {format_rupiah(feed_cost)}
                Profit: {format_rupiah(profit)}

                Standar:
                HDP optimal 90-96 (alert <85)
                FCR optimal 1.9-2.2 (alert >2.3)

                Berikan analisis profesional.
                """
                response = model.generate_content(prompt)
                st.write(response.text)

            if st.button("💾 Simpan ke Database"):

                save_df = pd.DataFrame({
                    "tanggal": [selected_date],
                    "FCR": [fcr],
                    "HDP": [hdp],
                    "Feed Cost": [feed_cost],
                    "Profit": [profit]
                })

                if os.path.exists(db_path):
                    old = pd.read_csv(db_path)
                    save_df = pd.concat([old, save_df], ignore_index=True)

                save_df.to_csv(db_path, index=False)
                st.success("Data tersimpan.")

# =====================================================
# PRODUKTIVITAS
# =====================================================

elif menu == "Produktivitas":

    st.header("📈 Database Produktivitas")

    if os.path.exists(db_path):

        db = pd.read_csv(db_path)
        st.dataframe(db)

        avg_hdp = db["HDP"].mean() if "HDP" in db.columns else 0
        avg_fcr = db["FCR"].mean() if "FCR" in db.columns else 0
        total_profit = db["Profit"].sum() if "Profit" in db.columns else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Rata-rata HDP (%)", round(avg_hdp, 2))
        col2.metric("Rata-rata FCR", round(avg_fcr, 3))
        col3.metric("Total Profit", format_rupiah(total_profit))

        if avg_hdp < HDP_ALERT:
            st.error("⚠ Rata-rata HDP di bawah standar!")
        if avg_fcr > FCR_ALERT:
            st.error("⚠ Rata-rata FCR melebihi standar!")

        if st.button("🗑 Hapus Semua Data"):
            os.remove(db_path)
            st.success("Database berhasil dihapus.")
            st.rerun()

        st.subheader("🤖 Konsultasi AI")

        user_input = st.text_area("Tulis pertanyaan")

        if st.button("Generate Jawaban AI"):

            context_dashboard = ""
            if "dashboard_df" in st.session_state:
                context_dashboard = st.session_state["dashboard_df"].to_string()

            prompt = f"""
            DATA DASHBOARD:
            {context_dashboard}

            DATABASE PRODUKTIVITAS:
            {db.to_string()}

            Standar:
            HDP optimal 90-96 (alert <85)
            FCR optimal 1.9-2.2 (alert >2.3)

            Pertanyaan:
            {user_input}

            Berikan analisis dan solusi profesional.
            """

            response = model.generate_content(prompt)
            st.write(response.text)

    else:
        st.warning("Database kosong.")

# =====================================================
# VISUALISASI
# =====================================================

elif menu == "Visualisasi Data":

    st.header("📊 Visualisasi Data")

    if os.path.exists(db_path):

        db = pd.read_csv(db_path)

        st.scatter_chart(db.set_index("tanggal")["FCR"])
        st.scatter_chart(db.set_index("tanggal")["HDP"])
        st.scatter_chart(db.set_index("tanggal")["Feed Cost"])

        corr = db.corr(numeric_only=True)

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Database kosong.")

# =====================================================
# SUMMARY
# =====================================================

elif menu == "Summary":

    st.header("📋 Executive AI Summary")

    if os.path.exists(db_path):

        db = pd.read_csv(db_path)

        avg_fcr = db["FCR"].mean()
        avg_hdp = db["HDP"].mean()
        total_profit = db["Profit"].sum()

        st.metric("Rata-rata FCR", round(avg_fcr, 3))
        st.metric("Rata-rata HDP", round(avg_hdp, 2))
        st.metric("Total Profit", format_rupiah(total_profit))

        if st.button("🤖 Generate Executive Report"):

            prompt = f"""
            Rata-rata FCR: {avg_fcr}
            Rata-rata HDP: {avg_hdp}
            Total Profit: {format_rupiah(total_profit)}

            Standar:
            HDP optimal 90-96
            FCR optimal 1.9-2.2

            Buat laporan profesional tanpa identitas.
            """

            response = model.generate_content(prompt)
            report_text = response.text
            st.write(report_text)

            # =============================
            # FUNGSI FOOTER
            # =============================
            def markdown_to_html(text):
                # Convert **bold** menjadi <b>bold</b>
                text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
                return text
            
            def add_footer(canvas, doc):
                canvas.saveState()
                footer_text = f"EggRow AI System | Generated: {datetime.now().strftime('%d-%m-%Y')} | Page {doc.page}"
                canvas.setFont("Helvetica", 8)
                canvas.setFillColor(colors.grey)
                canvas.drawCentredString(A4[0] / 2, 0.5 * inch, footer_text)
                canvas.restoreState()

            # =============================
            # GENERATE PDF
            # =============================

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=40,
                leftMargin=40,
                topMargin=60,
                bottomMargin=60
            )

            elements = []
            styles = getSampleStyleSheet()

            # ===== CUSTOM STYLES =====

            title_style = ParagraphStyle(
                name="TitleStyle",
                parent=styles["Title"],
                alignment=TA_CENTER,
                fontSize=18,
                spaceAfter=20
            )

            heading_style = ParagraphStyle(
                name="HeadingStyle",
                parent=styles["Heading2"],
                fontSize=14,
                spaceBefore=12,
                spaceAfter=6
            )

            subheading_style = ParagraphStyle(
                name="SubHeadingStyle",
                parent=styles["Heading3"],
                fontSize=12,
                spaceBefore=8,
                spaceAfter=4
            )

            body_style = ParagraphStyle(
                name="BodyStyle",
                parent=styles["Normal"],
                alignment=TA_JUSTIFY,
                fontSize=11,
                leading=16,
                spaceAfter=8
            )

            # ===== TITLE =====

            elements.append(Paragraph("LAPORAN EXECUTIVE AYAM LAYER", title_style))
            elements.append(HRFlowable(width="100%", thickness=1, color=colors.black))
            elements.append(Spacer(1, 0.3 * inch))

            # ===== FORMAT TEKS OTOMATIS =====

            lines = report_text.split("\n")

            for line in lines:
                line = line.strip()

                if not line:
                    continue

                # 1️⃣ Convert semua markdown bold terlebih dahulu
                line = markdown_to_html(line)

                # =============================
                # BULLET LIST
                # =============================
                if line.startswith("•") or line.startswith("*"):

                    # Hapus simbol bullet asli di depan
                    clean_line = line.lstrip("*• ").strip()

                    elements.append(Paragraph(f"• {clean_line}", body_style))

                # =============================
                # NUMBERED LIST (1. 2. 3.)
                # =============================
                elif line[0].isdigit() and "." in line:
                    elements.append(Paragraph(f"<b>{line}</b>", subheading_style))

                # =============================
                # HEADING (jika satu baris full bold)
                # =============================
                elif line.startswith("<b>") and line.endswith("</b>"):
                    elements.append(Paragraph(line, heading_style))

                # =============================
                # NORMAL TEXT
                # =============================
                else:
                    elements.append(Paragraph(line, body_style))

                elements.append(Spacer(1, 0.1 * inch))

            # ===== BUILD DOCUMENT =====

            doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)

            st.download_button(
                "📥 Download PDF",
                data=buffer.getvalue(),
                file_name="laporan_ayam_layer_profesional.pdf",
                mime="application/pdf"
            )

    else:
        st.warning("Database kosong.")
