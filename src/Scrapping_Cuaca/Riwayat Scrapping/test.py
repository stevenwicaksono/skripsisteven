import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

raw = """Aceh,2018,329515.78,56.49,1861567.10
Aceh,2019,310012.46,55.30,1714437.60
Aceh,2020,317869.41,55.28,1757313.07
Aceh,2021,297058.38,55.03,1634639.60
Aceh,2022,271750.20,55.55,1509456.00
Aceh,2023,254287.38,55.22,1404234.82
Aceh,2024,301196.35,55.11,1659966.28
Bali,2018,110978.37,60.11,667069.06
Bali,2019,95319.34,60.78,579320.53
Bali,2020,90980.69,58.49,532168.45
Bali,2021,105201.31,58.83,618910.81
Bali,2022,112320.60,60.59,680601.60
Bali,2023,108514.06,62.07,673580.65
Bali,2024,103803.93,61.22,635473.35
Banten,2018,344836.06,48.94,1687783.30
Banten,2019,303731.80,48.41,1470503.35
Banten,2020,325333.24,50.88,1655170.09
Banten,2021,318248.46,50.38,1603247.00
Banten,2022,337240.70,53.04,1788583.00
Banten,2023,311199.73,54.19,1686483.29
Banten,2024,299090.79,51.84,1550623.46
Bengkulu,2018,65891.16,43.83,288810.52
Bengkulu,2019,64406.86,46.03,296472.07
Bengkulu,2020,64137.28,45.66,292834.04
Bengkulu,2021,55704.69,48.67,271117.19
Bengkulu,2022,57151.84,49.27,281610.10
Bengkulu,2023,57877.18,49.53,286684.43
Bengkulu,2024,55775.09,48.92,272848.55
DI Yogyakarta,2018,93956.45,54.81,514935.49
DI Yogyakarta,2019,111477.36,47.86,533477.40
DI Yogyakarta,2020,110548.12,47.35,523395.95
DI Yogyakarta,2021,107506.16,51.77,556531.03
DI Yogyakarta,2022,110927.20,50.64,561699.50
DI Yogyakarta,2023,105693.66,50.53,534113.69
DI Yogyakarta,2024,96976.13,46.70,452831.77
DKI Jakarta,2018,673.37,72.76,4899.14
DKI Jakarta,2019,622.59,53.96,3359.31
DKI Jakarta,2020,914.51,49.69,4543.93
DKI Jakarta,2021,559.97,58.03,3249.47
DKI Jakarta,2022,477.25,48.98,2337.77
DKI Jakarta,2023,542.93,49.26,2674.28
DKI Jakarta,2024,498.31,46.29,2306.54
Gorontalo,2018,56631.64,47.60,269540.40
Gorontalo,2019,49009.95,47.18,231211.11
Gorontalo,2020,48686.34,46.75,227627.20
Gorontalo,2021,48713.50,48.12,234392.86
Gorontalo,2022,46823.47,51.29,240134.50
Gorontalo,2023,49610.47,50.68,251431.76
Gorontalo,2024,46952.15,50.02,234862.88
Indonesia,2018,11377934.44,52.03,59200533.72
Indonesia,2019,10677887.15,51.14,54604033.34
Indonesia,2020,10657274.96,51.28,54649202.24
Indonesia,2021,10411801.22,52.26,54415294.22
Indonesia,2022,10452672.00,52.38,54748977.00
Indonesia,2023,10213705.17,52.85,53980993.19
Indonesia,2024,10046135.36,52.90,53142726.65
Jambi,2018,86202.68,44.44,383045.74
Jambi,2019,69536.06,44.57,309932.68
Jambi,2020,84772.93,45.58,386413.49
Jambi,2021,64412.26,46.29,298149.25
Jambi,2022,60539.59,45.88,277743.80
Jambi,2023,61236.64,45.06,275941.45
Jambi,2024,61625.68,45.60,281022.05
Jawa Barat,2018,1707253.81,56.51,9647358.75
Jawa Barat,2019,1578835.70,57.54,9084957.22
Jawa Barat,2020,1586888.63,56.82,9016772.58
Jawa Barat,2021,1604109.31,56.81,9113573.08
Jawa Barat,2022,1662404.00,56.75,9433723.00
Jawa Barat,2023,1583656.28,57.71,9140039.20
Jawa Barat,2024,1475362.09,58.47,8626879.91
Jawa Tengah,2018,1821983.17,57.63,10499588.23
Jawa Tengah,2019,1678479.21,57.53,9655653.98
Jawa Tengah,2020,1666931.49,56.93,9489164.62
Jawa Tengah,2021,1696712.36,56.69,9618656.81
Jawa Tengah,2022,1688670.00,55.41,9356445.00
Jawa Tengah,2023,1642761.23,55.30,9084107.53
Jawa Tengah,2024,1554777.14,57.19,8891297.05
Jawa Timur,2018,1751191.67,58.26,10203213.17
Jawa Timur,2019,1702426.36,56.28,9580933.88
Jawa Timur,2020,1754380.30,56.68,9944538.26
Jawa Timur,2021,1747481.20,56.02,9789587.67
Jawa Timur,2022,1693211.00,56.26,9526516.00
Jawa Timur,2023,1698083.31,57.19,9710661.33
Jawa Timur,2024,1616985.05,57.33,9270435.29
Kalimantan Barat,2018,286476.03,27.92,799715.21
Kalimantan Barat,2019,290048.44,29.23,847875.13
Kalimantan Barat,2020,256575.43,30.33,778170.36
Kalimantan Barat,2021,223165.74,31.90,711898.01
Kalimantan Barat,2022,241478.60,30.28,731225.80
Kalimantan Barat,2023,224068.52,31.25,700290.80
Kalimantan Barat,2024,247207.72,30.94,764784.15
Kalimantan Selatan,2018,323091.21,41.09,1327492.41
Kalimantan Selatan,2019,356245.95,37.69,1342861.82
Kalimantan Selatan,2020,289836.35,39.69,1150306.66
Kalimantan Selatan,2021,254263.59,39.97,1016313.55
Kalimantan Selatan,2022,214908.90,38.13,819419.20
Kalimantan Selatan,2023,214283.82,40.86,875545.73
Kalimantan Selatan,2024,246112.42,41.83,1029567.93
Kalimantan Tengah,2018,147571.69,34.88,514769.05
Kalimantan Tengah,2019,146144.51,30.35,443561.33
Kalimantan Tengah,2020,143275.05,31.96,457952.00
Kalimantan Tengah,2021,125870.05,30.28,381189.55
Kalimantan Tengah,2022,108226.80,31.78,343918.80
Kalimantan Tengah,2023,101580.30,32.56,330781.05
Kalimantan Tengah,2024,111016.13,32.98,366146.82
Kalimantan Timur,2018,64961.16,40.45,262773.88
Kalimantan Timur,2019,69707.75,36.41,253818.37
Kalimantan Timur,2020,73568.44,35.67,262434.52
Kalimantan Timur,2021,66269.46,36.92,244677.96
Kalimantan Timur,2022,64970.01,36.85,239425.30
Kalimantan Timur,2023,57082.01,39.76,226972.07
Kalimantan Timur,2024,63041.65,39.60,249642.90
Kalimantan Utara,2018,13707.00,32.88,45063.53
Kalimantan Utara,2019,10294.70,32.40,33357.19
Kalimantan Utara,2020,9883.05,33.97,33574.28
Kalimantan Utara,2021,8880.83,33.74,29967.31
Kalimantan Utara,2022,8604.19,35.49,30533.59
Kalimantan Utara,2023,6499.92,36.31,23602.00
Kalimantan Utara,2024,8282.06,36.32,30079.77
Kep. Bangka Belitung,2018,17233.59,26.53,45724.69
Kep. Bangka Belitung,2019,17087.81,28.56,48805.68
Kep. Bangka Belitung,2020,17840.55,32.13,57324.32
Kep. Bangka Belitung,2021,18278.27,38.57,70496.25
Kep. Bangka Belitung,2022,15107.80,40.66,61425.07
Kep. Bangka Belitung,2023,15284.56,43.49,66468.89
Kep. Bangka Belitung,2024,18202.56,42.57,77489.79
Kep. Riau,2018,375.87,29.19,1097.00
Kep. Riau,2019,356.27,32.30,1150.80
Kep. Riau,2020,298.52,28.56,852.54
Kep. Riau,2021,270.16,31.65,855.01
Kep. Riau,2022,179.48,28.24,506.91
Kep. Riau,2023,115.27,28.11,324.01
Kep. Riau,2024,113.33,26.92,305.09
Lampung,2018,511940.93,48.61,2488641.91
Lampung,2019,464103.42,46.63,2164089.33
Lampung,2020,545149.05,48.62,2650289.64
Lampung,2021,489573.23,50.77,2485452.78
Lampung,2022,518256.10,51.87,2688160.00
Lampung,2023,530108.09,52.03,2757898.19
Lampung,2024,531715.12,52.50,2791347.53
Maluku,2018,29052.14,40.01,116228.86
Maluku,2019,25976.85,37.82,98254.75
Maluku,2020,28668.22,38.53,110447.30
Maluku,2021,28319.75,41.24,116803.67
Maluku,2022,23987.82,38.60,92601.06
Maluku,2023,22636.68,35.32,79958.34
Maluku,2024,23947.35,38.05,91125.35
Maluku Utara,2018,13412.75,36.57,49047.11
Maluku Utara,2019,11700.50,32.43,37945.64
Maluku Utara,2020,10301.91,42.11,43382.85
Maluku Utara,2021,7781.96,36.05,28050.80
Maluku Utara,2022,6416.45,38.16,24486.03
Maluku Utara,2023,7709.07,34.59,26663.23
Maluku Utara,2024,9366.71,33.34,31232.95
Nusa Tenggara Barat,2018,289242.59,50.49,1460338.81
Nusa Tenggara Barat,2019,281666.04,49.78,1402182.39
Nusa Tenggara Barat,2020,273460.82,48.17,1317189.81
Nusa Tenggara Barat,2021,276211.88,51.39,1419559.84
Nusa Tenggara Barat,2022,270092.90,53.79,1452945.00
Nusa Tenggara Barat,2023,287512.14,53.51,1538536.92
Nusa Tenggara Barat,2024,281717.84,51.59,1453408.37
Nusa Tenggara Timur,2018,218232.91,41.24,899935.88
Nusa Tenggara Timur,2019,198867.41,40.82,811724.18
Nusa Tenggara Timur,2020,181690.63,39.90,725024.30
Nusa Tenggara Timur,2021,174900.07,41.85,731877.74
Nusa Tenggara Timur,2022,183092.00,41.29,756049.90
Nusa Tenggara Timur,2023,184698.89,41.52,766810.46
Nusa Tenggara Timur,2024,168727.24,41.95,707792.54
Papua,2018,52411.95,42.57,223119.42
Papua,2019,54131.72,43.48,235339.51
Papua,2020,52727.52,31.48,166002.30
Papua,2021,64984.90,44.05,286279.80
Papua,2022,49741.91,38.99,193943.50
Papua,2023,840.18,44.76,3760.45
Papua,2024,1068.57,43.14,4609.95
Papua Barat,2018,7767.01,32.15,24967.13
Papua Barat,2019,7192.15,41.63,29943.56
Papua Barat,2020,7570.63,32.20,24378.33
Papua Barat,2021,6414.94,41.98,26926.93
Papua Barat,2022,5460.59,43.89,23963.92
Papua Barat,2023,5006.27,45.08,22566.81
Papua Barat,2024,5121.13,40.48,20729.15
Papua Barat Daya,2023,579.83,41.34,2396.95
Papua Barat Daya,2024,363.87,27.17,988.64
Papua Pegunungan,2023,14.14,43.59,61.63
Papua Pegunungan,2024,9.66,43.87,42.38
Papua Selatan,2023,44807.86,40.98,183627.83
Papua Selatan,2024,47168.57,46.17,217789.62
Papua Tengah,2023,2094.20,44.28,9273.22
Papua Tengah,2024,1436.12,42.28,6072.38
Riau,2018,71448.08,37.28,266375.53
Riau,2019,63142.04,36.56,230873.97
Riau,2020,64733.13,37.64,243685.04
Riau,2021,53062.35,40.98,217458.87
Riau,2022,51054.04,41.83,213557.20
Riau,2023,51914.14,39.68,205972.55
Riau,2024,56421.96,39.36,222055.71
Sulawesi Barat,2018,65303.78,48.46,316478.37
Sulawesi Barat,2019,62581.47,47.96,300142.22
Sulawesi Barat,2020,64826.18,53.23,345050.37
Sulawesi Barat,2021,59763.18,52.05,311072.46
Sulawesi Barat,2022,69323.95,50.99,353513.30
Sulawesi Barat,2023,58606.67,49.73,291458.59
Sulawesi Barat,2024,63181.59,50.47,318876.59
Sulawesi Selatan,2018,1185484.10,50.21,5952616.45
Sulawesi Selatan,2019,1010188.75,50.03,5054166.96
Sulawesi Selatan,2020,976258.14,48.23,4708464.97
Sulawesi Selatan,2021,985158.23,51.67,5090637.23
Sulawesi Selatan,2022,1038084.00,51.64,5360169.00
Sulawesi Selatan,2023,967790.21,50.39,4876386.11
Sulawesi Selatan,2024,951308.60,50.65,4818429.39
Sulawesi Tengah,2018,201279.24,46.05,926978.66
Sulawesi Tengah,2019,186100.44,45.40,844904.30
Sulawesi Tengah,2020,178066.94,44.49,792248.84
Sulawesi Tengah,2021,182186.62,47.59,867012.77
Sulawesi Tengah,2022,168993.20,44.05,744408.70
Sulawesi Tengah,2023,177699.03,46.22,821367.41
Sulawesi Tengah,2024,172606.22,44.14,761936.39
Sulawesi Tenggara,2018,136673.75,39.43,538876.14
Sulawesi Tenggara,2019,132343.86,39.27,519706.93
Sulawesi Tenggara,2020,133697.15,39.85,532773.49
Sulawesi Tenggara,2021,127517.29,41.57,530029.08
Sulawesi Tenggara,2022,118258.80,40.50,478958.00
Sulawesi Tenggara,2023,113930.26,42.08,479407.25
Sulawesi Tenggara,2024,129999.61,42.76,555836.08
Sulawesi Utara,2018,70352.62,46.47,326929.74
Sulawesi Utara,2019,62020.39,44.79,277776.31
Sulawesi Utara,2020,61827.86,40.25,248879.48
Sulawesi Utara,2021,59182.52,39.35,232884.76
Sulawesi Utara,2022,58195.56,41.88,243730.30
Sulawesi Utara,2023,54562.95,43.65,238193.41
Sulawesi Utara,2024,59121.96,46.20,273134.94
Sumatra Barat,2018,313050.82,47.37,1483076.48
Sumatra Barat,2019,311671.23,47.58,1482996.01
Sumatra Barat,2020,295664.47,46.92,1387269.29
Sumatra Barat,2021,272391.95,48.36,1317209.38
Sumatra Barat,2022,271883.10,50.52,1373532.00
Sumatra Barat,2023,300564.77,49.32,1482468.79
Sumatra Barat,2024,295278.98,45.94,1356467.93
Sumatra Selatan,2018,581574.61,51.48,2994191.84
Sumatra Selatan,2019,539316.52,48.27,2603396.24
Sumatra Selatan,2020,551320.76,49.75,2743059.68
Sumatra Selatan,2021,496241.65,51.44,2552443.19
Sumatra Selatan,2022,513378.20,54.06,2775069.00
Sumatra Selatan,2023,504143.03,56.19,2832773.92
Sumatra Selatan,2024,521092.21,55.83,2909411.67
Sumatra Utara,2018,408176.45,51.65,2108284.72
Sumatra Utara,2019,413141.24,50.32,2078901.59
Sumatra Utara,2020,388591.22,52.51,2040500.19
Sumatra Utara,2021,385405.00,52.00,2004142.51
Sumatra Utara,2022,411462.10,50.76,2088584.00
Sumatra Utara,2023,406109.49,51.40,2087474.15
Sumatra Utara,2024,419463.48,52.56,2204875.51"""

# Parse rows
rows = []
for line in raw.strip().split('\n'):
    parts = line.split(',')
    if len(parts) == 5:
        try:
            rows.append([parts[0], int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        except:
            pass

# Style helpers
HEADER_FILL   = PatternFill("solid", fgColor="1F4E79")
SUBHEAD_FILL  = PatternFill("solid", fgColor="2E75B6")
ALT_FILL      = PatternFill("solid", fgColor="D6E4F0")
TOTAL_FILL    = PatternFill("solid", fgColor="FFF2CC")
WHITE_FILL    = PatternFill("solid", fgColor="FFFFFF")
HEADER_FONT   = Font(name="Arial", bold=True, color="FFFFFF", size=11)
SUBHEAD_FONT  = Font(name="Arial", bold=True, color="FFFFFF", size=10)
BODY_FONT     = Font(name="Arial", size=10)
BOLD_FONT     = Font(name="Arial", bold=True, size=10)
TOTAL_FONT    = Font(name="Arial", bold=True, size=10, color="7B3F00")
CENTER        = Alignment(horizontal="center", vertical="center")
LEFT          = Alignment(horizontal="left",   vertical="center")
RIGHT         = Alignment(horizontal="right",  vertical="center")

thin = Side(style="thin", color="B0B0B0")
thick= Side(style="medium", color="1F4E79")
THIN_BORDER   = Border(left=thin, right=thin, top=thin, bottom=thin)
THICK_BORDER  = Border(left=thick, right=thick, top=thick, bottom=thick)

def style_header(cell, fill=HEADER_FILL, font=HEADER_FONT):
    cell.fill = fill; cell.font = font
    cell.alignment = CENTER; cell.border = THIN_BORDER

def style_body(cell, align=LEFT, alt=False):
    cell.font = BODY_FONT; cell.alignment = align
    cell.fill = ALT_FILL if alt else WHITE_FILL
    cell.border = THIN_BORDER

# Format angka Indonesia (titik sebagai ribuan, koma sebagai desimal)
def format_indo(num, decimals=2):
    """Format angka ke format Indonesia: 1.234.567,89"""
    if num is None:
        return "-"
    # Format dengan 2 desimal terlebih dahulu
    formatted = f"{num:,.{decimals}f}"
    # Ganti koma dengan placeholder, titik dengan koma, placeholder dengan titik
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return formatted

wb = openpyxl.Workbook()

# ── SHEET 1: Data Lengkap ────────────────────────────────────────────────────
ws1 = wb.active
ws1.title = "Data Lengkap"

# Title
ws1.merge_cells("A1:E1")
t = ws1["A1"]
t.value = "DATA PRODUKSI PADI INDONESIA PER PROVINSI (2018–2024)"
t.font  = Font(name="Arial", bold=True, size=13, color="FFFFFF")
t.fill  = HEADER_FILL
t.alignment = CENTER
ws1.row_dimensions[1].height = 30

# Sub-title
ws1.merge_cells("A2:E2")
s = ws1["A2"]
s.value = "Sumber: Badan Pusat Statistik (BPS) Indonesia"
s.font  = Font(name="Arial", italic=True, size=9, color="FFFFFF")
s.fill  = SUBHEAD_FILL
s.alignment = CENTER

# Headers
headers = ["Provinsi", "Tahun", "Luas Panen (ha)", "Produktivitas (ku/ha)", "Produksi (ton)"]
for col, h in enumerate(headers, 1):
    c = ws1.cell(row=3, column=col, value=h)
    style_header(c)
ws1.row_dimensions[3].height = 22

# Data rows
prev_prov = ""
prov_list = list(dict.fromkeys([r[0] for r in rows]))

for r_idx, row in enumerate(rows, 4):
    prov_rank = prov_list.index(row[0])
    use_alt   = (prov_rank % 2 == 1)

    for col_idx, val in enumerate(row, 1):
        if col_idx >= 3:  # Kolom numerik
            formatted_val = format_indo(val)
            c = ws1.cell(row=r_idx, column=col_idx, value=formatted_val)
        else:
            c = ws1.cell(row=r_idx, column=col_idx, value=val)
        
        if col_idx == 1:
            style_body(c, LEFT, use_alt)
            if val != prev_prov:
                c.font = Font(name="Arial", bold=True, size=10)
        elif col_idx == 2:
            style_body(c, CENTER, use_alt)
        else:
            style_body(c, RIGHT, use_alt)

    prev_prov = row[0]

# Column widths
ws1.column_dimensions['A'].width = 26
ws1.column_dimensions['B'].width = 8
ws1.column_dimensions['C'].width = 22
ws1.column_dimensions['D'].width = 26
ws1.column_dimensions['E'].width = 22
ws1.freeze_panes = "A4"

# ── SHEET 2: Ringkasan per Provinsi ─────────────────────────────────────────
ws2 = wb.create_sheet("Ringkasan per Provinsi")

import statistics
provinces = list(dict.fromkeys([r[0] for r in rows]))

ws2.merge_cells("A1:G1")
t2 = ws2["A1"]
t2.value = "RINGKASAN PRODUKSI PADI PER PROVINSI (RATA-RATA 2018–2024)"
t2.font  = Font(name="Arial", bold=True, size=13, color="FFFFFF")
t2.fill  = HEADER_FILL; t2.alignment = CENTER
ws2.row_dimensions[1].height = 28

ws2.merge_cells("A2:G2")
s2 = ws2["A2"]
s2.value = "Nilai rata-rata selama periode yang tersedia"
s2.font  = Font(name="Arial", italic=True, size=9, color="FFFFFF")
s2.fill  = SUBHEAD_FILL; s2.alignment = CENTER

heads2 = ["Provinsi", "Tahun Tersedia", "Rata-rata Luas Panen (ha)",
          "Rata-rata Produktivitas (ku/ha)", "Rata-rata Produksi (ton)",
          "Produksi Tertinggi (ton)", "Produksi Terendah (ton)"]
for col, h in enumerate(heads2, 1):
    c = ws2.cell(row=3, column=col, value=h)
    style_header(c)
ws2.row_dimensions[3].height = 36

for r_idx, prov in enumerate(provinces, 4):
    prov_rows = [r for r in rows if r[0] == prov]
    tahun_str = f"{min(r[1] for r in prov_rows)}–{max(r[1] for r in prov_rows)}"
    avg_lp  = sum(r[2] for r in prov_rows) / len(prov_rows)
    avg_pr  = sum(r[3] for r in prov_rows) / len(prov_rows)
    avg_prod= sum(r[4] for r in prov_rows) / len(prov_rows)
    max_prod= max(r[4] for r in prov_rows)
    min_prod= min(r[4] for r in prov_rows)
    use_alt = (r_idx % 2 == 0)

    vals = [prov, tahun_str, avg_lp, avg_pr, avg_prod, max_prod, min_prod]
    for col_idx, val in enumerate(vals, 1):
        if col_idx >= 3:  # Kolom numerik
            display_val = format_indo(val)
        else:
            display_val = val
            
        c = ws2.cell(row=r_idx, column=col_idx, value=display_val)
        c.border = THIN_BORDER
        c.fill   = ALT_FILL if use_alt else WHITE_FILL
        if col_idx == 1:
            c.font = BOLD_FONT; c.alignment = LEFT
        elif col_idx == 2:
            c.font = BODY_FONT; c.alignment = CENTER
        else:
            c.font = BODY_FONT; c.alignment = RIGHT

ws2.column_dimensions['A'].width = 26
ws2.column_dimensions['B'].width = 16
ws2.column_dimensions['C'].width = 26
ws2.column_dimensions['D'].width = 30
ws2.column_dimensions['E'].width = 26
ws2.column_dimensions['F'].width = 26
ws2.column_dimensions['G'].width = 26
ws2.freeze_panes = "A4"

# ── SHEET 3: Tren Tahunan (pivot) ────────────────────────────────────────────
ws3 = wb.create_sheet("Tren Tahunan")

years = sorted(set(r[1] for r in rows))

ws3.merge_cells(f"A1:{get_column_letter(1+len(years))}1")
t3 = ws3["A1"]
t3.value = "TREN PRODUKSI PADI (ton) PER PROVINSI — PIVOT TAHUNAN"
t3.font  = Font(name="Arial", bold=True, size=13, color="FFFFFF")
t3.fill  = HEADER_FILL; t3.alignment = CENTER
ws3.row_dimensions[1].height = 28

# Header row
ws3.cell(row=2, column=1, value="Provinsi")
style_header(ws3.cell(row=2, column=1))
for c_idx, yr in enumerate(years, 2):
    cell = ws3.cell(row=2, column=c_idx, value=yr)
    style_header(cell)
ws3.row_dimensions[2].height = 20

# Build pivot dict
pivot = {}
for r in rows:
    if r[0] not in pivot:
        pivot[r[0]] = {}
    pivot[r[0]][r[1]] = r[4]

for r_idx, prov in enumerate(provinces, 3):
    use_alt = (r_idx % 2 == 0)
    c0 = ws3.cell(row=r_idx, column=1, value=prov)
    c0.font = BOLD_FONT; c0.alignment = LEFT
    c0.fill = ALT_FILL if use_alt else WHITE_FILL
    c0.border = THIN_BORDER
    for c_idx, yr in enumerate(years, 2):
        val = pivot.get(prov, {}).get(yr, None)
        if val is not None:
            display_val = format_indo(val)
            c = ws3.cell(row=r_idx, column=c_idx, value=display_val)
            c.alignment = RIGHT
        else:
            c = ws3.cell(row=r_idx, column=c_idx, value='-')
            c.alignment = CENTER
        
        c.font = BODY_FONT
        c.fill = ALT_FILL if use_alt else WHITE_FILL
        c.border = THIN_BORDER

ws3.column_dimensions['A'].width = 26
for c_idx in range(2, 2+len(years)):
    ws3.column_dimensions[get_column_letter(c_idx)].width = 18
ws3.freeze_panes = "B3"

out = 'Data_Padi_Indonesia.xlsx'
wb.save(out)
print('Saved:', out)